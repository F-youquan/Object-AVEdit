
import xformers.ops
import torch
import math
import sys
import torch.nn.functional as F

def register_attention_control(model, controller):
    def ca_forward_sa(self):
        def forward(x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            enable_flash_attn = False
            qkv = self.qkv(x)
            qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

            qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if self.qk_norm_legacy:
                if self.rope:
                    q = self.rotary_emb(q)
                    k = self.rotary_emb(k)
                q, k = self.q_norm(q), self.k_norm(k)
            else:
                q, k = self.q_norm(q), self.k_norm(k)
                if self.rope:
                    q = self.rotary_emb(q)
                    k = self.rotary_emb(k)

            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = controller.SAforward(attn)
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x_output_shape = (B, N, C)
            x = x.transpose(1, 2)
            x = x.reshape(x_output_shape)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        return forward
    def ca_forward(self):
        def forward(x, cond, mask=None):
            #print(F"mask:{mask}")
            #print(F"mask:{mask}")S
            """
            x: [B, N, C]，其中 B=批次数，N=查询序列长度，C=通道数
            cond: [1, total_K, C]，其中 total_K 为所有 batch 累计的 key 数，
                mask: list，长度为 B，每个元素表示该 batch 对应的 key 数（例如：[28, 29, 28, 29]）
            """
            B, N, C = x.shape
            #print(F"x.shape:{x.shape}")
            if mask is None:
                Bc, Nc, _ = cond.shape
                assert Bc == B, "cond 的 batch 与 x 必须匹配"
                mask = [Nc] * B

            # 计算 q, [B, N, C] → [B, N, num_heads, head_dim]
            q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)

            # 如果 cond 只有 1 个 batch，则复制到 B 个（保证每个查询对应各自的 key 区间）
            if cond.shape[0] == 1 and B > 1:
                cond = cond.repeat(B, 1, 1)
            # 计算 k,v, [B, total_K, C] → [B, total_K, 2, num_heads, head_dim]
            kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)  # 各自形状：[B, total_K, num_heads, head_dim]

            # 按标准 attention 缩放 q（这里假设没有在 q_linear 内部做缩放）
            scale = 1.0 / math.sqrt(self.head_dim)
            q = q * scale

            # 转置方便后续计算：使得形状变为 [B, num_heads, N, head_dim] 等
            q = q.transpose(1, 2)           # [B, num_heads, N, head_dim]
            k = k.transpose(1, 2)           # [B, num_heads, total_K, head_dim]
            v = v.transpose(1, 2)           # [B, num_heads, total_K, head_dim]
            # 由于原来使用 BlockDiagonalMask 对整个 batch 做分块，
            # 这里我们手动根据 mask 对每个 batch 的 key/v 做分块计算。
            outputs = []
            attn_list = []  # 保存每个 batch 的注意力分数，形状为 [num_heads, N, key_len]
            v_b_list = []
            key_start = 0
            for b in range(B):
                key_len = mask[b]  # 当前 batch 有效 key 数
                # 对于 batch b，从总的 key 中取出当前分块：注意这里 key 是按 cond 中的顺序排列的
                k_b = k[b, :, key_start:key_start+key_len, :]  # [num_heads, key_len, head_dim]
                v_b = v[b, :, key_start:key_start+key_len, :]  # [num_heads, key_len, head_dim]
                v_b_list.append(v_b)
                # q_b 对应当前 batch：形状 [num_heads, N, head_dim]
                q_b = q[b]  # [num_heads, N, head_dim]
                #print(F"q_b.sjape:{q_b.shape} k_b.shape:{k_b.shape} v_b.shape:{v_b.shape}")
                # 计算注意力分数：[num_heads, N, key_len]
                attn_scores = torch.matmul(q_b, k_b.transpose(-2, -1))
                
                # 计算 softmax 得到注意力权重（沿 key 维度归一化）
                attn_weights = torch.softmax(attn_scores, dim=-1)

                # 应用 dropout（如果设置了 dropout 概率）
                attn_weights = self.attn_drop(attn_weights)
                attn_list.append(attn_weights)
                #print(attn_weights.shape)
                # 利用注意力权重与 v_b 计算输出，[num_heads, N, head_dim]
                key_start += key_len  # 移动到下一个 batch 的 key 起始位置
            #sys.exit(0)
            # 通过 p2p 来修改注意力图
            attn_list = controller(attn_list)

            for b in range(B):
                attn_weights = attn_list[b]
                v_b = v_b_list[b]
                out = torch.matmul(attn_weights, v_b)
                outputs.append(out)
            # 将所有 batch 的输出堆叠：形状 [B, num_heads, N, head_dim]
            out = torch.stack(outputs, dim=0)
            # 转置回 [B, N, num_heads, head_dim]，再 reshape 为 [B, N, C]
            out = out.transpose(1, 2).reshape(B, N, C)
            # 后续的线性投影和 dropout
            out = self.proj(out)
            out = self.proj_drop(out)
            # 返回最终输出以及每个 batch 的注意力分数（方便直接访问 q @ k）
            return out
        return forward
    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0
        def SAforward(self, attn):
            return attn
    if controller == None:
        controller = DummyController()
    cross_att_count = 0
    def register_recr(net_, count):
        if net_.__class__.__name__ == 'MultiHeadCrossAttention':
            net_.forward = ca_forward(net_)
            return count + 1
        elif net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward_sa(net_)
            return count
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count)
        return count
    for net in model.named_children():
        if 'DiTMoudle' in net[0]:
            for layer in net[1]:
                cross_att_count += register_recr(layer, 0)

    controller.num_att_layers = cross_att_count
    print(F"cross_attention:{cross_att_count}")