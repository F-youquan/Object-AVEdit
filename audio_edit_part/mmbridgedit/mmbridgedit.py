import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel
from .blocks_bridge import (
    CaptionEmbedder,
    PatchEmbed2D,
    PositionEmbedding2D,
    TimestepEmbedder,
    MultiHeadCrossAttention,
    Attention,
    T2IFinalLayer,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from transformers import PretrainedConfig, PreTrainedModel

class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn


        attn_cls = Attention
        mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.attn.apply(self._init_weights)
        self.cross_attn.apply(self._init_weights)
        self.mlp.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask1=None,  # y1 mask
    ):  
        # 暂时我们仅允许语言部分(y1)有mask
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp  = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
      
        x_m = self.attn(x_m)
        
        # modulate (attention)
        x_m_s = gate_msa * x_m
        x = x + self.drop_path(x_m_s)

        # cross attention
        x = x + self.cross_attn(x, y, mask1)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
    
        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m

        # residual
        x = x + self.drop_path(x_m_s)
        return x



class MMBridgeDiTConfig(PretrainedConfig):
    model_type = "MMBridgeDiT"
    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        vin_channels = 12,
        ain_channels=64,
        vpatch_size = (1,2,2),
        apatch_size= 2,
        hidden_size=1152,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=False,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=True,
        enable_layernorm_kernel=False,
        only_train_temporal=False,
        video_depth = 4,
        audio_depth = 4,
        audio_frame_depth = 4,
        video_frame_depth = 4,
        depth = 1,
        skip_y_embedder=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.input_sq_size = input_sq_size  #这个值保留，且不要改动
        self.vin_channels = vin_channels
        self.ain_channels = ain_channels 
        self.vpatch_size = vpatch_size
        self.apatch_size = apatch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.only_train_temporal = only_train_temporal
        self.video_depth = video_depth
        self.audio_depth = audio_depth
        self.audio_frame_depth = audio_frame_depth
        self.video_frame_depth = video_frame_depth
        self.depth = depth
        self.skip_y_embedder = skip_y_embedder
 


class MMBridgeDiT(PreTrainedModel):
    config_class = MMBridgeDiTConfig
    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.ain_channels = config.ain_channels
        self.aout_channels = config.ain_channels * 2 if config.pred_sigma else config.ain_channels

        # model size related
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # input size related
        self.apatch_size = config.apatch_size
        self.input_sq_size = config.input_sq_size
        self.apos_embed = PositionEmbedding2D(config.hidden_size)
        # embedding
        self.ax_embedder = PatchEmbed2D(config.apatch_size, config.ain_channels, config.hidden_size, norm_layer=nn.LayerNorm)

        #self.a_t_embedder = TimestepEmbedder(config.hidden_size)
        #self.a_fps_embedder = SizeEmbedder(self.hidden_size)
        self.a_t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

                                    
        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.DiTMoudle = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                )
                for i in range(config.depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.apatch_size), self.aout_channels)
        self.t_embedder = TimestepEmbedder(config.hidden_size)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            mask = mask.to(int)
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens
        
    # 在MMBridgeDiT类中添加初始化方法
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)



    def forward(self, ax, y, mask, timestep):
        # ax输入：B，C，L
        ax, lift_pad, right_pad = self.ax_embedder(ax)   # B,C,W,H
        _,_,aW,aH = ax.shape
        aS = aW*aH
        abase_size = round(aS**0.5)
        apos_emb = self.apos_embed(ax,aW, aH, base_size=abase_size)
        #ax = rearrange(ax,"B C (T faW) aH -> (B T) (faW aH) C", T = T, faW = aW//T, aH = aH)
        ax = rearrange(ax,"B C aW aH -> B (aW aH) C")
        ax = ax + apos_emb # 每一帧对应的音频都嵌入相同的位置编码
        # === get y embed ===
        y, y_lens = self.encode_text(y, mask)
        y_lens = [int(i) for i in y_lens]

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=ax.dtype)  # [B, C]
        t_mlp = self.a_t_block(t)
        for block in self.DiTMoudle:
            ax = block(ax, y, t_mlp, y_lens)

        # === final layer ===
        ax = self.final_layer(ax, t)
        ax = self.unpatchify(ax, aW, aH, lift_pad, right_pad)
        return ax

    def unpatchify(self, x, aW, aH, lift_pad, right_pad):
        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        H_p, W_p = self.apatch_size
        x = rearrange(
            x,
            "B (aW aH) (H_p W_p C_out) -> B C_out (aH H_p) (aW W_p)",
            aW = aW,
            aH = aH,
            H_p=H_p,
            W_p=W_p,
            C_out=self.aout_channels,
        )
        # unpad
        x = x[:, :, :, lift_pad:aW*W_p-right_pad]
        return x
    

    



