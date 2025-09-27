import abc
import torch
import torch.nn as nn
from seq_aligner import get_replacement_mapper, get_mapper, get_word_inds
import torch.nn.functional as nnf
import sys
class LocalBlend:
    """
    a cat in the home
    a dog in the home
    """
    def __call__(self, x_t, attention_store, step):
        if self.attention_store != None:
            attention_store = self.attention_store
        Batch, C, T, H, W = x_t.shape
        mask0 = attention_store[0][ :, :, :, self.ind_list[0]].sum(dim = [0, 1, 3]).reshape(1, T, H // 2, W // 2) 
        mask0 = nnf.interpolate(mask0, size = (H, W)) # 填充
        mask0 = mask0 / mask0.max(2, keepdims=True)[0].max(3, keepdims=True)[0] # 找到最大的
        mask0 = mask0.gt(self.threshold)
        """
        猫
        狗
        latent = 想要改的地方(mask = 1) 狗
        """
        for i in range(1, Batch):
            x_t[i] = x_t[0] + mask0.to(x_t[0].device) * (x_t[i] - x_t[0])
        return x_t
    def __init__(self, prompts, words, tokenizer, threshold: float = 0.3, device = 'cuda:0'):
        ind_list = []
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if words_ is None:
                ind_list.append([])
                continue
            ind_list_ = []
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                #print(F"ind:{ind} word:{word}")
                ind_list_ = ind_list_ + ind.tolist()
            #print(F'ind_list_:{ind_list_}')
            ind_list.append(torch.asarray(ind_list_))
        self.threshold = threshold
        self.ind_list = ind_list
        self.attention_store = None

def split_attention(attn):
    length = attn[0].shape[2]
    encoder_length = attn[1]
    hidden_length = length - encoder_length # attn[:, :,  :attn[0].shape[2] - attn[1], :attn[0].shape[2] - attn[1]]
    attentionmap = attn[0]
    return (
        attentionmap[:, :, :hidden_length, :hidden_length], 
        attentionmap[:, :, :hidden_length, hidden_length:],  # attn[:, :, :attn[0].shape[2] - attn[1], attn[0].shape[2] - attn[1]:]
        attentionmap[:, :, hidden_length:, :hidden_length], 
        attentionmap[:, :, hidden_length:, hidden_length:]
    )

def Merge(attn):
    attn0, attn1, attn2, attn3 = attn[0], attn[1], attn[2], attn[3]
    top = torch.cat((attn0, attn1), dim=3)
    bottom = torch.cat((attn2, attn3), dim=3)
    return torch.cat((top, bottom), dim=2)

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward (self, attn_list: list):
        raise NotImplementedError

    def __call__(self, attn_list: list):
        h = len(attn_list)
        attn_list = self.forward(attn_list)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn_list
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    def forward (self, attn_list: list):
        return attn_list
def split_attention_1(attn):
    length = attn[0].shape[2]
    encoder_length = attn[1]
    hidden_length = length - encoder_length
    attentionmap = attn[0]
    return attentionmap[:, :, :hidden_length, hidden_length:].clone()
class AttentionStore(AttentionControl):
    def get_empty_store(self):
        return None
    def between_steps(self):
        self.step_store = self.get_empty_store()
    def forward(self, attn_list: list):
        attn_ca_list = [split_attention_1(attn) for attn in attn_list]
        if self.attention_store == None:
            self.attention_store = attn_ca_list
        else:
            for i in range(len(attn_ca_list)):
                self.attention_store[i] += attn_ca_list[i]
                #print(attn_ca_list[i].shape)
            #assert self.attention_store[0].shape == (1, 24, 23850, 7)
        return attn_list
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.attention_store = self.get_empty_store()
class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, attn_replace, mapper):
        raise NotImplementedError
    def forward(self, attn_list: list):
        super(AttentionControlEdit, self).forward(attn_list)
        #print(F"haha:{attn_list[i][0][:, :, :attn_list[i][0].shape[2] - attn_list[i][1], attn_list[i][0].shape[2] - attn_list[i][1]:].shape}")
        #print(F"attn_list[0].shape:{attn_list[0].shape}")
        #print(F"attn_list[0][:, :, :attn_list[0][0].shape[2] - attn_list[0][1], attn_list[0][0].shape[2] - attn_list[0][1]:]:{attn_list[0][0][:, :, :attn_list[0][0].shape[2] - attn_list[0][1], attn_list[0][0].shape[2] - attn_list[0][1]:].shape}")
        for i in range(1, len(attn_list)):
            """
            To do:
            修改自注意力图和相互注意力图
            """
            # attn_list[0]
            if self.cross_attention != 0:
                if self.cur_step < self.cross_attention * self.is_paper:
                    attn_list[i][0][:, :, :attn_list[i][0].shape[2] - attn_list[i][1], attn_list[i][0].shape[2] - attn_list[i][1]:] = self.replace_cross_attention(
                        attn_list[0][0][:, :, :attn_list[0][0].shape[2] - attn_list[0][1], attn_list[0][0].shape[2] - attn_list[0][1]:], 
                        attn_list[i][0][:, :, :attn_list[i][0].shape[2] - attn_list[i][1], attn_list[i][0].shape[2] - attn_list[i][1]:], 
                        self.mapper_list[i - 1]
                    )

            if self.cur_step < self.self_attention * self.is_paper:
                attn_list[i][0][:, :,  :attn_list[i][0].shape[2] - attn_list[i][1], :attn_list[i][0].shape[2] - attn_list[i][1]] = attn_list[0][0][:, :,  :attn_list[0][0].shape[2] - attn_list[0][1], :attn_list[0][0].shape[2] - attn_list[0][1]]
                
            #attn_list[i] = [Merge([attni_0, attni_1, attni_2, attni_3]), attn_list[i][1]]
            # 2 4
        
        return attn_list
    def __init__(self, prompts, num_steps, tokenlizer, dtype, local_blend = None):
        #assert tokenlizer is not None
        super(AttentionControlEdit, self).__init__()
        self.cross_attention = None
        self.is_paper = None
        self.self_attention = None
        self.batch_size = len(prompts)
        self.self_attention = None
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, attn_replace, mapper): # (16, 8790, 29) @ (29, 28) -> (16, 2790, 28)
        #print(F"attn_base.dtype:{attn_base.dtype}, mapper.dtype:{mapper.dtype}")
        return attn_base @ mapper
    """
    A pig
    A dog
    navigate prompt n
    """
    def __init__(self, prompts, num_steps, tokenlizer, dtype, cross_attention, self_attention, is_paper, lb, device):
        super(AttentionReplace, self).__init__(prompts, num_steps, tokenlizer, dtype, lb)
        if cross_attention != 0:
            self.mapper_list = get_replacement_mapper(prompts, tokenlizer)
            for i in range(len(self.mapper_list)):
                self.mapper_list[i] = self.mapper_list[i].to(dtype = dtype, device = device)
                #print(self.mapper_list[i])
        #sys.exit(0)
        self.is_paper = 1
        if is_paper:
            self.is_paper = 2
        self.self_attention = self_attention
        self.cross_attention = cross_attention
class AttentionDelete(AttentionControlEdit):  
    def replace_cross_attention(self, attn_base, attn_replace, mapper):
        #print(F"attn_base.shape:{attn_base.shape} attn_replace.shape:{attn_replace.shape} mapper.shape:{mapper[0].shape} alpha.shape:{mapper[1].shape}")
        attn_base_replace = attn_base[:, :, :, mapper[0]]
        #print(attn_base_replace.device, mapper[1].device, attn_replace.device)
        #print(F"attn_base_replace.shape:{attn_base_replace.shape}")
        assert attn_base_replace.shape == attn_replace.shape
        return attn_base_replace * mapper[1] + attn_replace * (1 - mapper[1])
    def __init__(self, prompts, num_steps, tokenlizer, dtype, cross_attention, self_attention, is_paper, lb, device):
        super(AttentionDelete, self).__init__(prompts, num_steps, tokenlizer, dtype, lb)
        
        if cross_attention != 0:
            self.mapper_list = []
            for i in range(1, len(prompts)):
                mapper, alpha = get_mapper(prompts[0], prompts[i], tokenlizer)
                alpha = alpha.to(device = device, dtype = torch.bfloat16)
                mapper = mapper.to(device = device)
                self.mapper_list.append((mapper, alpha))
                #print(mapper, alpha)
        self.is_paper = 1
        if is_paper:
            self.is_paper = 2
        self.alpha_words_list_1 = [torch.ones(len(prompts) - 1, device = 'cuda') for _ in range(cross_attention * self.is_paper)] 
        self.alpha_words_list_2 = [torch.zeros(len(prompts) - 1, device = 'cuda') for _ in range(num_steps * self.is_paper - cross_attention * self.is_paper)]
        self.alpha_words_list = self.alpha_words_list_1 + self.alpha_words_list_2
        self.self_attention = self_attention
        self.cross_attention = cross_attention
         
        