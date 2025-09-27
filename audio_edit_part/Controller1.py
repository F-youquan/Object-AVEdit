import abc
import torch
import torch.nn as nn
from seq_aligner import get_replacement_mapper, get_mapper

from seq_aligner import get_replacement_mapper, get_mapper, get_word_inds
import torch.nn.functional as nnf
import sys
class LocalBlend:
    def __call__(self, x_t, attention_store, step):
        if self.attention_store != None:
            attention_store = self.attention_store
        Batch, C, H, W = x_t.shape
        mask0 = attention_store[0][:,:,self.ind_list[0]].sum(dim = [0, 2]).reshape(1, 1, 38, 16)
        #print(F"mask0:{mask0}")
        mask0 = nnf.interpolate(mask0, size = (75, 16)).permute(0, 1, 3, 2)
        mask0 = mask0 / mask0.max(2, keepdims = True)[0].max(3, keepdims=True)[0]
        mask0 = mask0.gt(self.threshold)
        for i in range(1, Batch):
            x_t[i] = x_t[0] + mask0 * (x_t[i] - x_t[0])
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
            print(F'ind_list_:{ind_list_}')
            ind_list.append(torch.asarray(ind_list_))
        self.threshold = threshold
        self.ind_list = ind_list
        self.attention_store = None
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
        attn_list[h // 2: ] = self.forward(attn_list[h // 2: ])
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
    def SAforward(self, attn):
        return attn

class EmptyControl(AttentionControl):
    def forward (self, attn_list: list):
        return attn_list
class AttentionStore(AttentionControl):
    def forward(self, attn_list: list):
        if self.attention_store == None:
            self.attention_store = attn_list
        else:
            for i in range(len(attn_list)):
                self.attention_store[i] += attn_list[i]
        return attn_list
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.attention_store = None
class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend != None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, attn_replace, mapper):
        raise NotImplementedError
    def forward(self, attn_list: list):
        super(AttentionControlEdit, self).forward(attn_list)
        alpha_words_list = self.alpha_words_list[self.cur_step]
        attn0 = attn_list[0]
        for i in range(1, len(attn_list)):
            alpha_words = alpha_words_list[i - 1]
            attni = attn_list[i]
            if self.cross_attention != 0:
                attni_new = self.replace_cross_attention(attn0, attni, self.mapper_list[i - 1])
            else:
                attni_new = attni
            attni = attni_new * alpha_words + attni * (1 - alpha_words)
            attn_list[i] = attni
        return attn_list
    def sa_forward(self, attn):
        B = attn.shape[0]
        if self.cur_step < self.self_attention:
            attn = attn[0:1, :, :, :].repeat(B, 1, 1, 1)
        return attn
    def SAforward(self, attn):
        B = attn.shape[0]
        attn[B // 2 : , :, :, :] = self.sa_forward(attn[B // 2 :, :, :, :])
        return attn
    def __init__(self, local_blend):
        super(AttentionControlEdit, self).__init__()
        self.local_blend = local_blend
class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, attn_replace, mapper): # (16, 8790, 29) @ (29, 28) -> (16, 2790, 28)
        return attn_base @ mapper
    def __init__(self, prompts, num_steps, tokenlizer, dtype, cross_attention, self_attention, is_paper, local_blend):
        super(AttentionReplace, self).__init__(local_blend)
        if cross_attention != 0:
            self.mapper_list = get_replacement_mapper(prompts, tokenlizer)
            for i in range(len(self.mapper_list)):
                self.mapper_list[i] = self.mapper_list[i].to(dtype = dtype)
        self.is_paper = 2 if is_paper else 1
        self.alpha_words_list_1 = [torch.ones(len(prompts) - 1, device = 'cuda') for _ in range(cross_attention * self.is_paper)] 
        self.alpha_words_list_2 = [torch.zeros(len(prompts) - 1, device = 'cuda') for _ in range(num_steps * self.is_paper - cross_attention * self.is_paper)]
        self.alpha_words_list = self.alpha_words_list_1 + self.alpha_words_list_2
        self.cross_attention = cross_attention
        self.self_attention = self_attention
class AttentionDelete(AttentionControlEdit):  
    def replace_cross_attention(self, attn_base, attn_replace, mapper):
        #print(F"attn_base.shape:{attn_base.shape}  attn_replace.shape:{attn_replace.shape} mapper:{mapper[0]}")
        attn_base_replace = attn_base[ :, :, mapper[0]]
        assert attn_base_replace.shape == attn_replace.shape
        return attn_base_replace * mapper[1] + attn_replace * (1 - mapper[1])
    def __init__(self, prompts, num_steps, tokenlizer, dtype, cross_attention, self_attention, is_paper, local_blend):
        super(AttentionDelete, self).__init__(local_blend)
        self.mapper_list = []
        for i in range(1, len(prompts)):
            mapper, alpha = get_mapper(prompts[0], prompts[i], tokenlizer)
            print(mapper)
            alpha = alpha.to(device = 'cuda', dtype = dtype)
            self.mapper_list.append((mapper, alpha))
        self.is_paper = 1
        if is_paper:
            self.is_paper = 2
        self.alpha_words_list_1 = [torch.ones(len(prompts) - 1, device = 'cuda') for _ in range(cross_attention * self.is_paper)] 
        self.alpha_words_list_2 = [torch.zeros(len(prompts) - 1, device = 'cuda') for _ in range(num_steps * self.is_paper - cross_attention * self.is_paper)]
        self.alpha_words_list = self.alpha_words_list_1 + self.alpha_words_list_2
        self.self_attention = self_attention
        self.cross_attention = cross_attention
         
        