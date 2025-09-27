import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
from audioldm.utils import default_audioldm_config
from audioldm.audio import wave_to_fbank, TacotronSTFT
import importlib.util
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import sys
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import functools
import torch.distributed as dist
import numpy as np
import json
from diffusers import AutoencoderKL
from typing import Optional
import math




def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: Union[int, None] = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> Union[_LRScheduler, None]: 
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
        # lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)

    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs
        


def loss_function(recon, x, mu, logvar) -> torch.Tensor:
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 这个具体是怎么搞
    #print("recon_loss",recon_loss.item(),"kl_loss",kl_loss.item())
    loss = recon_loss + kl_loss

    return loss


# 文件路径
def get_config(file_path):
    # 模块名（可以任意指定）
    module_name = "stage1_config"
    # 动态导入模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # 提取模块中的变量并形成字典
    config_dict = {}
    for key in dir(module):
        if not key.startswith("__"):  # 过滤掉内部变量
            config_dict[key] = getattr(module, key)
    return config_dict


class AudioCropPad:
    """将音频裁剪或填充到固定长度"""
    def __init__(self, target_length: int, mode: str = "random"):
        """
        Args:
            target_length (int): 目标音频长度（采样点数）
            mode (str): 处理模式，可选 ["random", "center"]
        """
        self.target_length = target_length
        self.mode = mode

    def __call__(self, waveform):
        """
        输入形状: (channels, time)
        输出形状: (channels, target_length)
        """
        channels, length = waveform.shape
        
        # 处理长度不足的情况（填充）
        if length < self.target_length:
            return self._pad_waveform(waveform)
        
        # 处理长度过长的情况（裁剪）
        return self._crop_waveform(waveform)

    def _pad_waveform(self, waveform):
        """对称填充音频"""
        padding = self.target_length - waveform.size(1)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return (F.pad(waveform, (pad_left, pad_right)),self.target_length/16000)

    def _crop_waveform(self, waveform):
        """根据模式裁剪音频"""
        length = waveform.size(1)
        
        if self.mode == "random":
            start = torch.randint(0, length - self.target_length, (1,)).item()
        elif self.mode == "center":
            start = (length - self.target_length) // 2
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return (waveform[:, start:start+self.target_length],self.target_length/16000)
    
def get_sigmas(timesteps, n_dim=4, device="cuda", dtype=torch.float16):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype) #scheduler.sigmas是一直是1000吗？
    schedule_timesteps = scheduler.timesteps.to(device, dtype=dtype)
    timesteps = timesteps.to(device,dtype=dtype)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class audio_VAE(nn.Module):
    def __init__(self, encoder, quant_conv, post_quant_conv, decoder):
        super(audio_VAE, self).__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0) #这个地方不会被转化，但混合精度对optimizer有要求。因此得提前转化


    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()
        recon = self.decoder(self.post_quant_conv(z))
        return recon,posterior


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class Get_loss():
    def __init__(self, disc_start = 5000, kl_weight=1.0e-06, pixelloss_weight=1.0,
                  disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=0, disc_loss="hinge"):

        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def compute_loss(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, discriminator, last_layer=None, logvar = None,disc_optimizer = None):
        rec_loss = torch.abs(inputs.contiguous()[:,:,:int(inputs.shape[2]//4*4),:] - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(logvar) + logvar
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        #print("weighted_nll_loss",weighted_nll_loss.item())
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #print("kl_loss",kl_loss.item())
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            return loss

        if optimizer_idx == 1:
            assert disc_optimizer is not None
            # second pass for discriminator update
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            logits_real = discriminator((inputs[:,:,:int(inputs.shape[2]//4*4),:]).contiguous().detach())
            loss_real = disc_factor*0.5 * torch.mean(F.relu(1. - logits_real))
            disc_optimizer.backward(loss_real)
            logits_fake = discriminator(reconstructions.contiguous().detach())
            loss_fake = disc_factor*0.5 * torch.mean(F.relu(1. + logits_fake))
            disc_optimizer.backward(loss_fake)
            d_loss = loss_real+loss_fake
            return d_loss

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

class Audioprepro:
    def __init__(self,sc=1,sr=16000):
        self.sc = sc
        self.sr = sr
    def __call__(self,input):
        (wave,source_sr) = input
        #wave: [sc,len]
        wave = wave[:source_sr,:]
        wave = torchaudio.functional.resample(wave, source_sr, self.sr)
        return wave
    

def create_tensorboard_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype = self.parameters.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class train_dis(nn.Module):
    def __init__(self):
        super(train_dis,self).__init__()
        self.layer = nn.Conv2d(1,1,kernel_size=3,padding=1)
        self.sigma = 1
    def forward(self,x):
        return self.layer(x)
    
def load_avae(json_path = "/lustre/zhanghy_group/fyq/avae/config2.json",model_dir = "/lustre/zhanghy_group/fyq/disc_avae/avae/epoch36-global_step120000/model"):
    with open(json_path, 'r') as f:
        vae_config = json.load(f)
    avae = AutoencoderKL(**vae_config)
    quant_conv = avae.quant_conv
    post_quant_conv = avae.post_quant_conv
    encoder = avae.encoder
    decoder = avae.decoder
    my_avae = MY_VAE(encoder, quant_conv, post_quant_conv, decoder)
    shard_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")]
    full_state_dict = {}
    for shard_file in shard_files:
        shard_state_dict = torch.load(shard_file, map_location=lambda storage, loc: storage)
        full_state_dict.update(shard_state_dict)
    my_avae.load_state_dict(full_state_dict)
    return my_avae # 采样的过程中是否要随机采样？


def collate_fn(batch):
    if not batch:  # 处理空批次
        return []
    # 创建拼接后的字典，每个键对应堆叠后的张量
    collated = {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }
    return collated


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

class AudioDataset(Dataset):
    def __init__(self, csv_dir, duration, transform=None):
        self.transform = transform
        a = pd.read_csv(csv_dir)
        a = a[a["duration"] > duration]
        self.dataframe = a
        print("总体训练音频数量:", len(a))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        audio_path = data['path']
        waveform, sr = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform((waveform,sr))
        ret = {
            "audio": waveform,
        }
        text_mask_dict = torch.load(data["text_path"], weights_only=True)
        ret.update(text_mask_dict)
        return ret # 可根据需要返回其他数据（如标签）


def encode_duration(
    audio_start_in_s,
    audio_end_in_s,
    device,
    do_classifier_free_guidance,
    batch_size,
    projection_model
):
    # 我想提醒大家一点，这里由于我的问题，输入数据并没有做到一致：
    #  audio_start_in_s:[0, 0]
    #  audio_end_in_s:tensor([10.0900, 10.0900], dtype=torch.float64)
    #  我们的目标是将全部的数据转换为tensor类型，并全部位于cuda上，所以我们需要将audio_start_in_s转换为tensor类型，并全部位于cuda上

    audio_start_in_s = [float(x) for x in audio_start_in_s]
    audio_start_in_s = torch.tensor(audio_start_in_s).to(device)
    audio_end_in_s = audio_end_in_s.to(device)

    projection_output = projection_model(
        start_seconds=audio_start_in_s,
        end_seconds=audio_end_in_s,
    )
    seconds_start_hidden_states = projection_output.seconds_start_hidden_states
    seconds_end_hidden_states = projection_output.seconds_end_hidden_states

    # For classifier free guidance, we need to do two forward passes.
    # Here we repeat the audio hidden states to avoid doing two forward passes
    if do_classifier_free_guidance:
        seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
        seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states], dim=0)

    return seconds_start_hidden_states, seconds_end_hidden_states


class CompositeModel(nn.Module):
    def __init__(self, amodel, bridge):
        super(CompositeModel, self).__init__()
        self.ammodel = amodel
        self.bridge = bridge

    def forward(self, x):
        pass



def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: Union[int, None] = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> Union[_LRScheduler, None]: 
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
        # lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)

    return lr_scheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs
        


def loss_function(recon, x, mu, logvar) -> torch.Tensor:
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 这个具体是怎么搞
    #print("recon_loss",recon_loss.item(),"kl_loss",kl_loss.item())
    loss = recon_loss + kl_loss

    return loss




auido_config = default_audioldm_config()
fn_STFT = TacotronSTFT(
    auido_config["preprocessing"]["stft"]["filter_length"],
    auido_config["preprocessing"]["stft"]["hop_length"],
    auido_config["preprocessing"]["stft"]["win_length"],
    auido_config["preprocessing"]["mel"]["n_mel_channels"],
    auido_config["preprocessing"]["audio"]["sampling_rate"],
    auido_config["preprocessing"]["mel"]["mel_fmin"],
    auido_config["preprocessing"]["mel"]["mel_fmax"],
)




def get_mel(wave,duration):
    mel, _, _ = wave_to_fbank(
            wave, target_length=int(duration * 100+1), fn_STFT=fn_STFT
        )
    return mel

class  GetMelTransform:
    def __call__(self, inputs):
        wave, duration = inputs
        return get_mel(wave, duration)

# 文件路径
def get_config(file_path):
    # 模块名（可以任意指定）
    module_name = "stage1_config"
    # 动态导入模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # 提取模块中的变量并形成字典
    config_dict = {}
    for key in dir(module):
        if not key.startswith("__"):  # 过滤掉内部变量
            config_dict[key] = getattr(module, key)
    return config_dict


class AudioCropPad:
    """将音频裁剪或填充到固定长度"""
    def __init__(self, target_length: int, mode: str = "random"):
        """
        Args:
            target_length (int): 目标音频长度（采样点数）
            mode (str): 处理模式，可选 ["random", "center"]
        """
        self.target_length = target_length
        self.mode = mode

    def __call__(self, waveform):
        """
        输入形状: (channels, time)
        输出形状: (channels, target_length)
        """
        channels, length = waveform.shape
        
        # 处理长度不足的情况（填充）
        if length < self.target_length:
            return self._pad_waveform(waveform)
        
        # 处理长度过长的情况（裁剪）
        return self._crop_waveform(waveform)

    def _pad_waveform(self, waveform):
        """对称填充音频"""
        padding = self.target_length - waveform.size(1)
        pad_left = padding // 2
        pad_right = padding - pad_left
        return (F.pad(waveform, (pad_left, pad_right)),self.target_length/16000)

    def _crop_waveform(self, waveform):
        """根据模式裁剪音频"""
        length = waveform.size(1)
        
        if self.mode == "random":
            start = torch.randint(0, length - self.target_length, (1,)).item()
        elif self.mode == "center":
            start = (length - self.target_length) // 2
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return (waveform[:, start:start+self.target_length],self.target_length/16000)
    

class MY_VAE(nn.Module):
    def __init__(self, encoder, quant_conv, post_quant_conv, decoder):
        super(MY_VAE, self).__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0) #这个地方不会被转化，但混合精度对optimizer有要求。因此得提前转化


    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()
        recon = self.decoder(self.post_quant_conv(z))
        return recon,posterior


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class Get_loss():
    def __init__(self, disc_start = 5000, kl_weight=1.0e-06, pixelloss_weight=1.0,
                  disc_factor=1.0, disc_weight=0.5,
                 perceptual_weight=0, disc_loss="hinge"):

        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def compute_loss(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, discriminator, last_layer=None, logvar = None,disc_optimizer = None):
        rec_loss = torch.abs(inputs.contiguous()[:,:,:int(inputs.shape[2]//4*4),:] - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(logvar) + logvar
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        #print("weighted_nll_loss",weighted_nll_loss.item())
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #print("kl_loss",kl_loss.item())
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            return loss

        if optimizer_idx == 1:
            assert disc_optimizer is not None
            # second pass for discriminator update
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            logits_real = discriminator((inputs[:,:,:int(inputs.shape[2]//4*4),:]).contiguous().detach())
            loss_real = disc_factor*0.5 * torch.mean(F.relu(1. - logits_real))
            disc_optimizer.backward(loss_real)
            logits_fake = discriminator(reconstructions.contiguous().detach())
            loss_fake = disc_factor*0.5 * torch.mean(F.relu(1. + logits_fake))
            disc_optimizer.backward(loss_fake)
            d_loss = loss_real+loss_fake
            return d_loss

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

class Audioprepro:
    def __init__(self,sc=1,sr=16000):
        self.sc = sc
        self.sr = sr
    def __call__(self,input):
        (wave,source_sr) = input
        #wave: [sc,len]
        wave = wave[:source_sr,:]
        wave = torchaudio.functional.resample(wave, source_sr, self.sr)
        return wave
    

def create_tensorboard_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype = self.parameters.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class train_dis(nn.Module):
    def __init__(self):
        super(train_dis,self).__init__()
        self.layer = nn.Conv2d(1,1,kernel_size=3,padding=1)
        self.sigma = 1
    def forward(self,x):
        return self.layer(x)
    
def load_avae(json_path = "/lustre/zhanghy_group/fyq/avae/config2.json",model_dir = "/lustre/zhanghy_group/fyq/disc_avae/avae/epoch36-global_step120000/model"):
    with open(json_path, 'r') as f:
        vae_config = json.load(f)
    avae = AutoencoderKL(**vae_config)
    quant_conv = avae.quant_conv
    post_quant_conv = avae.post_quant_conv
    encoder = avae.encoder
    decoder = avae.decoder
    my_avae = MY_VAE(encoder, quant_conv, post_quant_conv, decoder)
    shard_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")]
    full_state_dict = {}
    for shard_file in shard_files:
        shard_state_dict = torch.load(shard_file, map_location=lambda storage, loc: storage)
        full_state_dict.update(shard_state_dict)
    my_avae.load_state_dict(full_state_dict)
    return my_avae # 采样的过程中是否要随机采样？


def collate_fn(batch):
    if not batch:  # 处理空批次
        return []
    # 创建拼接后的字典，每个键对应堆叠后的张量
    collated = {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }
    return collated


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule