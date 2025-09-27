import os
import torch
import numpy as np
from diffusers import AutoencoderKLMochi, MochiTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
import safetensors
from safetensors import safe_open
from tqdm import tqdm
from FM import FlowMatchEulerDiscreteScheduler
import config
from transformers import T5TokenizerFast, T5EncoderModel
from edit_util import *
from edit_Controller import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_video",type = str)
parser.add_argument("--edit_type",type = str)
parser.add_argument("--GPU_num",type=int)
parser.add_argument("--source_prompt",type=str)
parser.add_argument("--target_prompt",type=str)
parser.add_argument("--word",type=str,default=None)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--CA",type=int,default=37)
parser.add_argument("--SA",type=int,default=37)
parser.add_argument("--threshold",type=float,default=0.1)

args = parser.parse_args()
torch.manual_seed(args.seed)

assert args.edit_type in ["Addition","Replacement","Removal"],"任务必须在addition,replacement,removal之内"
dtype = torch.bfloat16
print(args.GPU_num)
device = f"cuda:{args.GPU_num}" if torch.cuda.is_available() else 'cpu'
tokenizer = T5TokenizerFast.from_pretrained(config.path_tokenizer)

file_name = args.input_video
prompts = [args.source_prompt, args.target_prompt]
word = args.word

if args.edit_type == "Addition":
    assert args.word == None,"在addition模式下word必须为None"

if args.edit_type == "Addition":
    CA, SA, threshold = args.CA, args.SA, args.threshold
    lb = None
    controller = AttentionDelete(prompts, config.num_inference_steps, tokenizer, dtype, CA, SA, True, lb, device)
elif args.edit_type == "Replacement":
    CA, SA, threshold = args.CA, args.SA, args.threshold
    lb = LocalBlend(prompts, (word, None), tokenizer, threshold)
    controller = AttentionReplace(prompts, config.num_inference_steps, tokenizer, dtype, CA, SA, True, lb, device)
elif args.edit_type == "Removal":
    CA, SA, threshold = args.CA, args.SA, args.threshold
    # threshold越大，想改动的地方越小（越难以改动）
    # 逻辑：mask0 = mask0.gt(self.threshold)
    # x_t[i] = x_t[0] + mask0.to(F"cuda:0") * (x_t[i] - x_t[0])
    lb = LocalBlend(prompts, (word, None), tokenizer, threshold)
    controller = AttentionDelete(prompts, config.num_inference_steps, tokenizer, dtype, CA, SA, True, lb, device)
else:
    raise KeyError

VAE = AutoencoderKLMochi(**config.config_vae)
state_dict = safetensors.torch.load_file(config.path_vae, device='cpu')
VAE.load_state_dict(state_dict)
VAE.to(device)
VAE.eval()
video_reader = Video_reader(file_name)
video = video_reader.read_video(np.arange(video_reader.video_len())).permute(1, 0, 2, 3)
video = video.unsqueeze(0).to(device) # [1, Channel, Time, Height, Width]
video = (video - video.min()) / (video.max() - video.min())
video = video * 2.0 - 1.0
with torch.no_grad():
    video_z = VAE.encode(video).latent_dist.sample()
with torch.no_grad():
      has_latents_mean = hasattr(VAE.config, "latents_mean") and VAE.config.latents_mean is not None
      has_latents_std = hasattr(VAE.config, "latents_std") and VAE.config.latents_std is not None
      if has_latents_mean and has_latents_std:
            latents_mean = (
                  torch.tensor(VAE.config.latents_mean).view(1, 12, 1, 1, 1).to(video_z.device, video_z.dtype)
            )
            latents_std = (
                  torch.tensor(VAE.config.latents_std).view(1, 12, 1, 1, 1).to(video_z.device, video_z.dtype)
            )
            video_z = (video_z - latents_mean) / latents_std * VAE.config.scaling_factor 
      else:
            video_z = video_z * VAE.config.scaling_factor
del VAE
torch.cuda.empty_cache()
def Get_embedding(prompt, negative_prompt):
    text_encoder = T5EncoderModel.from_pretrained(config.path_text_encoder)
    text_encoder.eval().to(device)
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=config.do_classifier_free_guidance,
        num_videos_per_prompt=config.num_videos_per_prompt,
        max_sequence_length=config.max_sequence_length,
        device=device,
        dtype=dtype,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
    )
    del text_encoder
    torch.cuda.empty_cache()
    if config.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    return prompt_embeds, prompt_attention_mask
prompt_embeds_, prompt_attention_mask_ = Get_embedding(prompts, None)
sigmas = linear_quadratic_schedule(config.num_inference_steps, config.threshold_noise)
sigmas = np.array(sigmas)
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=1.0,
    base_image_seq_len=256,
    base_shift=0.5,
    invert_sigmas=True,
    max_image_seq_len=4096,
    max_shift=1.15,
    use_dynamic_shifting=False
)
timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            config.num_inference_steps,
            device,
            config.timesteps,
            sigmas,
        )
num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
_num_timesteps = len(timesteps)
timesteps = torch.cat((timesteps, torch.asarray([1000.0]).to(device)))
timesteps = timesteps.flip(dims=[0])
transformer = MochiTransformer3DModel().to(device, dtype)
tensor_dict = {} 
shard_files = [os.path.join(config.path_transformer, f) for f in os.listdir(config.path_transformer) if f.startswith("diffusion") and f.endswith(".safetensors")]
for file_path in shard_files:
    with safe_open(file_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)
transformer.load_state_dict(tensor_dict, strict=True)
transformer.eval()
scheduler.sigmas = scheduler.sigmas.to(device)
latents = video_z.to(dtype)
sigmas = scheduler.sigmas.flip(dims = [0])
L = 3
prompt_embeds = torch.cat((prompt_embeds_[0:1], prompt_embeds_[len(prompts) : len(prompts) + 1]))
prompt_attention_mask = torch.cat((prompt_attention_mask_[0:1], prompt_attention_mask_[len(prompts):len(prompts) + 1]))
with tqdm(total=len(timesteps) - 1) as progress_bar:
    inference_count = len(timesteps) - 1
    for i, (t, pret) in enumerate(zip(timesteps[0:inference_count], timesteps[1:inference_count+1])):
        cur_sigmas, nex_sigmas = sigmas[i], sigmas[i + 1]
        mid_sigmas = (cur_sigmas + nex_sigmas) / 2.0
        midt = (t + pret) / 2.0
        Latent_list = []
        latents_ = latents.clone()
        for _ in range(L):
            latent_model_input = torch.cat([latents_] * 2) if config.do_classifier_free_guidance else latents_
            timestep = (mid_sigmas * 1000).expand(latent_model_input.shape[0]).to(latents_.dtype)
            with torch.no_grad():
                noise_pred1 = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            noise_pred1 = noise_pred1.to(torch.float32)
            if config.do_classifier_free_guidance:
                noise_pred_uncond1, noise_pred_text1 = noise_pred1.chunk(2)
                noise_pred1 = noise_pred_uncond1 + config.inversion_guidance_scale * (noise_pred_text1 - noise_pred_uncond1)
            latents_dtype = latents_.dtype
            latents_mid, tmid = scheduler.rstep_paper_mid(noise_pred1, pret, latents.to(torch.float32), return_dict=False)
            latents_mid = latents_mid.to(latents_dtype)
            latent_model_input = torch.cat([latents_mid] * 2) if config.do_classifier_free_guidance else latents_mid
            timestep = (nex_sigmas * 1000).expand(latent_model_input.shape[0]).to(latents_mid.dtype) 
            with torch.no_grad():
                noise_pred2 = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]            
            noise_pred2 = noise_pred2.to(torch.float32)
            if config.do_classifier_free_guidance:
                noise_pred_uncond2, noise_pred_text2 = noise_pred2.chunk(2)
                noise_pred2 = noise_pred_uncond2 + config.inversion_guidance_scale * (noise_pred_text2 - noise_pred_uncond2)
            latents_ = scheduler.rstep_paper_end(noise_pred1, noise_pred2, pret, latents.to(torch.float32), return_dict=False)[0]
            latents_ = latents_.to(latents_dtype)
            Latent_list.append(latents_)
        latents = torch.mean(torch.stack(Latent_list), dim=0)
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
            progress_bar.update()

prompt_embeds, prompt_attention_mask = prompt_embeds_, prompt_attention_mask_
latents = latents.repeat(len(prompts), 1, 1, 1, 1)
batch_size = len(prompts)
sigmas = linear_quadratic_schedule(config.num_inference_steps, config.threshold_noise)
sigmas = np.array(sigmas)
timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            config.num_inference_steps,
            device,
            config.timesteps,
            sigmas,
        )
num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
_num_timesteps = len(timesteps)
register_attention_control(transformer, controller)
scheduler.sigmas = scheduler.sigmas.to(device)
prompt_embeds, prompt_attention_mask = prompt_embeds_, prompt_attention_mask_
with tqdm(total=len(timesteps)) as progress_bar:
    for i, t in enumerate(timesteps):
        curt, next = scheduler.sigmas[i], scheduler.sigmas[i + 1]
        midt = (curt + next) / 2
        latent_model_input = torch.cat([latents] * 2) if config.do_classifier_free_guidance else latents
        timestep = (curt * 1000).expand(latent_model_input.shape[0]).to(latents.dtype)
        from concurrent.futures import ThreadPoolExecutor, wait
        with torch.no_grad():
            latent_model_input_0, latent_model_input_1 = latent_model_input.chunk(2)
            prompt_embeds_0, prompt_embeds_1 = prompt_embeds.chunk(2)
            timestep_0, timestep_1 = timestep.chunk(2)
            prompt_attention_mask_0, prompt_attention_mask_1 = prompt_attention_mask.chunk(2)
            transformer.edit = False
            noise_pred_0_0 = transformer(latent_model_input_0, prompt_embeds_0, timestep_0, prompt_attention_mask_0)[0]
            transformer.edit = True
            noise_pred_0_1 = transformer(latent_model_input_1, prompt_embeds_1, timestep_1, prompt_attention_mask_1)[0]
            noise_pred_0 = torch.cat((noise_pred_0_0, noise_pred_0_1))
        noise_pred_0 = noise_pred_0.to(torch.float32)
        if config.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_0.chunk(2)
            noise_pred_0 = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents_dtype = latents.dtype
        mid_latents = latents.to(torch.float32) + (midt - curt) * noise_pred_0
        mid_latents = controller.step_callback(mid_latents)
        mid_latents = mid_latents.to(latents_dtype)
        latent_model_input = torch.cat([mid_latents] * 2) if config.do_classifier_free_guidance else mid_latents
        timestep = (midt * 1000).expand(latent_model_input.shape[0]).to(latents.dtype)
        with torch.no_grad():
            latent_model_input_0, latent_model_input_1 = latent_model_input.chunk(2)
            prompt_embeds_0, prompt_embeds_1 = prompt_embeds.chunk(2)
            timestep_0, timestep_1 = timestep.chunk(2)
            prompt_attention_mask_0, prompt_attention_mask_1 = prompt_attention_mask.chunk(2)
            transformer.edit = False
            noise_pred_1_0 = transformer(latent_model_input_0, prompt_embeds_0, timestep_0, prompt_attention_mask_0)[0]
            transformer.edit = True
            noise_pred_1_1 = transformer(latent_model_input_1, prompt_embeds_1, timestep_1, prompt_attention_mask_1)[0]
            noise_pred_1 = torch.cat((noise_pred_1_0, noise_pred_1_1))
        noise_pred_1 = noise_pred_1.to(torch.float32)
        if config.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_1.chunk(2)
            noise_pred_1 = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        two_del_noise = (noise_pred_1 - noise_pred_0) / (midt - curt)
        latents = latents + noise_pred_0 * (next - curt) + 0.5 * (next - curt) * (next - curt) * two_del_noise
        latents = controller.step_callback(latents)
        latents = latents.to(latents_dtype)
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
            progress_bar.update()
torch.cuda.empty_cache()
VAE = AutoencoderKLMochi(**config.config_vae)
VAE.load_state_dict(safetensors.torch.load_file(config.path_vae))
VAE.to(device)
VAE.eval()
VAE.enable_tiling()
has_latents_mean = hasattr(VAE.config, "latents_mean") and VAE.config.latents_mean is not None
has_latents_std = hasattr(VAE.config, "latents_std") and VAE.config.latents_std is not None
if has_latents_mean and has_latents_std:
    latents_mean = (
        torch.tensor(VAE.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(VAE.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents = latents * latents_std / VAE.config.scaling_factor + latents_mean
else:
    latents = latents / VAE.config.scaling_factor
latents = latents.to(torch.float32)
with torch.no_grad():
    video = VAE.decode(latents, return_dict=False)[0]
video_processor = VideoProcessor(vae_scale_factor=config.vae_spatial_scale_factor)
video = video_processor.postprocess_video(video, output_type='pil')
del VAE
torch.cuda.empty_cache()

out1 = F"editing.mp4"
head, tail = os.path.split(file_name)
output_path = os.path.join(head,tail.split('.')[0]+args.edit_type+f"CA:{CA}"+f"SA:{SA}"+f"threshold:{threshold}"+".mp4")
export_to_video(video[1], output_path, fps = 30)
output_path = os.path.join(head,tail.split('.')[0]+args.edit_type+f"CA:{CA}"+f"SA:{SA}"+f"threshold:{threshold}"+"-regenerated.mp4")
export_to_video(video[0], output_path, fps = 30)
print(F"saved:{output_path}")