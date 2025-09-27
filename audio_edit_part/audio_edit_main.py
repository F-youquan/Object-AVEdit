import sys
sys.path.append("/packages")
from utils import *
from mmbridgedit import *
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from text_encoder import T5Encoder
from transformers import SpeechT5HifiGan
import soundfile as sf
import matplotlib.pyplot as plt
from Controller1 import *
from transformers import AutoTokenizer
from p2p_util import register_attention_control
from tqdm import tqdm
import argparse
import torchaudio
parser = argparse.ArgumentParser()
parser.add_argument("--input_audio",type = str)
parser.add_argument("--edit_type",type = str)
parser.add_argument("--GPU_num",type=int)
parser.add_argument("--source_prompt",type=str)
parser.add_argument("--target_prompt",type=str)
parser.add_argument("--word",type=str,default=None)
parser.add_argument("--seed",type=int,default=20050329)
parser.add_argument("--CA",type=int,default=37)
parser.add_argument("--SA",type=int,default=37)
parser.add_argument("--threshold",type=float,default=0.1)

args = parser.parse_args()
torch.manual_seed(args.seed)

output_father_folder = os.path.split(args.input_audio)[0]
folder_name = os.path.split(args.input_audio)[1].split('.')[0]
output_folder = os.path.join(output_father_folder,folder_name)
os.makedirs(output_folder,exist_ok=True)

device = f"cuda:{args.GPU_num}" if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
inference_step = 200
guidance_scale = 7.5
inversion_guidance = 1.0
sample_rate = 16000
duration = 3.0
do_classifier_free_guidance = True
L = 2

with open("/audio_weight/config.json", 'r') as f:
    vae_config = json.load(f)
avae = AutoencoderKL(**vae_config)
quant_conv = avae.quant_conv
post_quant_conv = avae.post_quant_conv
encoder = avae.encoder
decoder = avae.decoder
avae = audio_VAE(encoder, quant_conv, post_quant_conv, decoder)
config = MMBridgeDiTConfig(depth = 96,hidden_size=1024, apatch_size=(1,2), ain_channels = 8, num_heads=16)
dit = MMBridgeDiT(config)
composite_model = CompositeModel(avae, dit)
model_dir = "/audio_weight"
shard_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")]
full_state_dict = {}
for shard_file in shard_files:
    shard_state_dict = torch.load(shard_file, map_location=lambda storage, loc: storage)
    full_state_dict.update(shard_state_dict)
composite_model.load_state_dict(full_state_dict)
composite_model.to(device,dtype)

vocoder = SpeechT5HifiGan.from_pretrained("path_to_AudioLDM_vocoder").cuda()
wave,sr = torchaudio.load(args.input_audio) # 第一个元素是音频tensor #第二个元素是其采样率
wave = torchaudio.functional.resample(wave, sr, 16000)
if len(wave.shape) >= 2 and wave.shape[1] == 2:
    wave = wave.mean(axis=1)  # 对两个通道的音频取平均
transform = GetMelTransform()
mel = transform((wave, 3.0))
mel = mel[:300].unsqueeze(0).unsqueeze(0)

plt.imshow(mel[0][0].float())  
plt.axis('off')
out_path = F"{output_folder}/original.png"
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

if args.edit_type == "Addition":
    prompts = [args.source_prompt,args.target_prompt]
    tokenizer = AutoTokenizer.from_pretrained('path_to_t5_large_text_encoder')
    controller = AttentionDelete(prompts, inference_step, tokenizer, torch.float16, args.CA, args.SA, True, None)

elif args.edit_type == "Replacement":
    prompts = [args.source_prompt,args.target_prompt]
    tokenizer = AutoTokenizer.from_pretrained('path_to_t5_large_text_encoder')
    lb = LocalBlend(prompts, (args.word, None), tokenizer, args.threshold)
    controller = AttentionReplace(prompts, inference_step, tokenizer, torch.float16, args.CA, args.SA, True, lb)

elif args.edit_type == "Removal":
    prompts = [args.source_prompt,args.target_prompt]
    tokenizer = AutoTokenizer.from_pretrained('path_to_t5_large_text_encoder')
    lb = LocalBlend(prompts, (args.word, None), tokenizer, args.threshold)
    controller = AttentionDelete(prompts, inference_step, tokenizer, torch.float16, args.CA, args.SA, True, lb)
else:
    raise KeyError()


def Text_encoder(text_in):
    text_encoder = T5Encoder(from_pretrained="path_to_t5_large_text_encoder", model_max_length=300,)
    output = [''] * len(text_in)
    for str1 in text_in:
        output.append(str1)
    encoded_prompts = text_encoder.encode(output)
    text = encoded_prompts.pop('y').to(device,dtype)
    mask = encoded_prompts.pop('mask').to(device,dtype)
    del text_encoder
    torch.cuda.empty_cache()
    return text, mask

def denorm(x):
    return x*1.1940251588821411-4.539258003234863
def ennorm(x):
    return (x + 4.539258003234863) / 1.1940251588821411


text_, mask_ = Text_encoder(prompts)
text, mask = torch.cat((text_[0:1], text_[len(prompts):len(prompts) + 1])), torch.cat((mask_[0:1], mask_[len(prompts):len(prompts) + 1]))
inference_step = 200
latent = (avae.quant_conv(avae.encoder(ennorm(mel).to(device, dtype))))
posterior = DiagonalGaussianDistribution(latent)
latent = posterior.sample()
latent = (latent / 3.126953125).permute(0, 1, 3, 2)
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(inference_step, device=device)
A = linear_quadratic_schedule(inference_step, 0.025)
A.append(0)
A = torch.tensor(A, device = device)
Sigmas = scheduler.sigmas
sigmas = Sigmas.flip(dims = [0])


with torch.no_grad():
    with tqdm(total = inference_step) as progress_bar:
        for i in range(inference_step):
            cur_sigmas, nex_sigmas = sigmas[i], sigmas[i + 1]
            mid_sigmas = (sigmas[i] + sigmas[i + 1]) / 2.0
            latent_ = latent.clone()
            Latent_list = []
            for _ in range(L):
                noisy_model_input = torch.cat([latent_, latent_]) if do_classifier_free_guidance else latent_            
                t_input = (nex_sigmas * 1000).expand(noisy_model_input.shape[0])
                noise_pred2 = dit(noisy_model_input, text, mask, t_input / 1000)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred2.chunk(2)
                    noise_pred2 = noise_pred_uncond + inversion_guidance * (noise_pred_text - noise_pred_uncond)
                latent_ = latent + (nex_sigmas - cur_sigmas) * noise_pred2
                latent_ = latent_.to(dtype)
                Latent_list.append(latent_)
            latent = torch.mean(torch.stack(Latent_list), dim=0)
            progress_bar.update()

latent = latent.repeat(len(prompts), 1, 1, 1)
text, mask = text_, mask_
register_attention_control(dit, controller)
sigmas = Sigmas
with torch.no_grad():
    from tqdm import tqdm
    with tqdm(total = inference_step) as progress_bar:
        for i in range(inference_step):
            cur_sigmas, nex_sigmas = sigmas[i], sigmas[i + 1]
            mid_sigmas, delta_sigmas = (cur_sigmas + nex_sigmas) / 2.0, (nex_sigmas - cur_sigmas) / 2.0
            noisy_model_input = torch.cat([latent, latent]) if do_classifier_free_guidance else latent
            t_input = (cur_sigmas * 1000).expand(noisy_model_input.shape[0]).to(dtype)
            noise_pred_0 = dit(noisy_model_input, text, mask, t_input / 1000)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_0.chunk(2)
                noise_pred_0 = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            mid_latent = latent + (mid_sigmas - cur_sigmas) * noise_pred_0
            mid_latent = controller.step_callback(mid_latent)

            mid_latent = mid_latent.to(device, dtype)
            noisy_model_input = torch.cat([mid_latent, mid_latent]) if do_classifier_free_guidance else latent
            t_input = (mid_sigmas * 1000).expand(noisy_model_input.shape[0])
            noise_pred_1 = dit(noisy_model_input, text, mask, t_input / 1000)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_1.chunk(2)
                noise_pred_1 = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            two_del_noise = (noise_pred_1 - noise_pred_0) / (mid_sigmas * 1000 - cur_sigmas * 1000)
            latent = latent + (nex_sigmas - cur_sigmas) * noise_pred_0 + 0.5 * (nex_sigmas - cur_sigmas) * (nex_sigmas - cur_sigmas) * two_del_noise
            latent = controller.step_callback(latent)
            latent = latent.to(device, dtype)
            progress_bar.update()
del dit

latent =  latent * 3.126953125
torch.cuda.empty_cache()
mel = avae.decoder(avae.post_quant_conv(latent.permute(0, 1, 3, 2)))

def mel_spectrogram_to_waveform(mel):
    # Mel: [bs, 1, t-steps, fbins]
    if len(mel.size()) == 4:
        mel = mel.squeeze(1)
    waveform = vocoder(mel)
    waveform = waveform.cpu().detach().numpy()
    return waveform

# save mel 
mel = denorm(mel)
outmel = mel.clone()
plt.imshow(outmel[0][0].float().detach().cpu())  
plt.axis('off')
out_path = F"{output_folder}/regenerated.png"
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
plt.clf()
plt.imshow(outmel[1][0].float().detach().cpu()) 
plt.axis('off')  
out_path = F"{output_folder}/edited.png"
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
plt.clf()
waveform = mel_spectrogram_to_waveform(mel.float())
out_path = F"{output_folder}/regenerated.wav"
sf.write(out_path, waveform[0], samplerate=16000)
out_path = F"{output_folder}/edited.wav"
sf.write(out_path, waveform[1], samplerate=16000)
