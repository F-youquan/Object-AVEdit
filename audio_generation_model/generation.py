import sys
sys.path.append("/packages")
from utils import *
from mmbridgedit import *
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from text_encoder import T5Encoder
from transformers import SpeechT5HifiGan
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("prompt",type = str)
parser.add_argument("--GPU_num",type=int)
args = parser.parse_args()


device = f"cuda:{args.GPU_num}" if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
do_classifier_free_guidance = True
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
vocoder = SpeechT5HifiGan.from_pretrained("path_to_AudioLDM_Mel_vocoder").cuda()
scheduler  = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
with torch.no_grad():
    text_in = args.prompt
    text_encoder = T5Encoder(from_pretrained="path_to_T5_text_encoder", model_max_length=300,)
    if do_classifier_free_guidance:
        encoded_prompts = text_encoder.encode(["",text_in])
        text = encoded_prompts.pop('y').to(device,dtype)
        mask = encoded_prompts.pop('mask').to(device,dtype)
    else:
        encoded_prompts = text_encoder.encode(text_in)
        text = encoded_prompts.pop('y').to(device,dtype)
        mask = encoded_prompts.pop('mask').unsqueeze(0).to(device,dtype)
    del text_encoder
torch.cuda.empty_cache()

with torch.no_grad():
    scheduler.set_timesteps(1000, device=device)
    latent = noise = torch.randn([1, 8, 16, 75]).to(device,dtype)
    timesteps = torch.tensor([1000]).cuda()
    scheduler.set_timesteps(100, device=device)
    timesteps = scheduler.timesteps

    for i, t in enumerate(timesteps[:]):
        t = t.unsqueeze(0)
        if do_classifier_free_guidance:
            noisy_model_input = torch.cat([latent, latent], 0)
            t_input = torch.cat([t, t], 0)
        else:
            noisy_model_input = latent
            t_input = t
        noise_pred = dit(noisy_model_input, text, mask,t_input/1000)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        else:   
            noise_pred = noise_pred
        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        latent = latent.to(device, dtype)
latent =  latent*3.126953125
mel = avae.decoder(avae.post_quant_conv(latent.permute(0,1,3,2)))
def mel_spectrogram_to_waveform(mel):
    if len(mel.size()) == 4:
        mel = mel.squeeze(1)
    waveform = vocoder(mel)
    waveform = waveform.cpu().detach().numpy()
    return waveform
def denorm(x):
    return x*1.1940251588821411-4.539258003234863

mel = denorm(mel)
waveform = mel_spectrogram_to_waveform(mel.float().squeeze())
sf.write("result.wav", waveform, samplerate=16000)
