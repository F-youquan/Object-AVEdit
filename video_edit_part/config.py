
path_vae = '/video_weight/vae/diffusion_pytorch_model.bf16.safetensors'
path_transformer = '/video_weight/mochi1/transformer'
path_text_encoder = '/video_weight/text_encoder'
path_tokenizer = '/video_weight/tokenizer'

config_vae = {
  "_class_name": "AutoencoderKLMochi",
  "_diffusers_version": "0.32.0.dev0",
  "act_fn": "silu",
  "add_attention_block": [
    False,
    True,
    True,
    True,
    True
  ],
  "decoder_block_out_channels": [
    128,
    256,
    512,
    768
  ],
  "encoder_block_out_channels": [
    64,
    128,
    256,
    384
  ],
  "in_channels": 15,
  "latent_channels": 12,
  "latents_mean": [
    -0.06730895953510081,
    -0.038011381506090416,
    -0.07477820912866141,
    -0.05565264470995561,
    0.012767231469026969,
    -0.04703542746246419,
    0.043896967884726704,
    -0.09346305707025976,
    -0.09918314763016893,
    -0.008729793427399178,
    -0.011931556316503654,
    -0.0321993391887285
  ],
  "latents_std": [
    0.9263795028493863,
    0.9248894543193766,
    0.9393059390890617,
    0.959253732819592,
    0.8244560132752793,
    0.917259975397747,
    0.9294154431013696,
    1.3720942357788521,
    0.881393668867029,
    0.9168315692124348,
    0.9185249279345552,
    0.9274757570805041
  ],
  "layers_per_block": [
    3,
    3,
    4,
    6,
    3
  ],
  "out_channels": 3,
  "scaling_factor": 1.0,
  "spatial_expansions": [
    2,
    2,
    2
  ],
  "temporal_expansions": [
    1,
    2,
    3
  ]
}
vae_spatial_scale_factor = 8
do_classifier_free_guidance = True
timesteps = None
num_videos_per_prompt = 1
generator = None
interrupt = False
max_sequence_length = 256
threshold_noise = 0.025
inversion_guidance_scale = 1.0
guidance_scale = 4.5
num_inference_steps= 64
