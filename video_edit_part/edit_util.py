from typing import Callable, List, Optional, Tuple, Union
import math
import inspect

import torch
import torch.nn.functional as F
import decord

def inplace_softmax(x: torch.Tensor):
    """
    用一系列原地操作来计算 softmax，以避免双倍内存峰值。
    注意：这个函数会“摧毁”原始输入张量 x。
    
    Args:
        x (torch.Tensor): 需要进行 softmax 的输入张量。
    """
    # 为了数值稳定性，减去最大值。
    # torch.max 返回 (values, indices)，我们只需要 values。
    # 这个操作本身不是原地的，但 max_val 是一个很小的张量，内存开销可忽略。
    max_val = torch.max(x, dim=-1, keepdim=True)[0]
    
    # 1. 原地减法: x = x - max_val
    x.sub_(max_val)
    
    # 2. 原地指数: x = exp(x)
    x.exp_()
    
    # 计算分母。这个操作也不是原地的，但 sum_val 同样是个很小的张量。
    sum_val = torch.sum(x, dim=-1, keepdim=True)
    
    # 3. 原地除法: x = x / sum_val
    # 加上一个很小的数 epsilon 防止除以零
    x.div_(sum_val + 1e-9)
    
    # 函数没有返回值，因为 x 本身已经被修改了

def register_attention_control(model, controller):
    def ca_forward(self):
        def Forward(
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            image_rotary_emb: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if model.edit == False:
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

                query = query.unflatten(2, (attn.heads, -1))
                key = key.unflatten(2, (attn.heads, -1))
                value = value.unflatten(2, (attn.heads, -1))

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

                encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)

                if image_rotary_emb is not None:

                    def apply_rotary_emb(x, freqs_cos, freqs_sin):
                        x_even = x[..., 0::2].float()
                        x_odd = x[..., 1::2].float()

                        cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                        sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                        return torch.stack([cos, sin], dim=-1).flatten(-2)

                    query = apply_rotary_emb(query, *image_rotary_emb)
                    key = apply_rotary_emb(key, *image_rotary_emb)

                query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                encoder_query, encoder_key, encoder_value = (
                    encoder_query.transpose(1, 2),
                    encoder_key.transpose(1, 2),
                    encoder_value.transpose(1, 2),
                )

                sequence_length = query.size(2)
                encoder_sequence_length = encoder_query.size(2)
                total_length = sequence_length + encoder_sequence_length

                batch_size, heads, _, dim = query.shape
                attn_outputs = []
                for idx in range(batch_size):
                    mask = attention_mask[idx][None, :]
                    valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

                    valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
                    valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
                    valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

                    valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
                    valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
                    valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

                    attn_output = F.scaled_dot_product_attention(
                        valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
                    )
                    valid_sequence_length = attn_output.size(2)
                    attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
                    attn_outputs.append(attn_output)

                hidden_states = torch.cat(attn_outputs, dim=0)
                hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

                hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                    (sequence_length, encoder_sequence_length), dim=1
                )

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if hasattr(attn, "to_add_out"):
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                return hidden_states, encoder_hidden_states
            """
            if(controller.cur_att_layer == 1):
                torch.cuda.empty_cache()
                while(1):
                    print('haha')
            """
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            if image_rotary_emb is not None:

                def apply_rotary_emb(x, freqs_cos, freqs_sin):
                    x_even = x[..., 0::2].float()
                    x_odd = x[..., 1::2].float()

                    cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                    sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                    return torch.stack([cos, sin], dim=-1).flatten(-2)

                query = apply_rotary_emb(query, *image_rotary_emb)
                key = apply_rotary_emb(key, *image_rotary_emb)

            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            encoder_query, encoder_key, encoder_value = (
                encoder_query.transpose(1, 2),
                encoder_key.transpose(1, 2),
                encoder_value.transpose(1, 2),
            )

            sequence_length = query.size(2)
            encoder_sequence_length = encoder_query.size(2)
            total_length = sequence_length + encoder_sequence_length

            batch_size, heads, _, dim = query.shape
            attn_outputs = []
            attn_list = []
            valid_prompt_token_indices_list = []
            #print(F"batch_size:{batch_size}")
            for idx in range(batch_size):
                #print(F"idx:{idx}")
                mask = attention_mask[idx][None, :]
                valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
                valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
                valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]

                query_ = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
                key_ = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
                L, S = query_.size(-2), key_.size(-2)
                scale_factor = 1 / math.sqrt(query_.size(-1))
                attn_weight = (query_ @ key_.transpose(-2, -1)).mul_(scale_factor)
                inplace_softmax(attn_weight)
                attn_list.append((attn_weight, valid_prompt_token_indices.shape[0]))
                valid_prompt_token_indices_list.append(valid_prompt_token_indices)
                
            attn_list = controller(attn_list)

            for idx in range(batch_size):
                valid_prompt_token_indices = valid_prompt_token_indices_list[idx]
                valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]
                value_ = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)
                attn_weight = attn_list[idx][0]
                attn_output = (attn_weight @ value_)
                valid_sequence_length = attn_output.size(2)
                attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
                attn_outputs.append(attn_output)
            
            hidden_states = torch.cat(attn_outputs, dim=0)
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                (sequence_length, encoder_sequence_length), dim=1
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if hasattr(attn, "to_add_out"):
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            A = Forward(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            #torch.cuda.empty_cache()
            return A
        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0
    if controller == None:
        controller = DummyController()
    cross_att_count = 0
    def register_recr(net_, count):
        #print(F"net_:{net_}")
        if net_.__class__.__name__ == 'MochiAttention':
            net_.forward = ca_forward(net_)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count)
        return count
    for net in model.named_children():
        if 'transformer_blocks' in net:
            cross_att_count += register_recr(net[1], 0)
    controller.num_att_layers = cross_att_count
    print(F"cross_attention:{cross_att_count}")

def Get_embedding(prompt, negative_prompt):

    tokenizer = T5TokenizerFast.from_pretrained(config.path_tokenizer)
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
class Video_reader():
    def __init__(self,video_path,type = "torch"):
        self.path = video_path
        decord.bridge.set_bridge(type)
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # 使用CPU
        
    def fps(self):
        return self.vr.get_avg_fps()
    
    def video_len(self):
        return len(self.vr)
    
    def read_video(self, frame_indices: List[int]):
        '''
        return [T C W H]
        '''
        video_tensor = self.vr.get_batch(frame_indices).permute(0, 3, 1, 2)
        return video_tensor
def encode_prompt(
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_encoder = None,
        tokenizer = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        #print(F"haha/device:", device)
        device = device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,        
                text_encoder = text_encoder,
                tokenizer = tokenizer,
            )
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = _get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                text_encoder = text_encoder,
                tokenizer = tokenizer,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

def _get_t5_prompt_embeds(
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,        
        text_encoder = None,
        tokenizer = None,
    ):
        device = device
        dtype = dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        #print(F"prompt:{prompt}")
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)
        # The original Mochi implementation zeros out empty negative prompts
        # but this can lead to overflow when placing the entire pipeline under the autocast context
        # adding this here so that we can enable zeroing prompts if necessary
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps