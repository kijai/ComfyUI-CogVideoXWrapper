# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import math

from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed

from comfy.utils import ProgressBar

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
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

class CogVideoXPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        original_mask = None
    ):
        super().__init__()

        self.register_modules(
            vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.original_mask = original_mask
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, timesteps, denoise_strength, num_inference_steps, latents=None, 
    ):
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=self.vae.dtype)
        if latents is None:
            latents = noise       
        else:
            latents = latents.to(device)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, denoise_strength, device)
            latent_timestep = timesteps[:1]
            
            noise = randn_tensor(shape, generator=generator, device=device, dtype=self.vae.dtype)
            frames_needed = noise.shape[1]
            current_frames = latents.shape[1]
            
            if frames_needed > current_frames:
                repeat_factor = frames_needed // current_frames
                additional_frame = torch.randn((latents.size(0), repeat_factor, latents.size(2), latents.size(3), latents.size(4)), dtype=latents.dtype, device=latents.device)
                latents = torch.cat((latents, additional_frame), dim=1)
            elif frames_needed < current_frames:
                latents = latents[:, :frames_needed, :, :, :]

            latents = self.scheduler.add_noise(latents, noise, latent_timestep)
        latents = latents * self.scheduler.init_noise_sigma # scale the initial noise by the standard deviation required by the scheduler
        return latents, timesteps, noise

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps.to(device), num_inference_steps - t_start
    
    def _gaussian_weights(self, t_tile_length, t_batch_size):
        from numpy import pi, exp, sqrt

        var = 0.01
        midpoint = (t_tile_length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        t_probs = [exp(-(t-midpoint)*(t-midpoint)/(t_tile_length*t_tile_length)/(2*var)) / sqrt(2*pi*var) for t in range(t_tile_length)]
        weights = torch.tensor(t_probs)
        weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, t_batch_size,1, 1, 1)
        return weights
    
    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        start_frame: int = None,
        end_frame: int = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )
        
        if start_frame is not None:
            freqs_cos = freqs_cos.view(num_frames, grid_height * grid_width, -1)
            freqs_sin = freqs_sin.view(num_frames, grid_height * grid_width, -1)

            freqs_cos = freqs_cos[start_frame:end_frame]
            freqs_sin = freqs_sin[start_frame:end_frame]

            freqs_cos = freqs_cos.view(-1, freqs_cos.shape[-1])
            freqs_sin = freqs_sin.view(-1, freqs_sin.shape[-1])

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        height: int = 480,
        width: int = 720,
        num_frames: int = 48,
        t_tile_length: int = 12,
        t_tile_overlap: int = 4,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        denoise_strength: float = 1.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device = torch.device("cuda"),
        scheduler_name: str = "DPM",
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """

        #assert (
        #    num_frames <= 48 and num_frames % fps == 0 and fps == 8
        #), f"The number of frames must be divisible by {fps=} and less than 48 frames (for now). Other values are not supported in CogVideoX."

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
       
        batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(self.transformer.dtype)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels

        if latents is None and num_frames == t_tile_length:
            num_frames += 1

        if self.original_mask is not None:
            image_latents = latents
            original_image_latents = image_latents

        latents, timesteps, noise = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            timesteps,
            denoise_strength,
            num_inference_steps,
            latents
        )
        latents = latents.to(self.transformer.dtype)
       
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.5. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # masks
        if self.original_mask is not None:
            mask = self.original_mask.to(device)
            print("self.original_mask: ", self.original_mask.shape)
            
            mask = F.interpolate(self.original_mask.unsqueeze(1), size=(latents.shape[-2], latents.shape[-1]), mode='bilinear', align_corners=False)
            if mask.shape[0] != latents.shape[1]:
                mask = mask.unsqueeze(1).repeat(1, latents.shape[1], 16, 1, 1)
            else:
                mask = mask.unsqueeze(0).repeat(1, 1, 16, 1, 1)
            print("latents: ", latents.shape)
            print("mask: ", mask.shape)

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        comfy_pbar = ProgressBar(num_inference_steps)

        # 8. Temporal tiling prep
        if "tiled" in scheduler_name:
            t_tile_weights = self._gaussian_weights(t_tile_length=t_tile_length, t_batch_size=1).to(latents.device).to(self.vae.dtype)
            temporal_tiling = True
            print("Temporal tiling enabled")
        else:
            temporal_tiling = False
            print("Temporal tiling disabled")
        print("latents.shape", latents.shape)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:    
            old_pred_original_sample = None # for DPM-solver++
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if temporal_tiling and isinstance(self.scheduler, CogVideoXDDIMScheduler):
                    #temporal tiling code based on https://github.com/mayuelala/FollowYourEmoji/blob/main/models/video_pipeline.py
                    # =====================================================
                    grid_ts = 0
                    cur_t = 0
                    while cur_t < latents.shape[1]:
                        cur_t = max(grid_ts * t_tile_length - t_tile_overlap * grid_ts, 0) + t_tile_length
                        grid_ts += 1

                    all_t = latents.shape[1]
                    latents_all_list = []
                    # =====================================================

                    for t_i in range(grid_ts):
                        if t_i < grid_ts - 1:
                            ofs_t = max(t_i * t_tile_length - t_tile_overlap * t_i, 0)
                        if t_i == grid_ts - 1:
                            ofs_t = all_t - t_tile_length

                        input_start_t = ofs_t
                        input_end_t = ofs_t + t_tile_length

                        #latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        image_rotary_emb = (
                            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device, input_start_t, input_end_t)
                            if self.transformer.config.use_rotary_positional_embeddings
                            else None
                        )

                        latents_tile = latents[:, input_start_t:input_end_t,:, :, :]
                        latent_model_input_tile = torch.cat([latents_tile] * 2) if do_classifier_free_guidance else latents_tile
                        latent_model_input_tile = self.scheduler.scale_model_input(latent_model_input_tile, t)

                        #t_input = t[None].to(device)
                        t_input = t.expand(latent_model_input_tile.shape[0]) # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                
                        # predict noise model_output
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input_tile,
                            encoder_hidden_states=prompt_embeds,
                            timestep=t_input,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred.float()                  

                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_tile = self.scheduler.step(noise_pred, t, latents_tile.to(self.vae.dtype), **extra_step_kwargs, return_dict=False)[0]            
                        latents_all_list.append(latents_tile)

                    # ==========================================
                    latents_all = torch.zeros(latents.shape, device=latents.device, dtype=self.vae.dtype)
                    contributors = torch.zeros(latents.shape, device=latents.device, dtype=self.vae.dtype)
                    # Add each tile contribution to overall latents
                    for t_i in range(grid_ts):
                        if t_i < grid_ts - 1:
                            ofs_t = max(t_i * t_tile_length - t_tile_overlap * t_i, 0)
                        if t_i == grid_ts - 1:
                            ofs_t = all_t - t_tile_length

                        input_start_t = ofs_t
                        input_end_t = ofs_t + t_tile_length

                        latents_all[:, input_start_t:input_end_t,:, :, :] += latents_all_list[t_i] * t_tile_weights
                        contributors[:, input_start_t:input_end_t,:, :, :] += t_tile_weights
                    
                    latents_all /= contributors

                    latents = latents_all
                    #print("latents",latents.shape)
                    # start diff diff
                    if i < len(timesteps) - 1 and self.original_mask is not None:
                        noise_timestep = timesteps[i + 1]
                        image_latent = self.scheduler.add_noise(original_image_latents, noise, torch.tensor([noise_timestep])
                        )
                        mask = mask.to(latents)
                        ts_from = timesteps[0]
                        ts_to = timesteps[-1]
                        threshold = (t - ts_to) / (ts_from - ts_to)
                        mask = torch.where(mask >= threshold, mask, torch.zeros_like(mask))
                        latents = image_latent * mask + latents * (1 - mask)
                        # end diff diff
                    
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        comfy_pbar.update(1)
                    # ==========================================
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()

                    if isinstance(self.scheduler, CogVideoXDPMScheduler):
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents.to(self.vae.dtype), **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents.to(self.vae.dtype),
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    # start diff diff
                    if i < len(timesteps) - 1 and self.original_mask is not None:
                        noise_timestep = timesteps[i + 1]
                        image_latent = self.scheduler.add_noise(original_image_latents, noise, torch.tensor([noise_timestep])
                        )
                        mask = mask.to(latents)
                        ts_from = timesteps[0]
                        ts_to = timesteps[-1]
                        threshold = (t - ts_to) / (ts_from - ts_to)
                        mask = torch.where(mask >= threshold, mask, torch.zeros_like(mask))
                        latents = image_latent * mask + latents * (1 - mask)
                        # end diff diff

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        comfy_pbar.update(1)
            

        # Offload all models
        self.maybe_free_model_hooks()

        return latents
