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
import math

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

#from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.loaders import CogVideoXLoraLoaderMixin

from .embeddings import get_3d_rotary_pos_embed
from .custom_cogvideox_transformer_3d import CogVideoXTransformer3DModel
from .enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight

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

class CogVideoXLatentFormat():
    latent_channels = 16
    latent_dimensions = 3
    scale_factor = 0.7
    taesd_decoder_name = None
   
    latent_rgb_factors = [[0.03197404301362048, 0.04091260743347359, 0.0015679806301828524], 
                          [0.005517101026578029, 0.0052348639043457755, -0.005613441650464035], 
                          [0.0012485338264583965, -0.016096744206117782, 0.025023940031635054], 
                          [0.01760126794276171, 0.0036818415416642893, -0.0006019202528157255], 
                          [0.000444954842288864, 0.006102128982092191, 0.0008457999272962447], 
                          [-0.010531904354560697, -0.0032275501924977175, -0.00886595780267917], 
                          [-0.0001454543946122991, 0.010199210750845965, -0.00012702234832386188], 
                          [0.02078497279904325, -0.001669617778939972, 0.006712703698951264], 
                          [0.005529571599763264, 0.009733929789086743, 0.001887302765339838], 
                          [0.012138415094654218, 0.024684961927224837, 0.037211249767461915], 
                          [0.0010364484570000384, 0.01983636315929172, 0.009864602025627755], 
                          [0.006802862648143341, -0.0010509255113510681, -0.007026003345126021], 
                          [0.0003532208468418043, 0.005351971582801936, -0.01845912126717106], 
                          [-0.009045079994694397, -0.01127941143183089, 0.0042294057970470806], 
                          [0.002548289972720752, 0.025224244654428216, -0.0006086130121693347], 
                          [-0.011135669222532816, 0.0018181308593668505, 0.02794541485349922]]
    latent_rgb_factors_bias = [ -0.023, 0.0, -0.017]

class CogVideoXModelPlaceholder():
    def __init__(self):
        self.latent_format = CogVideoXLatentFormat

class CogVideoXPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        dtype: torch.dtype = torch.bfloat16,
        is_fun_inpaint: bool = False,
    ):
        super().__init__()

        self.register_modules(transformer=transformer, scheduler=scheduler)
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.vae_latent_channels = 16
        self.vae_dtype = dtype
        self.is_fun_inpaint = is_fun_inpaint

        self.input_with_padding = True


    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, device, generator, timesteps, denoise_strength,
         num_inference_steps, latents=None, freenoise=True, context_size=None, context_overlap=None
    ):
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        
        noise = randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=self.vae_dtype)
        if freenoise:
            logger.info("Applying FreeNoise")
            # code and comments from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
            video_length = num_frames // 4
            delta = context_size - context_overlap
            for start_idx in range(0, video_length-context_size, delta):
                # start_idx corresponds to the beginning of a context window
                # goal: place shuffled in the delta region right after the end of the context window
                #       if space after context window is not enough to place the noise, adjust and finish
                place_idx = start_idx + context_size
                # if place_idx is outside the valid indexes, we are already finished
                if place_idx >= video_length:
                    break
                end_idx = place_idx - 1
                #print("video_length:", video_length, "start_idx:", start_idx, "end_idx:", end_idx, "place_idx:", place_idx, "delta:", delta)

                # if there is not enough room to copy delta amount of indexes, copy limited amount and finish
                if end_idx + delta >= video_length:
                    final_delta = video_length - place_idx
                    # generate list of indexes in final delta region
                    list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                    # shuffle list
                    list_idx = list_idx[torch.randperm(final_delta, generator=generator)]
                    # apply shuffled indexes
                    noise[:, place_idx:place_idx + final_delta, :, :, :] = noise[:, list_idx, :, :, :]
                    break
                # otherwise, do normal behavior
                # generate list of indexes in delta region
                list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                # shuffle list
                list_idx = list_idx[torch.randperm(delta, generator=generator)]
                # apply shuffled indexes
                #print("place_idx:", place_idx, "delta:", delta, "list_idx:", list_idx)
                noise[:, place_idx:place_idx + delta, :, :, :] = noise[:, list_idx, :, :, :]
        if latents is None:
            latents = noise.to(device)
        elif denoise_strength < 1.0:
            latents = latents.to(device)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, denoise_strength, device)
            latent_timestep = timesteps[:1]
            
            frames_needed = noise.shape[1]
            current_frames = latents.shape[1]
            
            if frames_needed > current_frames:
                repeat_factor = frames_needed - current_frames
                additional_frame = torch.randn((latents.size(0), repeat_factor, latents.size(2), latents.size(3), latents.size(4)), dtype=latents.dtype, device=latents.device)
                latents = torch.cat((additional_frame, latents), dim=1)
                self.additional_frames = repeat_factor
            elif frames_needed < current_frames:
                latents = latents[:, :frames_needed, :, :, :]

            latents = self.scheduler.add_noise(latents, noise.to(device), latent_timestep)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma # scale the initial noise by the standard deviation required by the scheduler
        return latents, timesteps

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

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        if p_t is None:
            # CogVideoX 1.0 I2V
            base_size_width = self.transformer.config.sample_width // p
            base_size_height = self.transformer.config.sample_height // p

            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
            )
        else:
            # CogVideoX 1.5 I2V
            base_size_width = self.transformer.config.sample_width // p
            base_size_height = self.transformer.config.sample_height // p
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
            )

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
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        denoise_strength: float = 1.0,
        sigmas: Optional[List[float]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        fun_mask: Optional[torch.Tensor] = None,
        image_cond_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device = torch.device("cuda"),
        context_schedule: Optional[str] = None,
        context_frames: Optional[int] = None,
        context_stride: Optional[int] = None,
        context_overlap: Optional[int] = None,
        freenoise: Optional[bool] = True,
        controlnet: Optional[dict] = None,
        tora: Optional[dict] = None,
        image_cond_start_percent: float = 0.0,
        image_cond_end_percent: float = 1.0,
        feta_args: Optional[dict] = None,
        
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
        
        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial

        self.num_frames = num_frames

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
        do_classifier_free_guidance = guidance_scale[0] > 1.0

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(self.vae_dtype)

        # 4. Prepare timesteps
        if sigmas is None:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, sigmas=sigmas, device=device)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.vae_latent_channels
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        if patch_size_t is None:
            self.transformer.config.patch_size_t = None
        ofs_embed_dim = getattr(self.transformer.config, "ofs_embed_dim", None)
        if ofs_embed_dim is None:
            self.transformer.config.ofs_embed_dim = None

        self.additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            self.additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += self.additional_frames * self.vae_scale_factor_temporal

        latents, timesteps = self.prepare_latents(
            batch_size,
            latent_channels,
            num_frames,
            height,
            width,
            device,
            generator,
            timesteps,
            denoise_strength,
            num_inference_steps,
            latents,
            context_size=context_frames,
            context_overlap=context_overlap,
            freenoise=freenoise,
        )
        latents = latents.to(self.vae_dtype)

        if self.is_fun_inpaint and fun_mask is None: # For FUN inpaint vid2vid, we need to mask all the latents
            fun_mask = torch.zeros_like(latents[:, :, :1, :, :], device=latents.device, dtype=latents.dtype)
            fun_masked_video_latents = torch.zeros_like(latents, device=latents.device, dtype=latents.dtype)

        # 5.5.
        if image_cond_latents is not None:
            image_cond_frame_count = image_cond_latents.size(1)
            patch_size_t = self.transformer.config.patch_size_t
            if image_cond_frame_count == 2:
                logger.info("More than one image conditioning frame received, interpolating")
                padding_shape = (
                    batch_size,
                    (latents.shape[1] - 2),
                    self.vae_latent_channels,
                    height // self.vae_scale_factor_spatial,
                    width // self.vae_scale_factor_spatial,
                )
                latent_padding = torch.zeros(padding_shape, device=device, dtype=self.vae_dtype)
                image_cond_latents = torch.cat([image_cond_latents[:, 0, :, :, :].unsqueeze(1), latent_padding, image_cond_latents[:, -1, :, :, :].unsqueeze(1)], dim=1)
                if patch_size_t:
                    first_frame = image_cond_latents[:, : image_cond_latents.size(1) % patch_size_t, ...]
                    image_cond_latents = torch.cat([first_frame, image_cond_latents], dim=1)

                logger.info(f"image cond latents shape: {image_cond_latents.shape}")
            elif image_cond_frame_count == 1:
                logger.info("Only one image conditioning frame received, img2vid")
                if self.input_with_padding:
                    padding_shape = (
                        batch_size,
                        (latents.shape[1] - 1),
                        self.vae_latent_channels,
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                    )
                    latent_padding = torch.zeros(padding_shape, device=device, dtype=self.vae_dtype)
                    image_cond_latents = torch.cat([image_cond_latents, latent_padding], dim=1)
                    # Select the first frame along the second dimension
                    if patch_size_t:
                        first_frame = image_cond_latents[:, : image_cond_latents.size(1) % patch_size_t, ...]
                        image_cond_latents = torch.cat([first_frame, image_cond_latents], dim=1)
                else:
                    image_cond_latents = image_cond_latents.repeat(1, latents.shape[1], 1, 1, 1)
            else:
                logger.info(f"Received {image_cond_latents.shape[1]} image conditioning frames")
                if fun_mask is not None and patch_size_t:
                    logger.info(f"1.5 model received {fun_mask.shape[1]} masks")
                    first_frame = image_cond_latents[:, : image_cond_frame_count % patch_size_t, ...]
                    image_cond_latents = torch.cat([first_frame, image_cond_latents], dim=1)
                    fun_mask_first_frame = fun_mask[:, : image_cond_frame_count % patch_size_t, ...]
                    fun_mask = torch.cat([fun_mask_first_frame, fun_mask], dim=1)
                    fun_mask[:, 1:, ...] = 0
            image_cond_latents = image_cond_latents.to(self.vae_dtype)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
      
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7. context schedule
        if context_schedule is not None:
            # if image_cond_latents is not None:
            #     raise NotImplementedError("Context schedule not currently supported with image conditioning")
            logger.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            use_context_schedule = True
            from .context import get_context_scheduler
            context = get_context_scheduler(context_schedule)
            #todo ofs embeds?

        else:
            use_context_schedule = False
            logger.info("Context schedule disabled")
            # 7.5. Create rotary embeds if required
            image_rotary_emb = (
                self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
                if self.transformer.config.use_rotary_positional_embeddings
                else None
            )
            # 7.6. Create ofs embeds if required
            ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

            if tora is not None and do_classifier_free_guidance:
                video_flow_features = tora["video_flow_features"].repeat(1, 2, 1, 1, 1).contiguous()

        #8. Controlnet
        if controlnet is not None:
            self.controlnet = controlnet["control_model"].to(device)
            if self.transformer.dtype == torch.float8_e4m3fn:
                for name, param in self.controlnet.named_parameters():
                    if "patch_embed" not in name and param.data.dtype != torch.float8_e4m3fn:
                        param.data = param.data.to(torch.float8_e4m3fn)
            else:
                self.controlnet.to(self.transformer.dtype)
            
            if getattr(self.transformer, 'fp8_matmul_enabled', False):
                from .fp8_optimization import convert_fp8_linear
                if not hasattr(self.controlnet, 'fp8_matmul_enabled') or not self.controlnet.fp8_matmul_enabled:
                    convert_fp8_linear(self.controlnet, torch.float16)
                    setattr(self.controlnet, "fp8_matmul_enabled", True)
            
            control_frames = controlnet["control_frames"].to(device).to(self.controlnet.dtype).contiguous()
            control_frames = torch.cat([control_frames] * 2) if do_classifier_free_guidance else control_frames
            control_weights = controlnet["control_weights"]
            logger.info(f"Controlnet enabled with weights: {control_weights}")
            control_start = controlnet["control_start"]
            control_end = controlnet["control_end"]
        else:
            controlnet_states = None
            control_weights= None
        # 9. Tora
        if tora is not None:
            trajectory_length = tora["video_flow_features"].shape[1]
            logger.info(f"Tora trajectory length: {trajectory_length}")
            #if trajectory_length != latents.shape[1]:
            #    raise ValueError(f"Tora trajectory length {trajectory_length} does not match inpaint_latents count {latents.shape[2]}")
            for module in self.transformer.fuser_list:
                for param in module.parameters():
                    param.data = param.data.to(self.vae_dtype).to(device)

        logger.info(f"Sampling {num_frames} frames in {latent_frames} latent frames at {width}x{height} with {num_inference_steps} inference steps")

        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            enable_enhance()
        else:
            disable_enhance()

        # reset TeaCache
        if hasattr(self.transformer, 'accumulated_rel_l1_distance'):
            delattr(self.transformer, 'accumulated_rel_l1_distance')
        self.transformer.teacache_counter = 0

        # 11. Denoising loop
        #from .latent_preview import prepare_callback
        #callback = prepare_callback(self.transformer, num_inference_steps)
        from latent_preview import prepare_callback
        self.model = CogVideoXModelPlaceholder()
        self.load_device = device
        callback = prepare_callback(self, num_inference_steps)

        comfy_pbar = ProgressBar(len(timesteps))
        with self.progress_bar(total=len(timesteps)) as progress_bar:    
            old_pred_original_sample = None # for DPM-solver++
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                current_step_percentage = i / num_inference_steps

                if feta_args is not None:
                    if feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enable_enhance()
                    else:
                        disable_enhance()
                # region context schedule sampling
                if use_context_schedule:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    counter = torch.zeros_like(latent_model_input)
                    noise_pred = torch.zeros_like(latent_model_input)

                    if image_cond_latents is not None:
                        latent_image_input = torch.cat([image_cond_latents] * 2) if do_classifier_free_guidance else image_cond_latents

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # use same rotary embeddings for all context windows
                    image_rotary_emb = (
                            self._prepare_rotary_positional_embeddings(height, width, context_frames, device)
                            if self.transformer.config.use_rotary_positional_embeddings
                            else None
                        )

                    context_queue = list(context(
                        i, num_inference_steps, latents.shape[1], context_frames, context_stride, context_overlap,
                    ))

                    if controlnet is not None:
                        # controlnet frames are not temporally compressed, so try to match the context frames that are
                        control_context_queue = list(context(
                            i, 
                            num_inference_steps, 
                            control_frames.shape[1], 
                            context_frames * self.vae_scale_factor_temporal, 
                            context_stride * self.vae_scale_factor_temporal, 
                            context_overlap * self.vae_scale_factor_temporal,
                        ))

                        for c, control_c in zip(context_queue, control_context_queue):
                            partial_latent_model_input = latent_model_input[:, c, :, :, :]
                            partial_control_frames = control_frames[:, control_c, :, :, :]

                            controlnet_states = None
                        
                            if (control_start <= current_step_percentage <= control_end):
                                # extract controlnet hidden state
                                controlnet_states = self.controlnet(
                                    hidden_states=partial_latent_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    image_rotary_emb=image_rotary_emb,
                                    controlnet_states=partial_control_frames,
                                    timestep=timestep,
                                    return_dict=False,
                                )[0]
                                if isinstance(controlnet_states, (tuple, list)):
                                    controlnet_states = [x.to(dtype=self.controlnet.dtype) for x in controlnet_states]
                                else:
                                    controlnet_states = controlnet_states.to(dtype=self.controlnet.dtype)
    
                            # predict noise model_output
                            noise_pred[:, c, :, :, :] += self.transformer(
                                hidden_states=partial_latent_model_input,
                                encoder_hidden_states=prompt_embeds,
                                timestep=timestep,
                                image_rotary_emb=image_rotary_emb,
                                return_dict=False,
                                controlnet_states=controlnet_states,
                                controlnet_weights=control_weights,
                            )[0]

                            counter[:, c, :, :, :] += 1
                            noise_pred = noise_pred.float()
                    else:
                        for c in context_queue:
                            print("c:", c)

                            partial_latent_model_input = latent_model_input[:, c, :, :, :]
                            if image_cond_latents is not None:
                                partial_latent_image_input = latent_image_input[:, :len(c), :, :, :]
                                partial_latent_model_input = torch.cat([partial_latent_model_input,partial_latent_image_input], dim=2)
                            
                            print(partial_latent_model_input.shape)
                            if (tora is not None and tora["start_percent"] <= current_step_percentage <= tora["end_percent"]):
                                if do_classifier_free_guidance:
                                    partial_video_flow_features = tora["video_flow_features"][:, c, :, :, :].repeat(1, 2, 1, 1, 1).contiguous()
                                else:
                                    partial_video_flow_features = tora["video_flow_features"][:, c, :, :, :]
                            else:
                                partial_video_flow_features = None
    
                            # predict noise model_output
                            noise_pred[:, c, :, :, :] += self.transformer(
                                hidden_states=partial_latent_model_input,
                                encoder_hidden_states=prompt_embeds,
                                timestep=timestep,
                                image_rotary_emb=image_rotary_emb,
                                video_flow_features=partial_video_flow_features,
                                return_dict=False
                            )[0]

                            counter[:, c, :, :, :] += 1
                            noise_pred = noise_pred.float()
                        
                    noise_pred /= counter
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self._guidance_scale[i] * (noise_pred_text - noise_pred_uncond)
                       
                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None: 
                            alpha_prod_t = self.scheduler.alphas_cumprod[t]
                            beta_prod_t = 1 - alpha_prod_t
                            callback_tensor = (alpha_prod_t**0.5) * latent_model_input[0][:, :16, :, :] - (beta_prod_t**0.5) * noise_pred.detach()[0]
                            callback(i, callback_tensor * 5, None, num_inference_steps)
                        else:
                            comfy_pbar.update(1)

                # region sampling
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if image_cond_latents is not None:
                        if not image_cond_start_percent <= current_step_percentage <= image_cond_end_percent:
                            latent_image_input = torch.zeros_like(latent_model_input)
                        else:
                            latent_image_input = torch.cat([image_cond_latents] * 2) if do_classifier_free_guidance else image_cond_latents
                        if fun_mask is not None: #for fun img2vid and interpolation
                            fun_inpaint_mask = torch.cat([fun_mask] * 2) if do_classifier_free_guidance else fun_mask
                            masks_input = torch.cat([fun_inpaint_mask, latent_image_input], dim=2)
                            latent_model_input = torch.cat([latent_model_input, masks_input], dim=2)
                        else:
                            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                    else: # for Fun inpaint vid2vid
                        if fun_mask is not None:
                            fun_inpaint_mask = torch.cat([fun_mask] * 2) if do_classifier_free_guidance else fun_mask
                            fun_inpaint_masked_video_latents = torch.cat([fun_masked_video_latents] * 2) if do_classifier_free_guidance else fun_masked_video_latents
                            fun_inpaint_latents = torch.cat([fun_inpaint_mask, fun_inpaint_masked_video_latents], dim=2).to(latents.dtype)
                            latent_model_input = torch.cat([latent_model_input, fun_inpaint_latents], dim=2)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    if controlnet is not None:
                        controlnet_states = None
                        if (control_start <= current_step_percentage <= control_end):
                            # extract controlnet hidden state
                            controlnet_states = self.controlnet(
                                hidden_states=latent_model_input,
                                encoder_hidden_states=prompt_embeds,
                                image_rotary_emb=image_rotary_emb,
                                controlnet_states=control_frames,
                                timestep=timestep,
                                return_dict=False,
                            )[0]
                            if isinstance(controlnet_states, (tuple, list)):
                                controlnet_states = [x.to(dtype=self.vae_dtype) for x in controlnet_states]
                            else:
                                controlnet_states = controlnet_states.to(dtype=self.vae_dtype)

                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        ofs=ofs_emb,
                        return_dict=False,
                        controlnet_states=controlnet_states,
                        controlnet_weights=control_weights,
                        video_flow_features=video_flow_features if (tora is not None and tora["start_percent"] <= current_step_percentage <= tora["end_percent"]) else None,
                    )[0]
                    noise_pred = noise_pred.float()
                    if isinstance(self.scheduler, CogVideoXDPMScheduler):
                        self._guidance_scale[i] = 1 + guidance_scale[i] * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self._guidance_scale[i] * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents.to(self.vae_dtype), **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents.to(self.vae_dtype),
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None: 
                            alpha_prod_t = self.scheduler.alphas_cumprod[t]
                            beta_prod_t = 1 - alpha_prod_t
                            callback_tensor = (alpha_prod_t**0.5) * latent_model_input[0][:, :16, :, :] - (beta_prod_t**0.5) * noise_pred.detach()[0]
                            callback(i, callback_tensor * 5, None, num_inference_steps)
                        else:
                            comfy_pbar.update(1)
            

        # Offload all models
        self.maybe_free_model_hooks()

        return latents