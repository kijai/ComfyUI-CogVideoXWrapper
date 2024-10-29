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
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange

from ..videosys.core.pipeline import VideoSysPipeline
from ..videosys.cogvideox_transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelPAB
from ..videosys.core.pab_mgr import set_pab_manager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoX_Fun_Pipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoX_Fun_Pipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
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


@dataclass
class CogVideoX_Fun_PipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class CogVideoX_Fun_Pipeline_Control(VideoSysPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "vae->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        pab_config = None
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

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        if pab_config is not None:
            set_pab_manager(pab_config)

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, timesteps, denoise_strength, num_inference_steps,
         latents=None, freenoise=True, context_size=None, context_overlap=None
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
        noise = randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=self.vae.dtype)
        if freenoise:
            print("Applying FreeNoise")
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

    def prepare_control_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

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
    
    def _gaussian_weights(self, t_tile_length, t_batch_size):
        from numpy import pi, exp, sqrt

        var = 0.01
        midpoint = (t_tile_length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        t_probs = [exp(-(t-midpoint)*(t-midpoint)/(t_tile_length*t_tile_length)/(2*var)) / sqrt(2*pi*var) for t in range(t_tile_length)]
        weights = torch.tensor(t_probs)
        weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, t_batch_size,1, 1, 1)
        return weights

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

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
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        context_frames: Optional[int] = None,
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
        
        if start_frame is not None or context_frames is not None:
            freqs_cos = freqs_cos.view(num_frames, grid_height * grid_width, -1)
            freqs_sin = freqs_sin.view(num_frames, grid_height * grid_width, -1)
            if context_frames is not None:
                freqs_cos = freqs_cos[context_frames]
                freqs_sin = freqs_sin[context_frames]
            else:
                freqs_cos = freqs_cos[start_frame:end_frame]
                freqs_sin = freqs_sin[start_frame:end_frame]

            freqs_cos = freqs_cos.view(-1, freqs_cos.shape[-1])
            freqs_sin = freqs_sin.view(-1, freqs_sin.shape[-1])

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        control_video: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        denoise_strength: float = 1.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        comfyui_progressbar: bool = False,
        control_strength: float = 1.0,
        control_start_percent: float = 0.0,
        control_end_percent: float = 1.0,
        scheduler_name: str = "DPM",
        context_schedule: Optional[str] = None,
        context_frames: Optional[int] = None,
        context_stride: Optional[int] = None,
        context_overlap: Optional[int] = None,
        freenoise: Optional[bool] = True,
        tora: Optional[dict] = None,
    ) -> Union[CogVideoX_Fun_PipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX_Fun is conditioned with (num_seconds * fps + 1) frames where
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoX_Fun_PipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoX_Fun_PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        # if num_frames > 49:
        #     raise ValueError(
        #         "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
        #     )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        latents, timesteps, noise = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            self.vae.dtype,
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
        if comfyui_progressbar:
            pbar.update(1)


        control_video_latents_input = (
            torch.cat([control_video] * 2) if do_classifier_free_guidance else control_video
        )
        control_latents = rearrange(control_video_latents_input, "b c f h w -> b f c h w")

        control_latents = control_latents * control_strength

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.5. Temporal tiling prep
        if context_schedule is not None and context_schedule == "temporal_tiling":
            t_tile_length = context_frames
            t_tile_overlap = context_overlap
            t_tile_weights = self._gaussian_weights(t_tile_length=t_tile_length, t_batch_size=1).to(latents.device).to(self.vae.dtype)
            use_temporal_tiling = True
            print("Temporal tiling enabled")
        elif context_schedule is not None:
            print(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            use_temporal_tiling = False
            use_context_schedule = True
            from .context import get_context_scheduler
            context = get_context_scheduler(context_schedule)

        else:
            use_temporal_tiling = False
            use_context_schedule = False
            print("Temporal tiling and context schedule disabled")
            # 7. Create rotary embeds if required
            image_rotary_emb = (
                self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
                if self.transformer.config.use_rotary_positional_embeddings
                else None
            )
            if tora is not None and do_classifier_free_guidance:
                video_flow_features = tora["video_flow_features"].repeat(1, 2, 1, 1, 1).contiguous()

        if tora is not None:
            for module in self.transformer.fuser_list:
                for param in module.parameters():
                    param.data = param.data.to(device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if use_temporal_tiling and isinstance(self.scheduler, CogVideoXDDIMScheduler):
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

                    image_rotary_emb = (
                            self._prepare_rotary_positional_embeddings(height, width, context_frames, device)
                            if self.transformer.config.use_rotary_positional_embeddings
                            else None
                        )

                    for t_i in range(grid_ts):
                        if t_i < grid_ts - 1:
                            ofs_t = max(t_i * t_tile_length - t_tile_overlap * t_i, 0)
                        if t_i == grid_ts - 1:
                            ofs_t = all_t - t_tile_length

                        input_start_t = ofs_t
                        input_end_t = ofs_t + t_tile_length

                        latents_tile = latents[:, input_start_t:input_end_t,:, :, :]
                        control_latents_tile = control_latents[:, input_start_t:input_end_t, :, :, :]

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
                            control_latents=control_latents_tile,
                        )[0]
                        noise_pred = noise_pred.float()                  

                        if do_classifier_free_guidance:
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
                    
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        pbar.update(1)
                    # ==========================================
                elif use_context_schedule:
                    
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Calculate the current step percentage
                    current_step_percentage = i / num_inference_steps

                    # Determine if control_latents should be applied
                    apply_control = control_start_percent <= current_step_percentage <= control_end_percent
                    current_control_latents = control_latents if apply_control else torch.zeros_like(control_latents)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    context_queue = list(context(
                        i, num_inference_steps, latents.shape[1], context_frames, context_stride, context_overlap,
                    ))
                    counter = torch.zeros_like(latent_model_input)
                    noise_pred = torch.zeros_like(latent_model_input)

                    image_rotary_emb = (
                            self._prepare_rotary_positional_embeddings(height, width, context_frames, device)
                            if self.transformer.config.use_rotary_positional_embeddings
                            else None
                        )

                    for c in context_queue:
                        partial_latent_model_input = latent_model_input[:, c, :, :, :]
                        partial_control_latents = current_control_latents[:, c, :, :, :]

                        # predict noise model_output
                        noise_pred[:, c, :, :, :] += self.transformer(
                            hidden_states=partial_latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                            control_latents=partial_control_latents,
                        )[0]

                        counter[:, c, :, :, :] += 1
                        noise_pred = noise_pred.float()
                        
                    noise_pred /= counter
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                       
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

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    if comfyui_progressbar:
                        pbar.update(1)
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Calculate the current step percentage
                    current_step_percentage = i / num_inference_steps

                    # Determine if control_latents should be applied
                    apply_control = control_start_percent <= current_step_percentage <= control_end_percent
                    current_control_latents = control_latents if apply_control else torch.zeros_like(control_latents)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                        control_latents=current_control_latents,
                        video_flow_features=video_flow_features if (tora is not None and tora["start_percent"] <= current_step_percentage <= tora["end_percent"]) else None,

                    )[0]
                    noise_pred = noise_pred.float()

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

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

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    if comfyui_progressbar:
                        pbar.update(1)

        # if output_type == "numpy":
        #     video = self.decode_latents(latents)
        # elif not output_type == "latent":
        #     video = self.decode_latents(latents)
        #     video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        # else:
        #     video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     video = torch.from_numpy(video)

        return latents