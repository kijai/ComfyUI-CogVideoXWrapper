import os
import torch
import folder_paths
import comfy.model_management as mm
from einops import rearrange
from contextlib import nullcontext

from diffusers.schedulers import (
    CogVideoXDDIMScheduler, 
    CogVideoXDPMScheduler, 
    DDIMScheduler, 
    PNDMScheduler, 
    DPMSolverMultistepScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    LCMScheduler
    )

scheduler_mapping = {
    "DPM++": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "CogVideoXDDIM": CogVideoXDDIMScheduler,
    "CogVideoXDPMScheduler": CogVideoXDPMScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "LCMScheduler": LCMScheduler
}
available_schedulers = list(scheduler_mapping.keys())

from .cogvideox_fun.utils import get_image_to_video_latent, get_video_to_video_latent, ASPECT_RATIO_512, get_closest_ratio, to_pil

from PIL import Image
import numpy as np
import json

from .utils import log, check_diffusers_version

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "CogVideo" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("CogVideo", os.path.join(folder_paths.models_dir, "CogVideo"))
if not "cogvideox_loras" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("cogvideox_loras", os.path.join(folder_paths.models_dir, "CogVideo", "loras"))

#PAB
from .videosys.pab import CogVideoXPABConfig

class CogVideoPABConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "spatial_broadcast": ("BOOLEAN", {"default": True, "tooltip": "Enable Spatial PAB, highest impact"}),
            "spatial_threshold_start": ("INT", {"default": 850, "min": 0, "max": 1000, "tooltip": "PAB Start Timestep"} ),
            "spatial_threshold_end": ("INT", {"default": 100, "min": 0, "max": 1000, "tooltip": "PAB End Timestep"} ),
            "spatial_range": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Broadcast timesteps range, higher values are faster but quality may suffer"} ),
            "temporal_broadcast": ("BOOLEAN", {"default": False, "tooltip": "Enable Temporal PAB, medium impact"}),
            "temporal_threshold_start": ("INT", {"default": 850, "min": 0, "max": 1000, "tooltip": "PAB Start Timestep"} ),
            "temporal_threshold_end": ("INT", {"default": 100, "min": 0, "max": 1000, "tooltip": "PAB End Timestep"} ),
            "temporal_range": ("INT", {"default": 4, "min": 0, "max": 10, "tooltip": "Broadcast timesteps range, higher values are faster but quality may suffer"} ),
            "cross_broadcast": ("BOOLEAN", {"default": False, "tooltip": "Enable Cross Attention PAB, low impact"}),
            "cross_threshold_start": ("INT", {"default": 850, "min": 0, "max": 1000, "tooltip": "PAB Start Timestep"} ),
            "cross_threshold_end": ("INT", {"default": 100, "min": 0, "max": 1000, "tooltip": "PAB End Timestep"} ),
            "cross_range": ("INT", {"default": 6, "min": 0, "max": 10, "tooltip": "Broadcast timesteps range, higher values are faster but quality may suffer"} ),

            "steps": ("INT", {"default": 50, "min": 0, "max": 1000, "tooltip": "Should match the sampling steps"} ),
            }
        }

    RETURN_TYPES = ("PAB_CONFIG",)
    RETURN_NAMES = ("pab_config", )
    FUNCTION = "config"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "EXPERIMENTAL:Pyramid Attention Broadcast (PAB) speeds up inference by mitigating redundant attention computation. Increases memory use"

    def config(self, spatial_broadcast, spatial_threshold_start, spatial_threshold_end, spatial_range, 
               temporal_broadcast, temporal_threshold_start, temporal_threshold_end, temporal_range, 
               cross_broadcast, cross_threshold_start, cross_threshold_end, cross_range, steps):
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        pab_config = CogVideoXPABConfig(
            steps=steps, 
            spatial_broadcast=spatial_broadcast, 
            spatial_threshold=[spatial_threshold_end, spatial_threshold_start], 
            spatial_range=spatial_range,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=[temporal_threshold_end, temporal_threshold_start],
            temporal_range=temporal_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=[cross_threshold_end, cross_threshold_start],
            cross_range=cross_range
            )

        return (pab_config, )

class CogVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard", "temporal_tiling"],),
            "context_frames": ("INT", {"default": 48, "min": 2, "max": 100, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            }
        }

    RETURN_TYPES = ("COGCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise
        }

        return (context_options,)

class CogVideoTransformerEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "remove_blocks": ("STRING", {"default": "15, 25, 37", "multiline": True, "tooltip": "Comma separated list of block indices to remove, 5b blocks: 0-41, 2b model blocks 0-29"} ),
            }
        }

    RETURN_TYPES = ("TRANSFORMERBLOCKS",)
    RETURN_NAMES = ("block_list", )
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "EXPERIMENTAL:Remove specific transformer blocks from the model"

    def process(self, remove_blocks):
        blocks_to_remove = [int(x.strip()) for x in remove_blocks.split(',')]
        log.info(f"Blocks selected for removal: {blocks_to_remove}")
        return (blocks_to_remove,)

class CogVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("cogvideox_loras"), 
                {"tooltip": "LORA models are expected to be in ComfyUI/models/CogVideo/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("COGLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            }
        }

    RETURN_TYPES = ("COGLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "CogVideoWrapper"

    def getlorapath(self, lora, strength, prev_lora=None):
        cog_loras_list = []

        cog_lora = {
            "path": folder_paths.get_full_path("cogvideox_loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
        }
        if prev_lora is not None:
            cog_loras_list.extend(prev_lora)
            
        cog_loras_list.append(cog_lora)
        print(cog_loras_list)
        return (cog_loras_list,)

class CogVideoXTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
            },
        }
    RETURN_TYPES = ("COMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
        }

        return (compile_args, )
    
#region TextEncode    
class CogVideoEncodePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, prompt, negative_prompt):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]

        pipe.text_encoder.to(device)
        pipe.transformer.to(offload_device)
        
        positive, negative = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=device,
            dtype=dtype,
        )
        pipe.text_encoder.to(offload_device)

        return (positive, negative)

# Inject clip_l and t5xxl w/ individual strength adjustments for ComfyUI's DualCLIPLoader node for CogVideoX. Use CLIPSave node from any SDXL model then load in a custom clip_l model. 
# For some reason seems to give a lot more movement and consistency on new CogVideoXFun img2vid? set 'type' to flux / DualClipLoader.
class CogVideoDualTextEncode_311:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l": ("STRING", {"default": "", "multiline": True}),
                "t5xxl": ("STRING", {"default": "", "multiline": True}),
                "clip_l_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}), # excessive max for testing, have found intesting results up to 20 max?
                "t5xxl_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}), # setting this to 0.0001 or level as high as 18 seems to work.
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, clip_l, t5xxl, clip_l_strength, t5xxl_strength):
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        # setup tokenizer for clip_l and t5xxl
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = 226
        clip.cond_stage_model.to(load_device)

        # tokenize clip_l and t5xxl
        tokens_l = clip.tokenize(clip_l, return_word_ids=True)
        tokens_t5 = clip.tokenize(t5xxl, return_word_ids=True)

        # encode the tokens separately
        embeds_l = clip.encode_from_tokens(tokens_l, return_pooled=False, return_dict=False)
        embeds_t5 = clip.encode_from_tokens(tokens_t5, return_pooled=False, return_dict=False)

        # apply strength adjustments to each embedding
        if embeds_l.dim() == 3:
            embeds_l *= clip_l_strength
        if embeds_t5.dim() == 3:
            embeds_t5 *= t5xxl_strength

        # combine the embeddings by summing them
        combined_embeds = embeds_l + embeds_t5

        # offload the model to save memory
        clip.cond_stage_model.to(offload_device)

        return (combined_embeds,)
    
class CogVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CLIP",)
    RETURN_NAMES = ("conditioning", "clip")
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, prompt, strength=1.0, force_offload=True):
        max_tokens = 226
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = max_tokens
        clip.cond_stage_model.to(load_device)
        tokens = clip.tokenize(prompt, return_word_ids=True)
        
        embeds = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)

        if embeds.shape[1] > 226:
            raise ValueError(f"Prompt is too long, max tokens supported is {max_tokens} or less, got {embeds.shape[1]}")
        embeds *= strength
        if force_offload:
            clip.cond_stage_model.to(offload_device)

        return (embeds, clip, )
    
class CogVideoTextEncodeCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "combination_mode": (["average", "weighted_average", "concatenate"], {"default": "weighted_average"}),
            "weighted_average_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, conditioning_1, conditioning_2, combination_mode, weighted_average_ratio):
        if conditioning_1.shape != conditioning_2.shape:
            raise ValueError("conditioning_1 and conditioning_2 must have the same shape")

        if combination_mode == "average":
            embeds = (conditioning_1 + conditioning_2) / 2
        elif combination_mode == "weighted_average":
            embeds = conditioning_1 * (1 - weighted_average_ratio) + conditioning_2 * weighted_average_ratio
        elif combination_mode == "concatenate":
            embeds = torch.cat((conditioning_1, conditioning_2), dim=-2)
        else:
            raise ValueError("Invalid combination mode")

        return (embeds, )

#region ImageEncode    
class CogVideoImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "image": ("IMAGE", ),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 16, "min": 4}),
                "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, image, chunk_size=8, enable_tiling=False, mask=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        B, H, W, C = image.shape

        vae = pipeline["pipe"].vae
        vae.enable_slicing()
        
        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        if not pipeline["cpu_offloading"]:
            vae.to(device)

        check_diffusers_version()
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass
        
        input_image = image.clone()
        if mask is not None:
            pipeline["pipe"].original_mask = mask
            # print(mask.shape)
            # mask = mask.repeat(B, 1, 1)  # Shape: [B, H, W]
            # mask = mask.unsqueeze(-1).repeat(1, 1, 1, C)
            # print(mask.shape)
            # input_image = input_image * (1 -mask)
        else:
            pipeline["pipe"].original_mask = None
            
        input_image = input_image * 2.0 - 1.0
        input_image = input_image.to(vae.dtype).to(device)
        input_image = input_image.unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        B, C, T, H, W = input_image.shape

        latents_list = []
        # Loop through the temporal dimension in chunks of 16
        for i in range(0, T, chunk_size):
            # Get the chunk of 16 frames (or remaining frames if less than 16 are left)
            end_index = min(i + chunk_size, T)
            image_chunk = input_image[:, :, i:end_index, :, :]  # Shape: [B, C, chunk_size, H, W]

            # Encode the chunk of images
            latents = vae.encode(image_chunk)

            sample_mode = "sample"
            if hasattr(latents, "latent_dist") and sample_mode == "sample":
                latents = latents.latent_dist.sample(generator)
            elif hasattr(latents, "latent_dist") and sample_mode == "argmax":
                latents = latents.latent_dist.mode()
            elif hasattr(latents, "latents"):
                latents = latents.latents

            latents = vae.config.scaling_factor * latents
            latents = latents.permute(0, 2, 1, 3, 4)  # B, T_chunk, C, H, W
            latents_list.append(latents)

        # Concatenate all the chunks along the temporal dimension
        final_latents = torch.cat(latents_list, dim=1)
        log.info(f"Encoded latents shape: {final_latents.shape}")
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        
        return ({"samples": final_latents}, )
    
class CogVideoImageInterpolationEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "start_image": ("IMAGE", ),
            "end_image": ("IMAGE", ),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, start_image, end_image, chunk_size=8, enable_tiling=False, mask=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        B, H, W, C = start_image.shape

        vae = pipeline["pipe"].vae
        vae.enable_slicing()
        
        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        if not pipeline["cpu_offloading"]:
            vae.to(device)

        check_diffusers_version()
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass
        
        if mask is not None:
            pipeline["pipe"].original_mask = mask
            # print(mask.shape)
            # mask = mask.repeat(B, 1, 1)  # Shape: [B, H, W]
            # mask = mask.unsqueeze(-1).repeat(1, 1, 1, C)
            # print(mask.shape)
            # input_image = input_image * (1 -mask)
        else:
            pipeline["pipe"].original_mask = None
            
        start_image = (start_image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        end_image = (end_image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3)
        B, T, C, H, W = start_image.shape

        latents_list = []           

        # Encode the chunk of images
        start_latents = vae.encode(start_image).latent_dist.sample(generator) * vae.config.scaling_factor
        end_latents = vae.encode(end_image).latent_dist.sample(generator) * vae.config.scaling_factor

        start_latents = start_latents.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
        end_latents = end_latents.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
        latents_list = [start_latents, end_latents]

        # Concatenate all the chunks along the temporal dimension
        final_latents = torch.cat(latents_list, dim=1)
        log.info(f"Encoded latents shape: {final_latents.shape}")
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        
        return ({"samples": final_latents}, )

#region Tora    
from .tora.traj_utils import process_traj, scale_traj_list_to_256
from torchvision.utils import flow_to_image

class ToraEncodeTrajectory:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "tora_model": ("TORAMODEL",),
            "coordinates": ("STRING", {"forceInput": True}),
            "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
            "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
            "num_frames": ("INT", {"default": 49, "min": 2, "max": 1024, "step": 1}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TORAFEATURES", "IMAGE", )
    RETURN_NAMES = ("tora_trajectory", "video_flow_images", )
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, width, height, num_frames, coordinates, strength, start_percent, end_percent, tora_model, enable_tiling=False):
        check_diffusers_version()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        vae = pipeline["pipe"].vae
        vae.enable_slicing()
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass

        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        if len(coordinates) < 10:
            coords_list = []
            for coords in coordinates:
                coords = json.loads(coords.replace("'", '"'))
                coords = [(coord['x'], coord['y']) for coord in coords]
                traj_list_range_256 = scale_traj_list_to_256(coords, width, height)
                coords_list.append(traj_list_range_256)
        else:
            coords = json.loads(coordinates.replace("'", '"'))
            coords = [(coord['x'], coord['y']) for coord in coords]
            coords_list = scale_traj_list_to_256(coords, width, height)
            

        video_flow, points = process_traj(coords_list, num_frames, (height,width), device=device)
        video_flow = rearrange(video_flow, "T H W C -> T C H W")
        video_flow = flow_to_image(video_flow).unsqueeze_(0).to(device)  # [1 T C H W]
        

        video_flow = (
            rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(vae.dtype)
        )
        video_flow_image = rearrange(video_flow, "B C T H W -> (B T) H W C")
        print(video_flow_image.shape)
        mm.soft_empty_cache()

        # VAE encode
        if not pipeline["cpu_offloading"]:
            vae.to(device)

        video_flow = vae.encode(video_flow).latent_dist.sample(generator) * vae.config.scaling_factor

        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)

        video_flow_features = tora_model["traj_extractor"](video_flow.to(torch.float32))
        video_flow_features = torch.stack(video_flow_features)

        video_flow_features = video_flow_features * strength

        log.info(f"video_flow shape: {video_flow.shape}")

        tora = {
            "video_flow_features" : video_flow_features,
            "start_percent" : start_percent,
            "end_percent" : end_percent,
            "traj_extractor" : tora_model["traj_extractor"],
            "fuser_list" : tora_model["fuser_list"],
        }

        return (tora, video_flow_image.cpu().float())

class ToraEncodeOpticalFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "tora_model": ("TORAMODEL",),
            "optical_flow": ("IMAGE", ),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
         
        }

    RETURN_TYPES = ("TORAFEATURES",)
    RETURN_NAMES = ("tora_trajectory",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, optical_flow, strength, tora_model, start_percent, end_percent):
        check_diffusers_version()
        B, H, W, C = optical_flow.shape
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        vae = pipeline["pipe"].vae
        vae.enable_slicing()
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass       

        video_flow = optical_flow * 2 - 1
        video_flow = rearrange(video_flow, "(B T) H W C -> B C T H W", T=B, B=1)
        print(video_flow.shape)
        mm.soft_empty_cache()

        # VAE encode
        if not pipeline["cpu_offloading"]:
            vae.to(device)
        video_flow = video_flow.to(vae.dtype).to(vae.device)
        video_flow = vae.encode(video_flow).latent_dist.sample(generator) * vae.config.scaling_factor
        vae.to(offload_device)

        video_flow_features = tora_model["traj_extractor"](video_flow.to(torch.float32))
        video_flow_features = torch.stack(video_flow_features)

        video_flow_features = video_flow_features * strength

        log.info(f"video_flow shape: {video_flow.shape}")

        tora = {
            "video_flow_features" : video_flow_features,
            "start_percent" : start_percent,
            "end_percent" : end_percent,
            "traj_extractor" : tora_model["traj_extractor"],
            "fuser_list" : tora_model["fuser_list"],
        }

        return (tora, )   
    
def add_noise_to_reference_video(image, ratio=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    
    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

class CogVideoControlImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "control_video": ("IMAGE", ),
            "base_resolution": ("INT", {"min": 64, "max": 1280, "step": 64, "default": 512, "tooltip": "Base resolution, closest training data bucket resolution is chosen based on the selection."}),
            "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("COGCONTROL_LATENTS", "INT", "INT",)
    RETURN_NAMES = ("control_latents", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, control_video, base_resolution, enable_tiling, noise_aug_strength=0.0563):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        B, H, W, C = control_video.shape

        vae = pipeline["pipe"].vae
        vae.enable_slicing()

        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        if not pipeline["cpu_offloading"]:
            vae.to(device)

        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        control_video = np.array(control_video.cpu().numpy() * 255, np.uint8)
        original_width, original_height = Image.fromarray(control_video[0]).size

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        log.info(f"Closest bucket size: {width}x{height}")
        
        video_length = int((B - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if B != 1 else 1
        input_video, input_video_mask, clip_image = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width))

        control_video = pipeline["pipe"].image_processor.preprocess(rearrange(input_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
        control_video = control_video.to(dtype=torch.float32)
        control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)

        masked_image = control_video.to(device=device, dtype=vae.dtype)
        if noise_aug_strength > 0:
            masked_image = add_noise_to_reference_video(masked_image, ratio=noise_aug_strength)
        bs = 1
        new_mask_pixel_values = []
        for i in range(0, masked_image.shape[0], bs):
            mask_pixel_values_bs = masked_image[i : i + bs]
            mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
            mask_pixel_values_bs = mask_pixel_values_bs.mode()
            new_mask_pixel_values.append(mask_pixel_values_bs)
        masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
        masked_image_latents = masked_image_latents * vae.config.scaling_factor      

        vae.to(offload_device)

        control_latents = {
            "latents": masked_image_latents,
            "num_frames" : B,
            "height" : height,
            "width" : width,
        }
        
        return (control_latents, width, height)
            
#region FasterCache
class CogVideoXFasterCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_step": ("INT", {"default": 15, "min": 0, "max": 1024, "step": 1}),
                "hf_step": ("INT", {"default": 30, "min": 0, "max": 1024, "step": 1}),
                "lf_step": ("INT", {"default": 40, "min": 0, "max": 1024, "step": 1}),
                "cache_device": (["main_device", "offload_device", "cuda:1"], {"default": "main_device", "tooltip": "The device to use for the cache, main_device is on GPU and uses a lot of VRAM"}),
            },
        }

    RETURN_TYPES = ("FASTERCACHEARGS",)
    RETURN_NAMES = ("fastercache", )
    FUNCTION = "args"
    CATEGORY = "CogVideoWrapper"

    def args(self, start_step, hf_step, lf_step, cache_device):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if cache_device == "cuda:1":
            device = torch.device("cuda:1")
        fastercache = {
            "start_step" : start_step,
            "hf_step" : hf_step,
            "lf_step" : lf_step,
            "cache_device" : device if cache_device != "offload_device" else offload_device
        }
        return (fastercache,)

#region Sampler    
class CogVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 49, "min": 16, "max": 1024, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (available_schedulers,
                    {
                        "default": 'CogVideoXDDIM'
                    }),
            },
            "optional": {
                "samples": ("LATENT", ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_cond_latents": ("LATENT", ),
                "context_options": ("COGCONTEXT", ),
                "controlnet": ("COGVIDECONTROLNET",),
                "tora_trajectory": ("TORAFEATURES", ),
                "fastercache": ("FASTERCACHEARGS", ),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, steps, cfg, seed, height, width, num_frames, scheduler, samples=None, 
                denoise_strength=1.0, image_cond_latents=None, context_options=None, controlnet=None, tora_trajectory=None, fastercache=None):
        mm.soft_empty_cache()

        base_path = pipeline["base_path"]

        assert "fun" not in base_path.lower(), "'Fun' models not supported in 'CogVideoSampler', use the 'CogVideoXFunSampler'"
        assert ("I2V" not in pipeline.get("model_name","") or num_frames == 49 or context_options is not None), "I2V model can only do 49 frames"

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        scheduler_config = pipeline["scheduler_config"]
        
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(device)
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        if scheduler in scheduler_mapping:
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
            pipe.scheduler = noise_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        if tora_trajectory is not None:
            pipe.transformer.fuser_list = tora_trajectory["fuser_list"]
        
        if context_options is not None:
            context_frames = context_options["context_frames"] // 4
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
        else:
            context_frames, context_stride, context_overlap = None, None, None

        if negative.shape[1] < positive.shape[1]:
            target_length = positive.shape[1]
            padding = torch.zeros((negative.shape[0], target_length - negative.shape[1], negative.shape[2]), device=negative.device)
            negative = torch.cat((negative, padding), dim=1)

        if fastercache is not None:
            pipe.transformer.use_fastercache = True
            pipe.transformer.fastercache_counter = 0
            pipe.transformer.fastercache_start_step = fastercache["start_step"]
            pipe.transformer.fastercache_lf_step = fastercache["lf_step"]
            pipe.transformer.fastercache_hf_step = fastercache["hf_step"]
            pipe.transformer.fastercache_device = fastercache["cache_device"]
        else:
            pipe.transformer.use_fastercache = False
            pipe.transformer.fastercache_counter = 0

        autocastcondition = not pipeline["onediff"] or not dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            latents = pipeline["pipe"](
                num_inference_steps=steps,
                height = height,
                width = width,
                num_frames = num_frames,
                guidance_scale=cfg,
                latents=samples["samples"] if samples is not None else None,
                image_cond_latents=image_cond_latents["samples"] if image_cond_latents is not None else None,
                denoise_strength=denoise_strength,
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                generator=generator,
                device=device,
                context_schedule=context_options["context_schedule"] if context_options is not None else None,
                context_frames=context_frames,
                context_stride= context_stride,
                context_overlap= context_overlap,
                freenoise=context_options["freenoise"] if context_options is not None else None,
                controlnet=controlnet,
                tora=tora_trajectory if tora_trajectory is not None else None,
            )
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(offload_device)

        if fastercache is not None:
            for block in pipe.transformer.transformer_blocks:
                if (hasattr, block, "cached_hidden_states") and block.cached_hidden_states is not None:
                    block.cached_hidden_states = None
                    block.cached_encoder_hidden_states = None
                    
        mm.soft_empty_cache()

        return (pipeline, {"samples": latents})

class CogVideoControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controlnet": ("COGVIDECONTROLNETMODEL",),
            "images": ("IMAGE", ),
            "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("COGVIDECONTROLNET",)
    RETURN_NAMES = ("cogvideo_controlnet",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, controlnet, images, control_strength, control_start_percent, control_end_percent):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        B, H, W, C = images.shape

        control_frames = images.permute(0, 3, 1, 2).unsqueeze(0) * 2 - 1
      
        controlnet = {
            "control_model": controlnet,
            "control_frames": control_frames,
            "control_weights": control_strength,
            "control_start": control_start_percent,
            "control_end": control_end_percent,
        }
        
        return (controlnet,)
    
#region VideoDecode    
class CogVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "samples": ("LATENT", ),
            "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
            },
            "optional": {
            "tile_sample_min_height": ("INT", {"default": 240, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile height, default is half the height"}),
            "tile_sample_min_width": ("INT", {"default": 360, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile width, default is half the width"}),
            "tile_overlap_factor_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
            "tile_overlap_factor_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
            "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Auto size based on height and width, default is half the size"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "CogVideoWrapper"

    def decode(self, pipeline, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width, auto_tile_size=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        vae = pipeline["pipe"].vae

        vae.enable_slicing()

        if not pipeline["cpu_offloading"]:
            vae.to(device)
        if enable_vae_tiling:
            if auto_tile_size:
                vae.enable_tiling()
            else:
                vae.enable_tiling(
                    tile_sample_min_height=tile_sample_min_height,
                    tile_sample_min_width=tile_sample_min_width,
                    tile_overlap_factor_height=tile_overlap_factor_height,
                    tile_overlap_factor_width=tile_overlap_factor_width,
                )
        else:
            vae.disable_tiling()
        latents = latents.to(vae.dtype).to(device)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / vae.config.scaling_factor * latents
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass
        frames = vae.decode(latents).sample
        vae.disable_tiling()
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        mm.soft_empty_cache()

        video = pipeline["pipe"].video_processor.postprocess_video(video=frames, output_type="pt")
        video = video[0].permute(0, 2, 3, 1).cpu().float()

        return (video,)

class CogVideoXFunResizeToClosestBucket:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE", ),
            "base_resolution": ("INT", {"min": 64, "max": 1280, "step": 64, "default": 512, "tooltip": "Base resolution, closest training data bucket resolution is chosen based on the selection."}),
            "upscale_method": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method to use"}),
            "crop": (["disabled","center"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "CogVideoWrapper"

    def resize(self, images, base_resolution, upscale_method, crop):
        from comfy.utils import common_upscale

        B, H, W, C = images.shape
        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        closest_size, closest_ratio = get_closest_ratio(H, W, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        log.info(f"Closest bucket size: {width}x{height}")

        resized_images = images.clone().movedim(-1,1)
        resized_images = common_upscale(resized_images, width, height, upscale_method, crop)
        resized_images = resized_images.movedim(1,-1)
        
        return (resized_images, width, height)

#region FunSamplers
class CogVideoXFunSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "video_length": ("INT", {"default": 49, "min": 5, "max": 2048, "step": 4}),
                "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (available_schedulers, {"default": 'DDIM'})
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
                "noise_aug_strength": ("FLOAT", {"default": 0.0563, "min": 0.0, "max": 1.0, "step": 0.001}),
                "context_options": ("COGCONTEXT", ),
                "tora_trajectory": ("TORAFEATURES", ),
                "fastercache": ("FASTERCACHEARGS",),
                "vid2vid_images": ("IMAGE",),
                "vid2vid_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline,  positive, negative, video_length, width, height, seed, steps, cfg, scheduler, 
                start_img=None, end_img=None, noise_aug_strength=0.0563, context_options=None, fastercache=None, 
                tora_trajectory=None, vid2vid_images=None, vid2vid_denoise=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]
        assert "fun" in base_path.lower(), "'Unfun' models not supported in 'CogVideoXFunSampler', use the 'CogVideoSampler'"
        assert "pose" not in base_path.lower(), "'Pose' models not supported in 'CogVideoXFunSampler', use the 'CogVideoXFunControlSampler'"
        

        if not pipeline["cpu_offloading"]:
            pipe.enable_model_cpu_offload(device=device)

        mm.soft_empty_cache()

        #vid2vid
        if vid2vid_images is not None:
            validation_video = np.array(vid2vid_images.cpu().numpy() * 255, np.uint8)
        #img2vid
        elif start_img is not None:
            start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
            end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None       
        
        # Load Sampler
        if context_options is not None and context_options["context_schedule"] == "temporal_tiling":
            log.info("Temporal tiling enabled, changing scheduler to CogVideoXDDIM")
            scheduler="CogVideoXDDIM"
        scheduler_config = pipeline["scheduler_config"]
        if scheduler in scheduler_mapping:
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
            pipe.scheduler = noise_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        #if not pipeline["cpu_offloading"]:
        #    pipe.transformer.to(device)

        if context_options is not None:
            context_frames = context_options["context_frames"] // 4
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
        else:
            context_frames, context_stride, context_overlap = None, None, None

        if tora_trajectory is not None:
            pipe.transformer.fuser_list = tora_trajectory["fuser_list"]

        if fastercache is not None:
            pipe.transformer.use_fastercache = True
            pipe.transformer.fastercache_counter = 0
            pipe.transformer.fastercache_start_step = fastercache["start_step"]
            pipe.transformer.fastercache_lf_step = fastercache["lf_step"]
            pipe.transformer.fastercache_hf_step = fastercache["hf_step"]
            pipe.transformer.fastercache_device = fastercache["cache_device"]
        else:
            pipe.transformer.use_fastercache = False
            pipe.transformer.fastercache_counter = 0

        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        autocastcondition = not pipeline["onediff"] or not dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            video_length = int((video_length - 1) // pipe.vae.config.temporal_compression_ratio * pipe.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            if vid2vid_images is not None:
                input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width))
            else:
                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            common_params = {
                "prompt_embeds": positive.to(dtype).to(device),
                "negative_prompt_embeds": negative.to(dtype).to(device),
                "num_frames": video_length,
                "height": height,
                "width": width,
                "generator": generator,
                "guidance_scale": cfg,
                "num_inference_steps": steps,
                "comfyui_progressbar": True,
                "context_schedule":context_options["context_schedule"] if context_options is not None else None,
                "context_frames":context_frames,
                "context_stride": context_stride,
                "context_overlap": context_overlap,
                "freenoise":context_options["freenoise"] if context_options is not None else None,
                "tora":tora_trajectory if tora_trajectory is not None else None,
            }
            latents = pipe(
                **common_params,
                video        = input_video,
                mask_video   = input_video_mask,
                noise_aug_strength = noise_aug_strength,
                strength = vid2vid_denoise,
            )
        #if not pipeline["cpu_offloading"]:
        #     pipe.transformer.to(offload_device)
        #clear FasterCache
        if fastercache is not None:
            for block in pipe.transformer.transformer_blocks:
                if (hasattr, block, "cached_hidden_states") and block.cached_hidden_states is not None:
                    block.cached_hidden_states = None
                    block.cached_encoder_hidden_states = None

        mm.soft_empty_cache()

        return (pipeline, {"samples": latents})

class CogVideoXFunVid2VidSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "note": ("STRING", {"default": "This node is deprecated, functionality moved to 'CogVideoXFunSampler' node instead.", "multiline": True}),
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"
    DEPRECATED = True
    def process(self):
        return ()
            
class CogVideoXFunControlSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_latents": ("COGCONTROL_LATENTS",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (available_schedulers, {"default": 'DDIM'}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "samples": ("LATENT", ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "context_options": ("COGCONTEXT", ),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, seed, steps, cfg, scheduler, control_latents, 
                control_strength=1.0, control_start_percent=0.0, control_end_percent=1.0, t_tile_length=16, t_tile_overlap=8, 
                samples=None, denoise_strength=1.0, context_options=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]

        assert "fun" in base_path.lower(), "'Unfun' models not supported in 'CogVideoXFunSampler', use the 'CogVideoSampler'"

        if not pipeline["cpu_offloading"]:
            pipe.enable_model_cpu_offload(device=device)

        mm.soft_empty_cache()

        if context_options is not None:
            context_frames = context_options["context_frames"] // 4
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
        else:
            context_frames, context_stride, context_overlap = None, None, None

        # Load Sampler
        scheduler_config = pipeline["scheduler_config"]
        if context_options is not None and context_options["context_schedule"] == "temporal_tiling":
            log.info("Temporal tiling enabled, changing scheduler to CogVideoXDDIM")
            scheduler="CogVideoXDDIM"
        if scheduler in scheduler_mapping:
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
            pipe.scheduler = noise_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        autocastcondition = not pipeline["onediff"] or not dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:

            common_params = {
                "prompt_embeds": positive.to(dtype).to(device),
                "negative_prompt_embeds": negative.to(dtype).to(device),
                "num_frames": control_latents["num_frames"],
                "height": control_latents["height"],
                "width": control_latents["width"],
                "generator": generator,
                "guidance_scale": cfg,
                "num_inference_steps": steps,
                "comfyui_progressbar": True,
            }

            latents = pipe(
                **common_params,
                control_video=control_latents["latents"],
                control_strength=control_strength,
                control_start_percent=control_start_percent,
                control_end_percent=control_end_percent,
                scheduler_name=scheduler,
                latents=samples["samples"] if samples is not None else None,
                denoise_strength=denoise_strength,
                context_schedule=context_options["context_schedule"] if context_options is not None else None,
                context_frames=context_frames,
                context_stride= context_stride,
                context_overlap= context_overlap,
                freenoise=context_options["freenoise"] if context_options is not None else None
                
            )

        return (pipeline, {"samples": latents})

class CogVideoLatentPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "min_val": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 0.0, "step": 0.001}),
                 "max_val": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "r_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "g_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "b_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("images", "latent_rgb_factors",)
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, samples, seed, min_val, max_val, r_bias, g_bias, b_bias):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()
        print("in sample", latents.shape)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
 
        #[[0.0658900170023352, 0.04687556512203313, -0.056971557475649186], [-0.01265770449940036, -0.02814809569100843, -0.0768912512529372], [0.061456544746314665, 0.0005511617552452358, -0.0652574975291287], [-0.09020669168815276, -0.004755440180558637, -0.023763970904494294], [0.031766964513999865, -0.030959599938418375, 0.08654669098083616], [-0.005981764690055846, -0.08809119252349802, -0.06439852368217663], [-0.0212114426433989, 0.08894281999597677, 0.05155629477559985], [-0.013947446911030725, -0.08987475069900677, -0.08923124751217484], [-0.08235967967978511, 0.07268025379974379, 0.08830486164536037], [-0.08052049179735378, -0.050116143175332195, 0.02023752569687405], [-0.07607527759162447, 0.06827156419895981, 0.08678111754261035], [-0.04689089232553825, 0.017294986041038893, -0.10280492336438908], [-0.06105783150270304, 0.07311850680875913, 0.019995735372550075], [-0.09232589996527711, -0.012869815059053047, -0.04355587834255975], [-0.06679931010802251, 0.018399815879067458, 0.06802404982033876], [-0.013062632927118165, -0.04292991477896661, 0.07476243356192845]]
        latent_rgb_factors =[[0.11945946736445662, 0.09919175788574555, -0.004832707433877734], [-0.0011977028264356232, 0.05496505130267682, 0.021321622433638193], [-0.014088548986590666, -0.008701477861945644, -0.020991313281459367], [0.03063921972519621, 0.12186477097625073, 0.0139593690235148], [0.0927403067854673, 0.030293187650929136, 0.05083134241694003], [0.0379112441305742, 0.04935199882777209, 0.058562766246777774], [0.017749911959153715, 0.008839453404921545, 0.036005638019226294], [0.10610119248526109, 0.02339855688237826, 0.057154257614084596], [0.1273639464837117, -0.010959856130713416, 0.043268631260428896], [-0.01873510946881321, 0.08220930648486932, 0.10613256772247093], [0.008429116376722327, 0.07623856561000408, 0.09295712117576727], [0.12938137079617007, 0.12360403483892413, 0.04478930933220116], [0.04565908794779364, 0.041064156741596365, -0.017695041535528512], [0.00019003240570281826, -0.013965147883381978, 0.05329669529635849], [0.08082391586738358, 0.11548306825496074, -0.021464170006615893], [-0.01517932393230994, -0.0057985555313003236, 0.07216646476618871]]
        import random
        random.seed(seed)
        latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(16)]
        out_factors = latent_rgb_factors
        print(latent_rgb_factors)
       
        latent_rgb_factors_bias = [0.085, 0.137, 0.158]
        #latent_rgb_factors_bias = [r_bias, g_bias, b_bias]
        
        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)
        
        return (latent_images.float().cpu(), out_factors)
    
NODE_CLASS_MAPPINGS = {
    "CogVideoSampler": CogVideoSampler,
    "CogVideoDecode": CogVideoDecode,
    "CogVideoTextEncode": CogVideoTextEncode,
    "CogVideoDualTextEncode_311": CogVideoDualTextEncode_311,
    "CogVideoImageEncode": CogVideoImageEncode,
    "CogVideoImageInterpolationEncode": CogVideoImageInterpolationEncode,
    "CogVideoXFunSampler": CogVideoXFunSampler,
    "CogVideoXFunVid2VidSampler": CogVideoXFunVid2VidSampler,
    "CogVideoXFunControlSampler": CogVideoXFunControlSampler,
    "CogVideoTextEncodeCombine": CogVideoTextEncodeCombine,
    "CogVideoPABConfig": CogVideoPABConfig,
    "CogVideoTransformerEdit": CogVideoTransformerEdit,
    "CogVideoControlImageEncode": CogVideoControlImageEncode,
    "CogVideoLoraSelect": CogVideoLoraSelect,
    "CogVideoContextOptions": CogVideoContextOptions,
    "CogVideoControlNet": CogVideoControlNet,
    "ToraEncodeTrajectory": ToraEncodeTrajectory,
    "ToraEncodeOpticalFlow": ToraEncodeOpticalFlow,
    "CogVideoXFasterCache": CogVideoXFasterCache,
    "CogVideoXFunResizeToClosestBucket": CogVideoXFunResizeToClosestBucket,
    "CogVideoLatentPreview": CogVideoLatentPreview,
    "CogVideoXTorchCompileSettings": CogVideoXTorchCompileSettings
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode",
    "CogVideoTextEncode": "CogVideo TextEncode",
    "CogVideoDualTextEncode_311": "CogVideo DualTextEncode",
    "CogVideoImageEncode": "CogVideo ImageEncode",
    "CogVideoImageInterpolationEncode": "CogVideo ImageInterpolation Encode",
    "CogVideoXFunSampler": "CogVideoXFun Sampler",
    "CogVideoXFunVid2VidSampler": "CogVideoXFun Vid2Vid Sampler",
    "CogVideoXFunControlSampler": "CogVideoXFun Control Sampler",
    "CogVideoTextEncodeCombine": "CogVideo TextEncode Combine",
    "CogVideoPABConfig": "CogVideo PABConfig",
    "CogVideoTransformerEdit": "CogVideo TransformerEdit",
    "CogVideoControlImageEncode": "CogVideo Control ImageEncode",
    "CogVideoLoraSelect": "CogVideo LoraSelect",
    "CogVideoContextOptions": "CogVideo Context Options",
    "ToraEncodeTrajectory": "Tora Encode Trajectory",
    "ToraEncodeOpticalFlow": "Tora Encode OpticalFlow",
    "CogVideoXFasterCache": "CogVideoX FasterCache",
    "CogVideoXFunResizeToClosestBucket": "CogVideoXFun ResizeToClosestBucket",
    "CogVideoLatentPreview": "CogVideo LatentPreview",
    "CogVideoXTorchCompileSettings": "CogVideo TorchCompileSettings",
    }