import os
import torch
import json
from einops import rearrange
from contextlib import nullcontext

from .utils import log, check_diffusers_version, print_memory
check_diffusers_version()
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

from diffusers.video_processor import VideoProcessor

import folder_paths
import comfy.model_management as mm

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "CogVideo" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("CogVideo", os.path.join(folder_paths.models_dir, "CogVideo"))
if not "cogvideox_loras" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("cogvideox_loras", os.path.join(folder_paths.models_dir, "CogVideo", "loras"))

class CogVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        return (kwargs, )

class CogVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
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

        if embeds.shape[1] > max_tokens:
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

class CogVideoImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE",),
            "start_image": ("IMAGE", ),
            },
            "optional": {
                "end_image": ("IMAGE", ),
                "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Augment image with noise"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, vae, start_image, end_image=None, enable_tiling=False, noise_aug_strength=0.0, strength=1.0, start_percent=0.0, end_percent=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        
        try:
            vae.enable_slicing()
        except:
            pass
       
        vae_scaling_factor = vae.config.scaling_factor
        
        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        vae.to(device)

        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass
        
       
        latents_list = []

        start_image = (start_image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if noise_aug_strength > 0:
            start_image = add_noise_to_reference_video(start_image, ratio=noise_aug_strength)
        start_latents = vae.encode(start_image).latent_dist.sample(generator)
        start_latents = start_latents.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
        

        if end_image is not None:
            end_image = (end_image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3)
            if noise_aug_strength > 0:
                end_image = add_noise_to_reference_video(end_image, ratio=noise_aug_strength)
            end_latents = vae.encode(end_image).latent_dist.sample(generator)
            end_latents = end_latents.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
            latents_list = [start_latents, end_latents]
            final_latents = torch.cat(latents_list, dim=1)
        else:
            final_latents = start_latents

        final_latents = final_latents * vae_scaling_factor * strength
        
        log.info(f"Encoded latents shape: {final_latents.shape}")
        vae.to(offload_device)
        
        return ({
            "samples": final_latents,
            "start_percent": start_percent,
            "end_percent": end_percent
            }, )
    
class CogVideoImageEncodeFunInP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE",),
            "start_image": ("IMAGE", ),
            "num_frames": ("INT", {"default": 49, "min": 2, "max": 1024, "step": 1}),
            },
            "optional": {
                "end_image": ("IMAGE", ),
                "enable_tiling": ("BOOLEAN", {"default": False, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Augment image with noise"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("image_cond_latents",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, vae, start_image, num_frames, end_image=None, enable_tiling=False, noise_aug_strength=0.0):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        
        try:
            vae.enable_slicing()
        except:
            pass
       
        vae_scaling_factor = vae.config.scaling_factor
        
        if enable_tiling:
            from .mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        vae.to(device)

        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass
        
        if end_image is not None:
            # Create a tensor of zeros for padding
            padding = torch.zeros((num_frames - 2, start_image.shape[1], start_image.shape[2], 3), device=end_image.device, dtype=end_image.dtype) * -1
            # Concatenate start_image, padding, and end_image
            input_image = torch.cat([start_image, padding, end_image], dim=0)
        else:
            # Create a tensor of zeros for padding
            padding = torch.zeros((num_frames - 1, start_image.shape[1], start_image.shape[2], 3), device=start_image.device, dtype=start_image.dtype) * -1
            # Concatenate start_image and padding
            input_image = torch.cat([start_image, padding], dim=0) 
        
        input_image = input_image * 2.0 - 1.0
        input_image = input_image.to(vae.dtype).to(device)
        input_image = input_image.unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
      
        B, C, T, H, W = input_image.shape
        if noise_aug_strength > 0:
            input_image = add_noise_to_reference_video(input_image, ratio=noise_aug_strength)
        
        bs = 1
        new_mask_pixel_values = []
        for i in range(0, input_image.shape[0], bs):
            mask_pixel_values_bs = input_image[i : i + bs]
            mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
            mask_pixel_values_bs = mask_pixel_values_bs.mode()
            new_mask_pixel_values.append(mask_pixel_values_bs)
        masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
        masked_image_latents = masked_image_latents.permute(0, 2, 1, 3, 4)  # B, T, C, H, W

        mask = torch.zeros_like(masked_image_latents[:, :, :1, :, :])
        #if end_image is not None:
        #    mask[:, -1, :, :, :] = 0
        mask[:, 0, :, :, :] = vae_scaling_factor

        final_latents = masked_image_latents * vae_scaling_factor
        
        log.info(f"Encoded latents shape: {final_latents.shape}")
        vae.to(offload_device)
        
        return ({
            "samples": final_latents,
            "mask": mask
                 },)

#region Tora    
from .tora.traj_utils import process_traj, scale_traj_list_to_256
from torchvision.utils import flow_to_image

class ToraEncodeTrajectory:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "tora_model": ("TORAMODEL",),
            "vae": ("VAE",),
            "coordinates": ("STRING", {"forceInput": True}),
            "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
            "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
            "num_frames": ("INT", {"default": 49, "min": 2, "max": 1024, "step": 1}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TORAFEATURES", "IMAGE", )
    RETURN_NAMES = ("tora_trajectory", "video_flow_images", )
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, vae, width, height, num_frames, coordinates, strength, start_percent, end_percent, tora_model, enable_tiling=False):
        check_diffusers_version()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        try:
            vae.enable_slicing()
        except:
            pass
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
        video_flow = (rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(vae.dtype))
        video_flow_image = rearrange(video_flow, "B C T H W -> (B T) H W C")
        #print(video_flow_image.shape)
        mm.soft_empty_cache()

        # VAE encode
        vae.to(device)
        video_flow = vae.encode(video_flow).latent_dist.sample(generator) * vae.config.scaling_factor
        log.info(f"video_flow shape after encoding: {video_flow.shape}") #torch.Size([1, 16, 4, 80, 80])
        vae.to(offload_device)

        tora_model["traj_extractor"].to(device)
        #print("video_flow shape before traj_extractor: ", video_flow.shape) #torch.Size([1, 16, 4, 80, 80])
        video_flow_features = tora_model["traj_extractor"](video_flow.to(torch.float32))
        tora_model["traj_extractor"].to(offload_device)
        video_flow_features = torch.stack(video_flow_features)
        #print("video_flow_features after traj_extractor: ", video_flow_features.shape) #torch.Size([42, 4, 128, 40, 40])

        video_flow_features = video_flow_features * strength

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
            "vae": ("VAE",),
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

    def encode(self, vae, optical_flow, strength, tora_model, start_percent, end_percent):
        check_diffusers_version()
        B, H, W, C = optical_flow.shape
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        try:
            vae.enable_slicing()
        except:
            pass
       
        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass       

        video_flow = optical_flow * 2 - 1
        video_flow = rearrange(video_flow, "(B T) H W C -> B C T H W", T=B, B=1)
        print(video_flow.shape)
        mm.soft_empty_cache()

        # VAE encode
        
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
                "num_blocks_to_cache": ("INT", {"default": 42, "min": 0, "max": 1024, "step": 1, "tooltip": "Number of transformer blocks to cache, 5b model has 42 blocks, tradeoff between speed and memory"}),
            },
        }

    RETURN_TYPES = ("FASTERCACHEARGS",)
    RETURN_NAMES = ("fastercache", )
    FUNCTION = "args"
    CATEGORY = "CogVideoWrapper"

    def args(self, start_step, hf_step, lf_step, cache_device, num_blocks_to_cache):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if cache_device == "cuda:1":
            device = torch.device("cuda:1")
        fastercache = {
            "start_step" : start_step,
            "hf_step" : hf_step,
            "lf_step" : lf_step,
            "cache_device" : device if cache_device != "offload_device" else offload_device,
            "num_blocks_to_cache" : num_blocks_to_cache,
        }
        return (fastercache,)
    
class CogVideoXTeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Cache threshold, higher values are faster while sacrificing quality"}),
            }
        }
    
    RETURN_TYPES = ("TEACACHEARGS",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "args"
    CATEGORY = "CogVideoWrapper"
    
    def args(self, rel_l1_thresh):
        teacache = {
            "rel_l1_thresh": rel_l1_thresh
        }
        return (teacache,)

#region Sampler    
class CogVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("COGVIDEOMODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (available_schedulers,
                    {
                        "default": 'CogVideoXDDIM'
                    }),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "image_cond_latents": ("LATENT",{"tooltip": "Latent to use for image2video conditioning"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "context_options": ("COGCONTEXT", ),
                "controlnet": ("COGVIDECONTROLNET",),
                "tora_trajectory": ("TORAFEATURES", ),
                "fastercache": ("FASTERCACHEARGS", ),
                "feta_args": ("FETAARGS", ),
                "teacache_args": ("TEACACHEARGS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, model, positive, negative, steps, cfg, seed, scheduler, num_frames, samples=None,
                denoise_strength=1.0, image_cond_latents=None, context_options=None, controlnet=None, tora_trajectory=None, fastercache=None, feta_args=None, teacache_args=None):
        mm.unload_all_models()
        mm.soft_empty_cache()

        model_name = model.get("model_name", "")
        supports_image_conds = True if (
        "I2V" in model_name or 
        "interpolation" in model_name.lower() or 
        "fun" in model_name.lower() or
        "img2vid" in model_name.lower()
        ) else False
        if "fun" in model_name.lower() and not ("pose" in model_name.lower() or "control" in model_name.lower()) and image_cond_latents is not None:
            assert image_cond_latents["mask"] is not None, "For fun inpaint models use CogVideoImageEncodeFunInP"
            fun_mask = image_cond_latents["mask"]
        else:
            fun_mask = None
            
        if image_cond_latents is not None:
            assert supports_image_conds, "Image condition latents only supported for I2V and Interpolation models"
            image_conds = image_cond_latents["samples"]
            image_cond_start_percent = image_cond_latents.get("start_percent", 0.0)
            image_cond_end_percent = image_cond_latents.get("end_percent", 1.0)
            if ("1.5" in model_name or "1_5" in model_name) and not "fun" in model_name.lower():
                image_conds = image_conds / 0.7 # needed for 1.5 models
        else:
            if not "fun" in model_name.lower():
                assert not supports_image_conds, "Image condition latents required for I2V models"
            image_conds = None

        if samples is not None:
            if len(samples["samples"].shape) == 5:
                B, T, C, H, W = samples["samples"].shape
                latents = samples["samples"]
            if len(samples["samples"].shape) == 4:
                B, C, H, W = samples["samples"].shape
                latents = None
        if image_cond_latents is not None:
            B, T, C, H, W = image_cond_latents["samples"].shape
        height = H * 8
        width = W * 8        

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = model["pipe"]
        dtype = model["dtype"]
        scheduler_config = model["scheduler_config"]
        
        if not model["cpu_offloading"] and model["manual_offloading"]:
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
            pipe.transformer.fastercache_num_blocks_to_cache = fastercache["num_blocks_to_cache"]
            log.info(f"FasterCache enabled for {pipe.transformer.fastercache_num_blocks_to_cache} blocks out of {len(pipe.transformer.transformer_blocks)}")
        else:
            pipe.transformer.use_fastercache = False
            pipe.transformer.fastercache_counter = 0

        if teacache_args is not None:
            pipe.transformer.use_teacache = True
            pipe.transformer.teacache_rel_l1_thresh = teacache_args["rel_l1_thresh"]
            log.info(f"TeaCache enabled with rel_l1_thresh: {pipe.transformer.teacache_rel_l1_thresh}")
        else:
            pipe.transformer.use_teacache = False

        if not isinstance(cfg, list):
            cfg = [cfg for _ in range(steps)]
        else:
            assert len(cfg) == steps, "Length of cfg list must match number of steps"
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
  
        autocast_context = torch.autocast(
            mm.get_autocast_device(device), dtype=dtype
        ) if any(q in model["quantization"] for q in ("e4m3fn", "GGUF")) else nullcontext()
        with autocast_context:
            latents = model["pipe"](
                num_inference_steps=steps,
                height = height,
                width = width,
                num_frames = num_frames,
                guidance_scale=cfg,
                latents=latents if samples is not None else None,
                fun_mask = fun_mask,
                image_cond_latents=image_conds,
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
                image_cond_start_percent=image_cond_start_percent if image_cond_latents is not None else 0.0,
                image_cond_end_percent=image_cond_end_percent if image_cond_latents is not None else 1.0,
                feta_args=feta_args,
            )
        if not model["cpu_offloading"] and model["manual_offloading"]:
            pipe.transformer.to(offload_device)

        if fastercache is not None:
            for block in pipe.transformer.transformer_blocks:
                if (hasattr, block, "cached_hidden_states") and block.cached_hidden_states is not None:
                    block.cached_hidden_states = None
                    block.cached_encoder_hidden_states = None

        print_memory(device)

        if teacache_args is not None:
            log.info(f"TeaCache skipped steps: {pipe.transformer.teacache_counter}")
        mm.soft_empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        additional_frames = getattr(pipe, "additional_frames", 0)
        return ({
            "samples": latents,
            "additional_frames": additional_frames,
            },)

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
                    "vae": ("VAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_sample_min_height": ("INT", {"default": 240, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile height, default is half the height"}),
                    "tile_sample_min_width": ("INT", {"default": 360, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile width, default is half the width"}),
                    "tile_overlap_factor_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "tile_overlap_factor_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Auto size based on height and width, default is half the size"}),
                    },            
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "CogVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width, 
               auto_tile_size=True, pipeline=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        
        additional_frames = samples.get("additional_frames", 0)

        try:
            vae.enable_slicing()
        except:
            pass

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
        try:
            frames = vae.decode(latents[:, :, additional_frames:]).sample
        except:
            mm.soft_empty_cache()
            log.warning("Failed to decode, retrying with tiling")
            vae.enable_tiling()
            frames = vae.decode(latents[:, :, additional_frames:]).sample

        vae.disable_tiling()
        vae.to(offload_device)
        mm.soft_empty_cache()

        video_processor = VideoProcessor(vae_scale_factor=8)
        video_processor.config.do_resize = False

        video = video_processor.postprocess_video(video=frames, output_type="pt")
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
        from .cogvideox_fun.utils import ASPECT_RATIO_512, get_closest_ratio

        B, H, W, C = images.shape
        # Find most suitable height and width
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        closest_size, closest_ratio = get_closest_ratio(H, W, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        log.info(f"Closest bucket size: {width}x{height}")

        resized_images = images.clone().movedim(-1,1)
        resized_images = common_upscale(resized_images, width, height, upscale_method, crop)
        resized_images = resized_images.movedim(1,-1)
        
        return (resized_images, width, height)

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
 
        #[[0.0658900170023352, 0.04687556512203313, -0.056971557475649186], [-0.01265770449940036, -0.02814809569100843, -0.0768912512529372], [0.061456544746314665, 0.0005511617552452358, -0.0652574975291287], [-0.09020669168815276, -0.004755440180558637, -0.023763970904494294], [0.031766964513999865, -0.030959599938418375, 0.08654669098083616], [-0.005981764690055846, -0.08809119252349802, -0.06439852368217663], [-0.0212114426433989, 0.08894281999597677, 0.05155629477559985], [-0.013947446911030725, -0.08987475069900677, -0.08923124751217484], [-0.08235967967978511, 0.07268025379974379, 0.08830486164536037], [-0.08052049179735378, -0.050116143175332195, 0.02023752569687405], [-0.07607527759162447, 0.06827156419895981, 0.08678111754261035], [-0.04689089232553825, 0.017294986041038893, -0.10280492336438908], [-0.06105783150270304, 0.07311850680875913, 0.019995735372550075], [-0.09232589996527711, -0.012869815059053047, -0.04355587834255975], [-0.06679931010802251, 0.018399815879067458, 0.06802404982033876], [-0.013062632927118165, -0.04292991477896661, 0.07476243356192845]]
        #latent_rgb_factors =[[0.11945946736445662, 0.09919175788574555, -0.004832707433877734], [-0.0011977028264356232, 0.05496505130267682, 0.021321622433638193], [-0.014088548986590666, -0.008701477861945644, -0.020991313281459367], [0.03063921972519621, 0.12186477097625073, 0.0139593690235148], [0.0927403067854673, 0.030293187650929136, 0.05083134241694003], [0.0379112441305742, 0.04935199882777209, 0.058562766246777774], [0.017749911959153715, 0.008839453404921545, 0.036005638019226294], [0.10610119248526109, 0.02339855688237826, 0.057154257614084596], [0.1273639464837117, -0.010959856130713416, 0.043268631260428896], [-0.01873510946881321, 0.08220930648486932, 0.10613256772247093], [0.008429116376722327, 0.07623856561000408, 0.09295712117576727], [0.12938137079617007, 0.12360403483892413, 0.04478930933220116], [0.04565908794779364, 0.041064156741596365, -0.017695041535528512], [0.00019003240570281826, -0.013965147883381978, 0.05329669529635849], [0.08082391586738358, 0.11548306825496074, -0.021464170006615893], [-0.01517932393230994, -0.0057985555313003236, 0.07216646476618871]]
        latent_rgb_factors = [[0.03197404301362048, 0.04091260743347359, 0.0015679806301828524], [0.005517101026578029, 0.0052348639043457755, -0.005613441650464035], [0.0012485338264583965, -0.016096744206117782, 0.025023940031635054], [0.01760126794276171, 0.0036818415416642893, -0.0006019202528157255], [0.000444954842288864, 0.006102128982092191, 0.0008457999272962447], [-0.010531904354560697, -0.0032275501924977175, -0.00886595780267917], [-0.0001454543946122991, 0.010199210750845965, -0.00012702234832386188], [0.02078497279904325, -0.001669617778939972, 0.006712703698951264], [0.005529571599763264, 0.009733929789086743, 0.001887302765339838], [0.012138415094654218, 0.024684961927224837, 0.037211249767461915], [0.0010364484570000384, 0.01983636315929172, 0.009864602025627755], [0.006802862648143341, -0.0010509255113510681, -0.007026003345126021], [0.0003532208468418043, 0.005351971582801936, -0.01845912126717106], [-0.009045079994694397, -0.01127941143183089, 0.0042294057970470806], [0.002548289972720752, 0.025224244654428216, -0.0006086130121693347], [-0.011135669222532816, 0.0018181308593668505, 0.02794541485349922]]
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
    "CogVideoImageEncode": CogVideoImageEncode,
    "CogVideoTextEncodeCombine": CogVideoTextEncodeCombine,
    "CogVideoTransformerEdit": CogVideoTransformerEdit,
    "CogVideoContextOptions": CogVideoContextOptions,
    "CogVideoControlNet": CogVideoControlNet,
    "ToraEncodeTrajectory": ToraEncodeTrajectory,
    "ToraEncodeOpticalFlow": ToraEncodeOpticalFlow,
    "CogVideoXFasterCache": CogVideoXFasterCache,
    "CogVideoXFunResizeToClosestBucket": CogVideoXFunResizeToClosestBucket,
    "CogVideoLatentPreview": CogVideoLatentPreview,
    "CogVideoXTorchCompileSettings": CogVideoXTorchCompileSettings,
    "CogVideoImageEncodeFunInP": CogVideoImageEncodeFunInP,
    "CogVideoEnhanceAVideo": CogVideoEnhanceAVideo,
    "CogVideoXTeaCache": CogVideoXTeaCache,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode",
    "CogVideoTextEncode": "CogVideo TextEncode",
    "CogVideoImageEncode": "CogVideo ImageEncode",
    "CogVideoTextEncodeCombine": "CogVideo TextEncode Combine",
    "CogVideoTransformerEdit": "CogVideo TransformerEdit",
    "CogVideoContextOptions": "CogVideo Context Options",
    "ToraEncodeTrajectory": "Tora Encode Trajectory",
    "ToraEncodeOpticalFlow": "Tora Encode OpticalFlow",
    "CogVideoXFasterCache": "CogVideoX FasterCache",
    "CogVideoXFunResizeToClosestBucket": "CogVideoXFun ResizeToClosestBucket",
    "CogVideoLatentPreview": "CogVideo LatentPreview",
    "CogVideoXTorchCompileSettings": "CogVideo TorchCompileSettings",
    "CogVideoImageEncodeFunInP": "CogVideo ImageEncode FunInP",
    "CogVideoEnhanceAVideo": "CogVideo Enhance-A-Video",
    "CogVideoXTeaCache": "CogVideoX TeaCache",
    }
