import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from .pipeline_cogvideox import CogVideoXPipeline
from contextlib import nullcontext

from .cogvideox_fun.transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelFun
from .cogvideox_fun.autoencoder_magvit import AutoencoderKLCogVideoX as AutoencoderKLCogVideoXFun
from .cogvideox_fun.utils import get_image_to_video_latent, get_video_to_video_latent, ASPECT_RATIO_512, get_closest_ratio, to_pil
from .cogvideox_fun.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from PIL import Image
import numpy as np
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadCogVideoModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "THUDM/CogVideoX-2b",
                        "THUDM/CogVideoX-5b",
                        "THUDM/CogVideoX-5b-I2V",
                        "bertjiazheng/KoolCogVideoX-5b",
                        "kijai/CogVideoX-Fun-2b",
                        "kijai/CogVideoX-Fun-5b",
                    ],
                ),

            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16", "tooltip": "official recommendation is that 2b model should be fp16, 5b model should be bf16"}
                ),
                "fp8_transformer": (['disabled', 'enabled', 'fastmode'], {"default": 'disabled', "tooltip": "enabled casts the transformer to torch.float8_e4m3fn, fastmode is only for latest nvidia GPUs"}),
                "compile": (["disabled","onediff","torch"], {"tooltip": "compile the model for faster inference, these are advanced options only available on Linux, see readme for more info"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, precision, fp8_transformer="disabled", compile="disabled", enable_sequential_cpu_offload=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        if "Fun" in model:
            repo_id = "kijai/CogVideoX-Fun-pruned"
            download_path = os.path.join(folder_paths.models_dir, "CogVideo")
            if "2b" in model:
                base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-2b-InP") # location of the official model
                scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
                if not os.path.exists(base_path):
                    base_path = os.path.join(download_path, "CogVideoX-Fun-2b-InP")
            elif "5b" in model:
                base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-5b-InP") # location of the official model
                scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
                if not os.path.exists(base_path):
                    base_path = os.path.join(download_path, "CogVideoX-Fun-5b-InP")
            
        elif "2b" in model:
            base_path = os.path.join(folder_paths.models_dir, "CogVideo", "CogVideo2B")
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
            download_path = base_path
            repo_id = model
        elif "5b" in model:
            base_path = os.path.join(folder_paths.models_dir, "CogVideo", (model.split("/")[-1]))
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
            download_path = base_path
            repo_id = model
        
        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=["*text_encoder*", "*tokenizer*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        # transformer
        if "Fun" in model:
            transformer = CogVideoXTransformer3DModelFun.from_pretrained(base_path, subfolder="transformer")
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer")
        
        transformer = transformer.to(dtype).to(offload_device)
        
        if fp8_transformer == "enabled" or fp8_transformer == "fastmode":
            if "2b" in model:
                for name, param in transformer.named_parameters():
                    if name != "pos_embedding":
                        param.data = param.data.to(torch.float8_e4m3fn)
            elif "I2V" in model:
                for name, param in transformer.named_parameters():
                    if "patch_embed" not in name:
                        param.data = param.data.to(torch.float8_e4m3fn)
            else:
                transformer.to(torch.float8_e4m3fn)
        
            if fp8_transformer == "fastmode":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, dtype)

        with open(scheduler_path) as f:
            scheduler_config = json.load(f)
        scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config)  

        # VAE
        if "Fun" in model:
            vae = AutoencoderKLCogVideoXFun.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
            pipe = CogVideoX_Fun_Pipeline_Inpaint(vae, transformer, scheduler)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
            pipe = CogVideoXPipeline(vae, transformer, scheduler)

        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        # compilation
        if compile == "torch":
            torch._dynamo.config.suppress_errors = True
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        elif compile == "onediff":
            from onediffx import compile_pipe
            os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
            
            pipe = compile_pipe(
            pipe,
            backend="nexfort",
            options= {"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": False, "triton.fuse_attention_allow_fp16_reduction": False}},
            ignores=["vae"],
            fuse_qkv_projections=True,
            )

        pipeline = {
            "pipe": pipe,
            "dtype": dtype,
            "base_path": base_path,
            "onediff": True if compile == "onediff" else False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "scheduler_config": scheduler_config
        }

        return (pipeline,)

class DownloadAndLoadCogVideoGGUFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "CogVideoX_5b_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_GGUF_Q4_0.safetensors",
                    ],
                ),
            "vae_precision": (["fp16", "fp32", "bf16"], {"default": "bf16", "tooltip": "VAE dtype"}),
            "fp8_fastmode": ("BOOLEAN", {"default": False, "tooltip": "only supported on 4090 and later GPUs"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, vae_precision, fp8_fastmode, load_device):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        vae_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[vae_precision]
        download_path = os.path.join(folder_paths.models_dir, 'CogVideo', 'GGUF')
        gguf_path = os.path.join(folder_paths.models_dir, 'diffusion_models', model) # check MinusZone's model path first
        if not os.path.exists(gguf_path):
            gguf_path = os.path.join(download_path, model)
            if not os.path.exists(gguf_path):
                log.info(f"Downloading model to: {gguf_path}")
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id="MinusZoneAI/ComfyUI-CogVideoX-MZ",
                    allow_patterns=[f"*{model}*"],
                    local_dir=download_path,
                    local_dir_use_symlinks=False,
                )
        
        
        with open(os.path.join(script_directory, 'configs', 'transformer_config_5b.json')) as f:
                    transformer_config = json.load(f)
        sd = load_torch_file(gguf_path)

        from . import mz_gguf_loader
        import importlib
        importlib.reload(mz_gguf_loader)
        
        with mz_gguf_loader.quantize_lazy_load():
            if "fun" in model:
                transformer_config["in_channels"] = 33
                transformer = CogVideoXTransformer3DModelFun.from_config(transformer_config)
            else:
                transformer_config["in_channels"] = 16
                transformer = CogVideoXTransformer3DModel.from_config(transformer_config)

            transformer.to(torch.float8_e4m3fn)
            transformer = mz_gguf_loader.quantize_load_state_dict(transformer, sd, device="cpu")
            if load_device == "offload_device":
                transformer.to(offload_device)
            else:
                transformer.to(device)
        
        if fp8_fastmode:
           from .fp8_optimization import convert_fp8_linear
           convert_fp8_linear(transformer, vae_dtype)

        scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
        with open(scheduler_path) as f:
            scheduler_config = json.load(f)
        
        scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config, subfolder="scheduler")

        # VAE
        vae_dl_path = os.path.join(folder_paths.models_dir, 'CogVideo', 'VAE')
        vae_path = os.path.join(vae_dl_path, "cogvideox_vae.safetensors")
        if not os.path.exists(vae_path):
            log.info(f"Downloading VAE model to: {vae_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/CogVideoX-Fun-pruned",
                allow_patterns=["*cogvideox_vae.safetensors*"],
                local_dir=vae_dl_path,
                local_dir_use_symlinks=False,
            )
        with open(os.path.join(script_directory, 'configs', 'vae_config.json')) as f:
            vae_config = json.load(f)
        
        vae_sd = load_torch_file(vae_path)
        if "fun" in model:
            vae = AutoencoderKLCogVideoXFun.from_config(vae_config).to(vae_dtype).to(offload_device)
            vae.load_state_dict(vae_sd)
            pipe = CogVideoX_Fun_Pipeline_Inpaint(vae, transformer, scheduler)
        else:
            vae = AutoencoderKLCogVideoX.from_config(vae_config).to(vae_dtype).to(offload_device)
            vae.load_state_dict(vae_sd)
            pipe = CogVideoXPipeline(vae, transformer, scheduler)

        # compilation
        if compile == "torch":
            torch._dynamo.config.suppress_errors = True
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        elif compile == "onediff":
            from onediffx import compile_pipe
            os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
            
            pipe = compile_pipe(
            pipe,
            backend="nexfort",
            options= {"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": False, "triton.fuse_attention_allow_fp16_reduction": False}},
            ignores=["vae"],
            fuse_qkv_projections=True,
            )

        pipeline = {
            "pipe": pipe,
            "dtype": vae_dtype,
            "base_path": "Fun" if "fun" in model else "sad",
            "onediff": True if compile == "onediff" else False,
            "cpu_offloading": False,
            "scheduler_config": scheduler_config
        }

        return (pipeline,)
    
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

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, prompt, strength=1.0, force_offload=True):
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = 226
        clip.cond_stage_model.to(load_device)
        tokens = clip.tokenize(prompt, return_word_ids=True)

        embeds = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)
        embeds *= strength
        if force_offload:
            clip.cond_stage_model.to(offload_device)

        return (embeds, )
    
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
    
class CogVideoImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "image": ("IMAGE", ),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 16, "min": 1}),
                "enable_vae_slicing": ("BOOLEAN", {"default": True, "tooltip": "VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes."}),
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, image, chunk_size=8, enable_vae_slicing=True, mask=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)

        B, H, W, C = image.shape

        vae = pipeline["pipe"].vae
        
        if enable_vae_slicing:
            vae.enable_slicing()
        else:
            vae.disable_slicing()

        if not pipeline["cpu_offloading"]:
            vae.to(device)
        
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
        print("final latents: ", final_latents.shape)
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        
        return ({"samples": final_latents}, )

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
                "scheduler": (["DDIM", "DPM", "DDIM_tiled"], {"tooltip": "5B likes DPM, but it doesn't support temporal tiling"}),
                "t_tile_length": ("INT", {"default": 16, "min": 2, "max": 128, "step": 1, "tooltip": "Length of temporal tiling, use same alue as num_frames to disable, disabled automatically for DPM"}),
                "t_tile_overlap": ("INT", {"default": 8, "min": 2, "max": 128, "step": 1, "tooltip": "Overlap of temporal tiling"}),
            },
            "optional": {
                "samples": ("LATENT", ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_cond_latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, steps, cfg, seed, height, width, num_frames, scheduler, t_tile_length, t_tile_overlap, samples=None, 
                denoise_strength=1.0, image_cond_latents=None):
        mm.soft_empty_cache()

        base_path = pipeline["base_path"]

        assert "Fun" not in base_path, "'Fun' models not supported in 'CogVideoSampler', use the 'CogVideoXFunSampler'"
        assert t_tile_length > t_tile_overlap, "t_tile_length must be greater than t_tile_overlap"
        assert t_tile_length <= num_frames, "t_tile_length must be equal or less than num_frames"
        t_tile_length = t_tile_length // 4
        t_tile_overlap = t_tile_overlap // 4

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        scheduler_config = pipeline["scheduler_config"]
        
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        if scheduler == "DDIM" or scheduler == "DDIM_tiled":
            pipe.scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config)
        elif scheduler == "DPM":
            pipe.scheduler = CogVideoXDPMScheduler.from_config(scheduler_config)

        if negative.shape[1] < positive.shape[1]:
            target_length = positive.shape[1]
            padding = torch.zeros((negative.shape[0], target_length - negative.shape[1], negative.shape[2]), device=negative.device)
            negative = torch.cat((negative, padding), dim=1)

        autocastcondition = not pipeline["onediff"]
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            latents = pipeline["pipe"](
                num_inference_steps=steps,
                height = height,
                width = width,
                num_frames = num_frames,
                t_tile_length = t_tile_length,
                t_tile_overlap = t_tile_overlap,
                guidance_scale=cfg,
                latents=samples["samples"] if samples is not None else None,
                image_cond_latents=image_cond_latents["samples"] if image_cond_latents is not None else None,
                denoise_strength=denoise_strength,
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                generator=generator,
                device=device,
                scheduler_name=scheduler
            )
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(offload_device)
        mm.soft_empty_cache()
        print(latents.shape)

        return (pipeline, {"samples": latents})
    
class CogVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "samples": ("LATENT", ),
            "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
            },
            "optional": {
            "tile_sample_min_height": ("INT", {"default": 240, "min": 16, "max": 2048, "step": 8}),
            "tile_sample_min_width": ("INT", {"default": 360, "min": 16, "max": 2048, "step": 8}),
            "tile_overlap_factor_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
            "tile_overlap_factor_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
            "enable_vae_slicing": ("BOOLEAN", {"default": True, "tooltip": "VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "CogVideoWrapper"

    def decode(self, pipeline, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width, enable_vae_slicing=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        vae = pipeline["pipe"].vae
        if enable_vae_slicing:
            vae.enable_slicing()
        else:
            vae.disable_slicing()
        if not pipeline["cpu_offloading"]:
            vae.to(device)
        if enable_vae_tiling:
            vae.enable_tiling(
                tile_sample_min_height=tile_sample_min_height,
                tile_sample_min_width=tile_sample_min_width,
                tile_overlap_factor_height=tile_overlap_factor_height,
                tile_overlap_factor_width=tile_overlap_factor_width,
            )
        else:
            vae.disable_tiling()
        latents = latents.to(vae.dtype)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / vae.config.scaling_factor * latents
        vae._clear_fake_context_parallel_cache()
        frames = vae.decode(latents).sample
        vae.disable_tiling()
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        mm.soft_empty_cache()

        video = pipeline["pipe"].video_processor.postprocess_video(video=frames, output_type="pt")
        video = video[0].permute(0, 2, 3, 1).cpu().float()
        print(video.min(), video.max())

        return (video,)

class CogVideoXFunSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "video_length": ("INT", {"default": 49, "min": 5, "max": 49, "step": 4}),
                "base_resolution": (
                    [ 
                        256,
                        320,
                        384,
                        448,
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                        "CogVideoXDDIM",
                        "CogVideoXDPMScheduler",
                    ],
                    {
                        "default": 'DDIM'
                    }
                )
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
                "opt_empty_latent": ("LATENT",),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline,  positive, negative, video_length, base_resolution, seed, steps, cfg, scheduler, 
                start_img=None, end_img=None, opt_empty_latent=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]
        assert "Fun" in base_path, "'Unfun' models not supported in 'CogVideoXFunSampler', use the 'CogVideoSampler'"

        pipe.enable_model_cpu_offload(device=device)

        mm.soft_empty_cache()

        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        if start_img is not None:
            start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
            end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
            # Count most suitable height and width
            original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        else:
            original_width = opt_empty_latent["samples"][0].shape[-1] * 8
            original_height = opt_empty_latent["samples"][0].shape[-2] * 8
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        print(f"Closest size: {width}x{height}")
        
        # Load Sampler
        scheduler_config = pipeline["scheduler_config"]
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler.from_config(scheduler_config)
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler.from_config(scheduler_config)
        elif scheduler == "CogVideoXDDIM":
            noise_scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config)
        elif scheduler == "CogVideoXDPMScheduler":
            noise_scheduler = CogVideoXDPMScheduler.from_config(scheduler_config)
        pipe.scheduler = noise_scheduler

        #if not pipeline["cpu_offloading"]:
        #    pipe.transformer.to(device)
        generator= torch.Generator(device=device).manual_seed(seed)

        autocastcondition = not pipeline["onediff"]
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            video_length = int((video_length - 1) // pipe.vae.config.temporal_compression_ratio * pipe.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            latents = pipe(
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                num_frames = video_length,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                comfyui_progressbar = True,
            )
        #if not pipeline["cpu_offloading"]:
        #     pipe.transformer.to(offload_device)
        mm.soft_empty_cache()
        print(latents.shape)

        return (pipeline, {"samples": latents})

class CogVideoXFunVid2VidSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "video_length": ("INT", {"default": 49, "min": 5, "max": 49, "step": 4}),
                "base_resolution": (
                    [ 
                        256,
                        320,
                        384,
                        448,
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                        "CogVideoXDDIM",
                        "CogVideoXDPMScheduler",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
                "denoise_strength": ("FLOAT", {"default": 0.70, "min": 0.05, "max": 1.00, "step": 0.01}),
                "validation_video": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, validation_video):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]

        pipe.enable_model_cpu_offload(device=device)

        mm.soft_empty_cache()

        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        validation_video = np.array(validation_video.cpu().numpy() * 255, np.uint8)
        original_width, original_height = Image.fromarray(validation_video[0]).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        
        base_path = pipeline["base_path"]

        # Load Sampler
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "CogVideoXDDIM":
            noise_scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        elif scheduler == "CogVideoXDPMScheduler":
            noise_scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder= 'scheduler')
        pipe.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)

        autocastcondition = not pipeline["onediff"]
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            video_length = int((video_length - 1) // pipe.vae.config.temporal_compression_ratio * pipe.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width))

            # for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
            #     pipeline = merge_lora(pipeline, _lora_path, _lora_weight)

            latents = pipe(
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                num_frames = video_length,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                strength = float(denoise_strength),
                comfyui_progressbar = True,
            )

            # for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
            #     pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)
        return (pipeline, {"samples": latents})

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": DownloadAndLoadCogVideoModel,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoDecode": CogVideoDecode,
    "CogVideoTextEncode": CogVideoTextEncode,
    "CogVideoDualTextEncode_311": CogVideoDualTextEncode_311,
    "CogVideoImageEncode": CogVideoImageEncode,
    "CogVideoXFunSampler": CogVideoXFunSampler,
    "CogVideoXFunVid2VidSampler": CogVideoXFunVid2VidSampler,
    "CogVideoTextEncodeCombine": CogVideoTextEncodeCombine,
    "DownloadAndLoadCogVideoGGUFModel": DownloadAndLoadCogVideoGGUFModel
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": "(Down)load CogVideo Model",
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode",
    "CogVideoTextEncode": "CogVideo TextEncode",
    "CogVideoDualTextEncode_311": "CogVideo DualTextEncode",
    "CogVideoImageEncode": "CogVideo ImageEncode",
    "CogVideoXFunSampler": "CogVideoXFun Sampler",
    "CogVideoXFunVid2VidSampler": "CogVideoXFun Vid2Vid Sampler",
    "CogVideoTextEncodeCombine": "CogVideo TextEncode Combine",
    "DownloadAndLoadCogVideoGGUFModel": "(Down)load CogVideo GGUF Model"
    }
