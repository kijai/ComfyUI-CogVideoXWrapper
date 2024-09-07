import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from .pipeline_cogvideox import CogVideoXPipeline
from contextlib import nullcontext


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class DownloadAndLoadCogVideoModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "THUDM/CogVideoX-2b",
                        "THUDM/CogVideoX-5b",
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
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "offload the model to CPU, this is useful for large models and small batch sizes"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, precision, fp8_transformer="disabled", compile="disabled", enable_sequential_cpu_offload=False, enable_model_cpu_offload=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        if "2b" in model:
            base_path = os.path.join(folder_paths.models_dir, "CogVideo", "CogVideoX-2b")
        elif "5b" in model:
            base_path = os.path.join(folder_paths.models_dir, "CogVideo", "CogVideoX-5b")

        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                ignore_patterns=["*text_encoder*", "*tokenizer*"],
                local_dir=base_path,
                local_dir_use_symlinks=False,
            )
        if fp8_transformer == "enabled" or fp8_transformer == "fastmode":
            transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer").to(offload_device)
            if "2b" in model:
                for name, param in transformer.named_parameters():
                    if name != "pos_embedding":
                        param.data = param.data.to(torch.float8_e4m3fn)
            else:
                transformer.to(torch.float8_e4m3fn)

            if fp8_transformer == "fastmode":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, dtype)
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer").to(dtype).to(offload_device)

        vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
        scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")

        pipe = CogVideoXPipeline(vae, transformer, scheduler)
        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        if compile == "torch":
            torch._dynamo.config.suppress_errors = True
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        elif compile == "onediff":
            from onediffx import compile_pipe, quantize_pipe
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
            "cpu_offloading": enable_sequential_cpu_offload or enable_model_cpu_offload,
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
    
class CogVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, prompt):
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = 226
        clip.cond_stage_model.to(load_device)
        tokens = clip.tokenize(prompt, return_word_ids=True)

        embeds = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)
        clip.cond_stage_model.to(offload_device)

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
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, steps, cfg, seed, height, width, num_frames, scheduler, t_tile_length, t_tile_overlap, samples=None, denoise_strength=1.0):
        mm.soft_empty_cache()

        assert t_tile_length > t_tile_overlap, "t_tile_length must be greater than t_tile_overlap"
        assert t_tile_length <= num_frames, "t_tile_length must be equal or less than num_frames"
        t_tile_length = t_tile_length // 4
        t_tile_overlap = t_tile_overlap // 4

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]
        
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        if scheduler == "DDIM" or scheduler == "DDIM_tiled":
            pipe.scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        elif scheduler == "DPM":
            pipe.scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

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
            "tile_sample_min_height": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
            "tile_sample_min_width": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
            "tile_overlap_factor_height": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
            "tile_overlap_factor_width": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
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

        frames = vae.decode(latents).sample
        if not pipeline["cpu_offloading"]:
            vae.to(offload_device)
        mm.soft_empty_cache()

        video = pipeline["pipe"].video_processor.postprocess_video(video=frames, output_type="pt")
        video = video[0].permute(0, 2, 3, 1).cpu().float()
        print(video.min(), video.max())

        return (video,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": DownloadAndLoadCogVideoModel,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoDecode": CogVideoDecode,
    "CogVideoTextEncode": CogVideoTextEncode,
    "CogVideoImageEncode": CogVideoImageEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": "(Down)load CogVideo Model",
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode",
    "CogVideoTextEncode": "CogVideo TextEncode",
    "CogVideoImageEncode": "CogVideo ImageEncode"
    }