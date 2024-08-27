import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from .pipeline_cogvideox import CogVideoXPipeline

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
                "fp8_transformer": ("BOOLEAN", {"default": False, "tooltip": "cast the transformer to torch.float8_e4m3fn"}),
                "torch_compile": ("BOOLEAN", {"default": False, "tooltip": "use torch.compile to speed up inference, Linux only"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, precision, fp8_transformer, torch_compile):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        if fp8_transformer:
            transformer_dtype = torch.float8_e4m3fn
        else:
            transformer_dtype = dtype

        if "2b" in model:
            base_path = os.path.join(folder_paths.models_dir, "CogVideo", "CogVideo2B")
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
        transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer").to(transformer_dtype).to(offload_device)
        vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
        scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")

        pipe = CogVideoXPipeline(vae, transformer, scheduler)

        if torch_compile:
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

        pipeline = {
            "pipe": pipe,
            "dtype": dtype,
            "base_path": base_path
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
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, pipeline, image):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        vae = pipeline["pipe"].vae
        vae.to(device)
  
        input_image = image.clone() * 2.0 - 1.0
        input_image = input_image.to(vae.dtype).to(device)
        input_image = input_image.unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        B, C, T, H, W = input_image.shape
        chunk_size = 16
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
                "scheduler": (["DDIM", "DPM"], {"tooltip": "5B likes DPM, but it doesn't support temporal tiling"}),
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
        
        pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        if scheduler == "DDIM":
            pipe.scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        elif scheduler == "DPM":
            pipe.scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        with torch.autocast(mm.get_autocast_device(device)):
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
                device=device
            )
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
            "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "CogVideoWrapper"

    def decode(self, pipeline, samples, enable_vae_tiling):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        vae = pipeline["pipe"].vae
        vae.to(device)
        if enable_vae_tiling:
            vae.enable_tiling(
                tile_sample_min_height=96,
                tile_sample_min_width=96,
                tile_overlap_factor_height=1 / 12,
                tile_overlap_factor_width=1 / 12,
            )
        latents = latents.to(vae.dtype)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / vae.config.scaling_factor * latents

        frames = vae.decode(latents).sample
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