import os
import torch
import folder_paths
import comfy.model_management as mm

from .pipeline_cogvideox import CogVideoXPipeline

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class DownloadAndLoadCogVideoModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            "optional": {
                "precision": (
                    [
                        "fp16",
                        "fp32",
                        "bf16",
                    ],
                    {"default": "fp16"},
                ),
            },
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        base_path = os.path.join(folder_paths.models_dir, "CogVideo", "CogVideo2B")

        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="THUDM/CogVideoX-2b",
                ignore_patterns=["*text_encoder*"],
                local_dir=base_path,
                local_dir_use_symlinks=False,
            )

        pipe = CogVideoXPipeline.from_pretrained(base_path, torch_dtype=dtype).to(offload_device)
        

        pipeline = {
            "pipe": pipe,
            "dtype": dtype
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
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = 226
        tokens = clip.tokenize(prompt, return_word_ids=True)

        embeds = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)

        return (embeds, )

class CogVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
            "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
            "num_frames": ("INT", {"default": 48, "min": 1, "max": 100, "step": 1}),
            "fps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, fps, steps, cfg, seed, height, width, num_frames):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]

        pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        latents = pipeline["pipe"](
            num_inference_steps=steps,
            height = height,
            width = width,
            num_frames = num_frames,
            fps = fps,
            guidance_scale=cfg,
            prompt_embeds=positive.to(dtype).to(device),
            negative_prompt_embeds=negative.to(dtype).to(device),
            #negative_prompt_embeds=torch.zeros_like(embeds),
            generator=generator,
            output_type="latents",
            device=device
        )
        pipe.transformer.to(offload_device)
        mm.soft_empty_cache()
        print(latents.shape)
        pipeline["fps"] = fps
        pipeline["num_frames"] = num_frames

        return (pipeline, {"samples": latents})
    
class CogVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("COGVIDEOPIPE",),
            "samples": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, samples):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        vae = pipeline["pipe"].vae
        vae.to(device)

        num_frames = pipeline["num_frames"]
        fps = pipeline["fps"]

        num_seconds = num_frames // fps
        
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / vae.config.scaling_factor * latents

        frames = []
        for i in range(num_seconds):
            # Whether or not to clear fake context parallel cache
            fake_cp = i + 1 < num_seconds
            start_frame, end_frame = (0, 3) if i == 0 else (2 * i + 1, 2 * i + 3)

            current_frames = vae.decode(latents[:, :, start_frame:end_frame], fake_cp=fake_cp).sample
            frames.append(current_frames)
        vae.to(offload_device)

        frames = torch.cat(frames, dim=2)
        video = pipeline["pipe"].video_processor.postprocess_video(video=frames, output_type="pt")
        print(video.shape)
        video = video[0].permute(0, 2, 3, 1).cpu().float()
        print(video.min(), video.max())

        return (video,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": DownloadAndLoadCogVideoModel,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoDecode": CogVideoDecode,
    "CogVideoTextEncode": CogVideoTextEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": "(Down)load CogVideo Model",
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode",
    "CogVideoTextEncode": "CogVideo TextEncode"
    }