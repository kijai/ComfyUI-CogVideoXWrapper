import torch
import comfy.model_management as mm
from ..utils import log
import os
import numpy as np
import folder_paths

class CogVideoDASTrackingEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "vae": ("VAE",),
                    "images": ("IMAGE", ),
                },
                "optional": {
                    "enable_tiling": ("BOOLEAN", {"default": True, "tooltip": "Enable tiling for the VAE to reduce memory usage"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
        }

    RETURN_TYPES = ("DASTRACKING",)
    RETURN_NAMES = ("das_tracking",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, vae, images, enable_tiling=False, strength=1.0, start_percent=0.0, end_percent=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        
        try:
            vae.enable_slicing()
        except:
            pass
       
        vae_scaling_factor = vae.config.scaling_factor
        
        if enable_tiling:
            from ..mz_enable_vae_encode_tiling import enable_vae_encode_tiling
            enable_vae_encode_tiling(vae)

        vae.to(device)

        try:
            vae._clear_fake_context_parallel_cache()
        except:
            pass

        tracking_maps = images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        tracking_first_frame = tracking_maps[:, :, 0:1, :, :]
        tracking_first_frame *= 2.0 - 1.0
        print("tracking_first_frame shape: ", tracking_first_frame.shape)

        tracking_first_frame_latent = vae.encode(tracking_first_frame).latent_dist.sample(generator).permute(0, 2, 1, 3, 4)
        tracking_first_frame_latent = tracking_first_frame_latent * vae_scaling_factor * strength
        log.info(f"Encoded tracking first frame latents shape: {tracking_first_frame_latent.shape}")

        tracking_latents = vae.encode(tracking_maps).latent_dist.sample(generator).permute(0, 2, 1, 3, 4)  # B, T, C, H, W

        tracking_latents = tracking_latents * vae_scaling_factor * strength
        
        log.info(f"Encoded tracking latents shape: {tracking_latents.shape}")
        vae.to(offload_device)
        
        return ({
            "tracking_maps": tracking_latents,
            "tracking_image_latents": tracking_first_frame_latent,
            "start_percent": start_percent,
            "end_percent": end_percent
            }, )

class DAS_SpaTrackerModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (folder_paths.get_filename_list("CogVideo"), {"tooltip": "These models are loaded from the 'ComfyUI/models/CogVideo' -folder",}),
            },
        }

    RETURN_TYPES = ("SPATRACKERMODEL",)
    RETURN_NAMES = ("spatracker_model",)
    FUNCTION = "load"
    CATEGORY = "CogVideoWrapper"

    def load(self, model):
        device = mm.get_torch_device()

        model_path = folder_paths.get_full_path("CogVideo", model)
        from .spatracker.predictor import SpaTrackerPredictor
        
        spatracker = SpaTrackerPredictor(
            checkpoint=model_path,
            interp_shape=(384, 576),
            seq_length=12
        ).to(device)

        return (spatracker,)
        
class DAS_SpaTracker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "spatracker": ("SPATRACKERMODEL",),
            "images": ("IMAGE", ),
            "depth_images": ("IMAGE", ),
            "density": ("INT", {"default": 70, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tracking_video",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, spatracker, images, depth_images, density):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        segm_mask = np.ones((480, 720), dtype=np.uint8)

        video = images.permute(0, 3, 1, 2).to(device).unsqueeze(0)
        video_depth = depth_images.permute(0, 3, 1, 2).to(device)
        video_depth = video_depth[:, 0:1, :, :]

        spatracker.to(device)
            
        pred_tracks, pred_visibility, T_Firsts = spatracker(
            video * 255, 
            video_depth=video_depth,
            grid_size=density,
            backward_tracking=False,
            depth_predictor=None,
            grid_query_frame=0,
            segm_mask=torch.from_numpy(segm_mask)[None, None].to(device),
            wind_length=12,
            progressive_tracking=False
        )

        spatracker.to(offload_device)

        from .spatracker.utils.visualizer import Visualizer
        vis = Visualizer(grayscale=False, fps=24, pad_value=0)

        msk_query = (T_Firsts == 0)
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(video=video, tracks=pred_tracks,
                        visibility=pred_visibility, save_video=False,
                        filename="temp")
        
        tracking_video = tracking_video.squeeze(0).permute(0, 2, 3, 1) # [T, H, W, C]
        tracking_video = (tracking_video / 255.0).float()

        return (tracking_video,)
    

NODE_CLASS_MAPPINGS = {
    "CogVideoDASTrackingEncode": CogVideoDASTrackingEncode,
    "DAS_SpaTracker": DAS_SpaTracker,
    "DAS_SpaTrackerModelLoader": DAS_SpaTrackerModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoDASTrackingEncode": "CogVideo DAS Tracking Encode",
    "DAS_SpaTracker": "DAS SpaTracker",
    "DAS_SpaTrackerModelLoader": "DAS SpaTracker Model Loader",
    }
