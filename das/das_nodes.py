import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
from ..utils import log
import os
import numpy as np
import folder_paths
from tqdm import tqdm
from PIL import Image, ImageDraw

from .motion import CameraMotionGenerator, ObjectMotionGenerator

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

        # cam_motion = CameraMotionGenerator(
        #     motion_type="trans",
        #     frame_num=49,
        #     W=720,
        #     H=480,
        #     fx=None,
        #     fy=None,
        #     fov=55,
        #     device=device,
        #     )
        # poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
        # pred_tracks = cam_motion.apply_motion_on_pts(pred_tracks, poses)
        # print("Camera motion applied")

        spatracker.to(offload_device)

        from .spatracker.utils.visualizer import Visualizer
        vis = Visualizer(
            grayscale=False, 
            fps=24, 
            pad_value=0,
            #tracks_leave_trace=-1
            )

        msk_query = (T_Firsts == 0)
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(
                video=video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                save_video=False,
                )
        
        tracking_video = tracking_video.squeeze(0).permute(0, 2, 3, 1) # [T, H, W, C]
        tracking_video = (tracking_video / 255.0).float()

        return (tracking_video,)

class DAS_MoGeTracker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MOGEMODEL",),
            "image": ("IMAGE", ),
            "num_frames": ("INT", {"default": 49, "min": 1, "max": 100, "step": 1}),
            "width": ("INT", {"default": 720, "min": 1, "max": 10000, "step": 1}),
            "height": ("INT", {"default": 480, "min": 1, "max": 10000, "step": 1}),
            "fov": ("FLOAT", {"default": 55.0, "min": 1.0, "max": 180.0, "step": 1.0}),
            "object_motion_type": (["none", "up", "down", "left", "right", "front", "back"],),
            "object_motion_distance": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
            "camera_motion_type": (["none","translation", "rotation", "spiral"],),
            },
            "optional": {
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tracking_video",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoWrapper"

    def encode(self, model, image, num_frames, width, height, fov, object_motion_type, object_motion_distance, camera_motion_type, mask=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        B, H, W, C = image.shape
        
        image_resized = common_upscale(image.movedim(-1,1), width, height, "lanczos", "disabled").movedim(1,-1)

        # Use the first frame from previously loaded video_tensor
        infer_result = model.infer(image_resized.permute(0, 3, 1, 2).to(device)[0].to(device))  # [C, H, W] in range [0,1]
        H, W = infer_result["points"].shape[0:2]

        motion_generator = ObjectMotionGenerator(num_frames, device=device)

        if mask is not None:
            mask = mask[0].bool()
            mask = torch.nn.functional.interpolate(
                mask[None, None].float(), 
                size=(H, W), 
                mode='nearest'
            )[0, 0].bool()
        else:
            mask = torch.ones(H, W, dtype=torch.bool)
        
        # Generate motion dictionary
        motion_dict = motion_generator.generate_motion(
            mask=mask,
            motion_type=object_motion_type,
            distance=object_motion_distance,
            num_frames=num_frames,
        )
        
        pred_tracks = motion_generator.apply_motion(
            infer_result["points"],
            motion_dict,
            tracking_method="moge"
        )
        print("pred_tracks shape: ", pred_tracks.shape)
        print("Object motion applied")

        camera_motion_type_mapping = {
            "none": "none",
            "translation": "trans",
            "rotation": "rot",
            "spiral": "spiral"
        }
        cam_motion = CameraMotionGenerator(
            motion_type=camera_motion_type_mapping[camera_motion_type],
            frame_num=num_frames,
            W=width,
            H=height,
            fx=None,
            fy=None,
            fov=fov,
            device=device,
            )
        # Apply camera motion if specified
        cam_motion.set_intr(infer_result["intrinsics"])
        poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
    
        pred_tracks_flatten = pred_tracks.reshape(num_frames, H*W, 3)
        pred_tracks = cam_motion.w2s(pred_tracks_flatten, poses).reshape([num_frames, H, W, 3]) # [T, H, W, 3]
        print("Camera motion applied")    
        
        
        points = pred_tracks.cpu().numpy()
        mask = infer_result["mask"].cpu().numpy()
        # Create color array
        T, H, W, _ = pred_tracks.shape
        
        print("points shape: ", points.shape)
        
        print("mask shape: ", mask.shape)
        colors = np.zeros((H, W, 3), dtype=np.uint8)

        # Set R channel - based on x coordinates (smaller on the left)
        colors[:, :, 0] = np.tile(np.linspace(0, 255, W), (H, 1))

        # Set G channel - based on y coordinates (smaller on the top)
        colors[:, :, 1] = np.tile(np.linspace(0, 255, H), (W, 1)).T

        # Set B channel - based on depth
        z_values = points[0, :, :, 2]  # get z values
        inv_z = 1 / z_values  # calculate 1/z
        # Calculate 2% and 98% percentiles
        p2 = np.percentile(inv_z, 2)
        p98 = np.percentile(inv_z, 98)
        # Normalize to [0,1] range
        normalized_z = np.clip((inv_z - p2) / (p98 - p2), 0, 1)
        colors[:, :, 2] = (normalized_z * 255).astype(np.uint8)
        colors = colors.astype(np.uint8)

        # First reshape points and colors
        points = points.reshape(T, -1, 3)  # (T, H*W, 3)
        colors = colors.reshape(-1, 3)      # (H*W, 3)
        
        # Create mask for each frame
        mask = mask.reshape(-1)  # Flatten mask to (H*W,)
        
        # Apply mask
        points = points[:, mask, :]         # (T, masked_points, 3)
        colors = colors[mask]               # (masked_points, 3)
        
        # Repeat colors for each frame
        colors = colors.reshape(1, -1, 3).repeat(T, axis=0)  # (T, masked_points, 3)
        
        # Initialize list to store frames
        frames = []
        pbar = ProgressBar(len(points))
        
        for i, pts_i in enumerate(tqdm(points)):
            pixels, depths = pts_i[..., :2], pts_i[..., 2]
            pixels[..., 0] = pixels[..., 0] * W
            pixels[..., 1] = pixels[..., 1] * H
            pixels = pixels.astype(int)
            
            valid = self.valid_mask(pixels, W, H)

            frame_rgb = colors[i][valid]
            pixels = pixels[valid]
            depths = depths[valid]
            
            img = Image.fromarray(np.uint8(np.zeros([H, W, 3])), mode="RGB")
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            step = 1
            sorted_pixels = sorted_pixels[::step]
            sorted_rgb = frame_rgb[sort_index][::step]
            
            for j in range(sorted_pixels.shape[0]):
                self.draw_rectangle(
                    img,
                    coord=(sorted_pixels[j, 0], sorted_pixels[j, 1]),
                    side_length=2,
                    color=sorted_rgb[j],
                )
            frames.append(np.array(img))
            pbar.update(1)

        # Convert frames to video tensor in range [0,1]
        tracking_video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        tracking_video = tracking_video.permute(0, 2, 3, 1)  # [B, H, W, C]
        print("tracking_video shape: ", tracking_video.shape)
        return (tracking_video,)

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds
        
        Args:
            pixels (numpy.ndarray): Pixel coordinates of shape [N, 2]
            W (int): Image width
            H (int): Image height
            
        Returns:
            numpy.ndarray: Boolean mask of valid pixels
        """
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values
        
        Args:
            points (numpy.ndarray): Points array of shape [N, 2]
            depths (numpy.ndarray): Depth values of shape [N]
            
        Returns:
            tuple: (sorted_points, sorted_depths, sort_index)
        """
        # Combine points and depths into a single array for sorting
        combined = np.hstack((points, depths[:, None]))  # Nx3 (points + depth)
        # Sort by depth (last column) in descending order
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
        # Split back into points and depths
        sorted_points = sorted_combined[:, :-1]
        sorted_depths = sorted_combined[:, -1]
        return sorted_points, sorted_depths, sort_index

    def draw_rectangle(self, rgb, coord, side_length, color=(255, 0, 0)):
        """Draw a rectangle on the image
        
        Args:
            rgb (PIL.Image): Image to draw on
            coord (tuple): Center coordinates (x, y)
            side_length (int): Length of rectangle sides
            color (tuple): RGB color tuple
        """
        draw = ImageDraw.Draw(rgb)
        # Calculate the bounding box of the rectangle
        left_up_point = (coord[0] - side_length//2, coord[1] - side_length//2)  
        right_down_point = (coord[0] + side_length//2, coord[1] + side_length//2)
        color = tuple(list(color))

        draw.rectangle(
            [left_up_point, right_down_point],
            fill=tuple(color),
            outline=tuple(color),
        )
    

NODE_CLASS_MAPPINGS = {
    "CogVideoDASTrackingEncode": CogVideoDASTrackingEncode,
    "DAS_SpaTracker": DAS_SpaTracker,
    "DAS_SpaTrackerModelLoader": DAS_SpaTrackerModelLoader,
    "DAS_MoGeTracker": DAS_MoGeTracker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoDASTrackingEncode": "CogVideo DAS Tracking Encode",
    "DAS_SpaTracker": "DAS SpaTracker",
    "DAS_SpaTrackerModelLoader": "DAS SpaTracker Model Loader",
    "DAS_MoGeTracker": "DAS MoGe Tracker",
    }
