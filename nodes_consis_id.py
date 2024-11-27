import os
import json
import torch

import folder_paths
import comfy.model_management as mm

class DownloadAndLoadConsisIDModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["BestWishYsh/ConsisID-preview",],),
                 "onnx_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML'], {
                        "default": 'CPU'
                    }), 
                "precision": (["fp16", "fp32", "bf16"],
                        {"default": "bf16", "tooltip": "official recommendation is that 2b model should be fp16, 5b model should be bf16"}),
            },
        }

    RETURN_TYPES = ("CONSISIDMODEL", )
    RETURN_NAMES = ("consis_id_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Downloads and loads the selected CogVideo model from Huggingface to 'ComfyUI/models/CogVideo'"

    def loadmodel(self, model, precision, onnx_device):
        import insightface
        from insightface.app import FaceAnalysis
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from .consis_id.models.eva_clip import create_model_and_transforms
        from .consis_id.models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        base_path = folder_paths.get_folder_paths("CogVideo")[0]
        model_path = os.path.join(base_path, "ConsisID-preview")
        face_encoder_path = os.path.join(model_path, "face_encoder")
        
        # 1. load face helper models
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
            model_rootpath=model_path
        )
        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=model_path)
        face_helper.face_det.eval()
        face_helper.face_parse.eval()

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(face_encoder_path, "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
        face_clip_model = model.visual
        face_clip_model.eval()

        eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        eva_transform_mean = eva_transform_mean
        eva_transform_std = eva_transform_std

        face_main_model = FaceAnalysis(name='antelopev2', root=face_encoder_path, providers=[onnx_device + 'ExecutionProvider',])
        handler_ante = insightface.model_zoo.get_model(f'{face_encoder_path}/models/antelopev2/glintr100.onnx', providers=[onnx_device + 'ExecutionProvider',])
        face_main_model.prepare(ctx_id=0, det_size=(640, 640))
        handler_ante.prepare(ctx_id=0)
            
        face_clip_model.to(device, dtype=dtype)
        face_helper.face_det.to(device)
        face_helper.face_parse.to(device)
        
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        consis_id_model = {
            "face_helper": face_helper,
            "face_clip_model": face_clip_model,
            "handler_ante": handler_ante,
            "eva_transform_mean": eva_transform_mean,
            "eva_transform_std": eva_transform_std,
            "face_main_model": face_main_model,
            "dtype": dtype,
        }

        return consis_id_model,

class ConsisIDFaceEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "consis_id_model": ("CONSISIDMODEL",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONSISID_CONDS", "IMAGE",)
    RETURN_NAMES = ("consis_id_conds", "face_image", )
    FUNCTION = "faceencode"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Downloads and loads the selected CogVideo model from Huggingface to 'ComfyUI/models/CogVideo'"

    def faceencode(self, image, consis_id_model):
        from .consis_id.models.utils import process_face_embeddings

        device = mm.get_torch_device()
        dtype = consis_id_model["dtype"]
        
        id_image = image[0].cpu().numpy() * 255

        face_helper = consis_id_model["face_helper"]
        face_clip_model = consis_id_model["face_clip_model"]
        handler_ante = consis_id_model["handler_ante"]
        eva_transform_mean = consis_id_model["eva_transform_mean"]
        eva_transform_std = consis_id_model["eva_transform_std"]
        face_main_model = consis_id_model["face_main_model"]
        id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(face_helper, face_clip_model, handler_ante, 
                                                                                eva_transform_mean, eva_transform_std, 
                                                                                face_main_model, device, dtype, id_image, 
                                                                                original_id_image=id_image, is_align_face=True, 
                                                                                cal_uncond=False)
        consis_id_conds = {
            "id_cond": id_cond,
            "id_vit_hidden": id_vit_hidden,
            #"align_crop_face_image": align_crop_face_image,
            #"face_kps": face_kps
        }
        print(align_crop_face_image.shape)
        align_crop_face_image = align_crop_face_image.permute(0, 2, 3, 1).float().cpu()
        return consis_id_conds, align_crop_face_image,
    
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadConsisIDModel": DownloadAndLoadConsisIDModel,
    "ConsisIDFaceEncode": ConsisIDFaceEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadConsisIDModel": "DownloadAndLoadConsisIDModel",
    "ConsisIDFaceEncode": "ConsisID FaceEncode",
    }