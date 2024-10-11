import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from einops import rearrange
import importlib.metadata

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.30.3'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

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


from diffusers.models import AutoencoderKLCogVideoX
from .custom_cogvideox_transformer_3d import CogVideoXTransformer3DModel
from .pipeline_cogvideox import CogVideoXPipeline
from contextlib import nullcontext

from .cogvideox_fun.transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelFun
from .cogvideox_fun.fun_pab_transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelFunPAB
from .cogvideox_fun.autoencoder_magvit import AutoencoderKLCogVideoX as AutoencoderKLCogVideoXFun
from .cogvideox_fun.utils import get_image_to_video_latent, get_video_to_video_latent, ASPECT_RATIO_512, get_closest_ratio, to_pil
from .cogvideox_fun.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from .cogvideox_fun.pipeline_cogvideox_control import CogVideoX_Fun_Pipeline_Control

from PIL import Image
import numpy as np
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "CogVideo" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("CogVideo", os.path.join(folder_paths.models_dir, "CogVideo"))
if not "cogvideox_loras" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("cogvideox_loras", os.path.join(folder_paths.models_dir, "CogVideo", "loras"))

class PABConfig:
    def __init__(
        self,
        steps: int,
        cross_broadcast: bool = False,
        cross_threshold: list = None,
        cross_range: int = None,
        spatial_broadcast: bool = False,
        spatial_threshold: list = None,
        spatial_range: int = None,
        temporal_broadcast: bool = False,
        temporal_threshold: list = None,
        temporal_range: int = None,
        mlp_broadcast: bool = False,
        mlp_spatial_broadcast_config: dict = None,
        mlp_temporal_broadcast_config: dict = None,
    ):
        self.steps = steps

        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold
        self.cross_range = cross_range

        self.spatial_broadcast = spatial_broadcast
        self.spatial_threshold = spatial_threshold
        self.spatial_range = spatial_range

        self.temporal_broadcast = temporal_broadcast
        self.temporal_threshold = temporal_threshold
        self.temporal_range = temporal_range

        self.mlp_broadcast = mlp_broadcast
        self.mlp_spatial_broadcast_config = mlp_spatial_broadcast_config
        self.mlp_temporal_broadcast_config = mlp_temporal_broadcast_config
        self.mlp_temporal_outputs = {}
        self.mlp_spatial_outputs = {}

class CogVideoXPABConfig(PABConfig):
    def __init__(
        self,
        steps: int = 50,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [100, 850],
        spatial_range: int = 2,
        temporal_broadcast: bool = False,
        temporal_threshold: list = [100, 850],
        temporal_range: int = 4,
        cross_broadcast: bool = False,
        cross_threshold: list = [100, 850],
        cross_range: int = 6,
    ):
        super().__init__(
            steps=steps,
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=temporal_threshold,
            temporal_range=temporal_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range

        )

from .videosys.cogvideox_transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelPAB

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

def remove_specific_blocks(model, block_indices_to_remove):
    import torch.nn as nn
    transformer_blocks = model.transformer_blocks
    new_blocks = [block for i, block in enumerate(transformer_blocks) if i not in block_indices_to_remove]
    model.transformer_blocks = nn.ModuleList(new_blocks)
    
    return model

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
        }

    RETURN_TYPES = ("COGLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "CogVideoWrapper"

    def getlorapath(self, lora, strength):

        cog_lora = {
            "path": folder_paths.get_full_path("cogvideox_loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
        }

        return (cog_lora,)
    
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
                        "alibaba-pai/CogVideoX-Fun-V1.1-2b-InP",
                        "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
                        "alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose",
                        "alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose",
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
                "pab_config": ("PAB_CONFIG", {"default": None}),
                "block_edit": ("TRANSFORMERBLOCKS", {"default": None}),
                "lora": ("COGLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, precision, fp8_transformer="disabled", compile="disabled", enable_sequential_cpu_offload=False, pab_config=None, block_edit=None, lora=None):
        
        check_diffusers_version()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        download_path = folder_paths.get_folder_paths("CogVideo")[0]
        
        if "Fun" in model:
            if not "1.1" in model:
                repo_id = "kijai/CogVideoX-Fun-pruned"
                if "2b" in model:
                    base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-2b-InP") # location of the official model
                    if not os.path.exists(base_path):
                        base_path = os.path.join(download_path, "CogVideoX-Fun-2b-InP")
                elif "5b" in model:
                    base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-5b-InP") # location of the official model
                    if not os.path.exists(base_path):
                        base_path = os.path.join(download_path, "CogVideoX-Fun-5b-InP")
            elif "1.1" in model:
                repo_id = model
                base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", (model.split("/")[-1])) # location of the official model
                if not os.path.exists(base_path):
                    base_path = os.path.join(download_path, (model.split("/")[-1]))
                download_path = base_path

        elif "2b" in model:
            base_path = os.path.join(download_path, "CogVideo2B")
            download_path = base_path
            repo_id = model
        elif "5b" in model:
            base_path = os.path.join(download_path, (model.split("/")[-1]))
            download_path = base_path
            repo_id = model

        if "2b" in model:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
        elif "5b" in model:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
        
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
            if pab_config is not None:
                transformer = CogVideoXTransformer3DModelFunPAB.from_pretrained(base_path, subfolder="transformer")
            else:
                transformer = CogVideoXTransformer3DModelFun.from_pretrained(base_path, subfolder="transformer")
        else:
            if pab_config is not None:
                transformer = CogVideoXTransformer3DModelPAB.from_pretrained(base_path, subfolder="transformer")
            else:
                transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer")
        
        transformer = transformer.to(dtype).to(offload_device)

        #LoRAs
        if lora is not None:
            from .cogvideox_fun.lora_utils import merge_lora, load_lora_into_transformer
            logging.info(f"Merging LoRA weights from {lora['path']} with strength {lora['strength']}")
            if "fun" in model.lower():
                transformer = merge_lora(transformer, lora["path"], lora["strength"])
            else:
                lora_sd = load_torch_file(lora["path"])
                transformer = load_lora_into_transformer(state_dict=lora_sd, transformer=transformer, adapter_name=lora["name"])

        if block_edit is not None:
            transformer = remove_specific_blocks(transformer, block_edit)
        
        #fp8
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
                #transformer.to(torch.float8_e4m3fn)
                for name, param in transformer.named_parameters():
                    if "lora" not in name:
                        param.data = param.data.to(torch.float8_e4m3fn)
        
            if fp8_transformer == "fastmode":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, dtype)

        with open(scheduler_path) as f:
            scheduler_config = json.load(f)
        scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config)     

        # VAE
        if "Fun" in model:
            vae = AutoencoderKLCogVideoXFun.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
            if "Pose" in model:
                pipe = CogVideoX_Fun_Pipeline_Control(vae, transformer, scheduler, pab_config=pab_config)
            else:
                pipe = CogVideoX_Fun_Pipeline_Inpaint(vae, transformer, scheduler, pab_config=pab_config)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)
            pipe = CogVideoXPipeline(vae, transformer, scheduler, pab_config=pab_config)        

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
            fuse_qkv_projections=True if pab_config is None else False,
            )

        pipeline = {
            "pipe": pipe,
            "dtype": dtype,
            "base_path": base_path,
            "onediff": True if compile == "onediff" else False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "scheduler_config": scheduler_config,
            "model_name": model
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
                        "CogVideoX_5b_I2V_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_1_1_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_1_1_Pose_GGUF_Q4_0.safetensors",
                    ],
                ),
            "vae_precision": (["fp16", "fp32", "bf16"], {"default": "bf16", "tooltip": "VAE dtype"}),
            "fp8_fastmode": ("BOOLEAN", {"default": False, "tooltip": "only supported on 4090 and later GPUs"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
            },
            "optional": {
                "pab_config": ("PAB_CONFIG", {"default": None}),
                "block_edit": ("TRANSFORMERBLOCKS", {"default": None}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, vae_precision, fp8_fastmode, load_device, enable_sequential_cpu_offload, pab_config=None, block_edit=None):

        check_diffusers_version()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        vae_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[vae_precision]
        download_path = os.path.join(folder_paths.models_dir, 'CogVideo', 'GGUF')
        gguf_path = os.path.join(folder_paths.models_dir, 'diffusion_models', model) # check MinusZone's model path first
        if not os.path.exists(gguf_path):
            gguf_path = os.path.join(download_path, model)
            if not os.path.exists(gguf_path):
                if "I2V" in model or "1_1" in model:
                    repo_id = "Kijai/CogVideoX_GGUF"
                else:
                    repo_id = "MinusZoneAI/ComfyUI-CogVideoX-MZ"
                log.info(f"Downloading model to: {gguf_path}")
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"*{model}*"],
                    local_dir=download_path,
                    local_dir_use_symlinks=False,
                )
        
        if "5b" in model:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
            transformer_path = os.path.join(script_directory, 'configs', 'transformer_config_5b.json')
        elif "2b" in model:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
            transformer_path = os.path.join(script_directory, 'configs', 'transformer_config_2b.json')
    
        with open(transformer_path) as f:
            transformer_config = json.load(f)

        sd = load_torch_file(gguf_path)

        from . import mz_gguf_loader
        import importlib
        importlib.reload(mz_gguf_loader)

        with mz_gguf_loader.quantize_lazy_load():
            if "fun" in model:
                if "Pose" in model:
                    transformer_config["in_channels"] = 32
                else:
                    transformer_config["in_channels"] = 33
                if pab_config is not None:
                    transformer = CogVideoXTransformer3DModelFunPAB.from_config(transformer_config)
                else:
                    transformer = CogVideoXTransformer3DModelFun.from_config(transformer_config)
            elif "I2V" in model:
                transformer_config["in_channels"] = 32
                if pab_config is not None:
                    transformer = CogVideoXTransformer3DModelPAB.from_config(transformer_config)
                else:
                    transformer = CogVideoXTransformer3DModel.from_config(transformer_config)
            else:
                transformer_config["in_channels"] = 16
                if pab_config is not None:
                    transformer = CogVideoXTransformer3DModelPAB.from_config(transformer_config)
                else:
                    transformer = CogVideoXTransformer3DModel.from_config(transformer_config)

            if "2b" in model:
                for name, param in transformer.named_parameters():
                    if name != "pos_embedding":
                        param.data = param.data.to(torch.float8_e4m3fn)
                    else:
                        param.data = param.data.to(torch.float16)
            else:
                transformer.to(torch.float8_e4m3fn)

            if block_edit is not None:
                transformer = remove_specific_blocks(transformer, block_edit)

            transformer = mz_gguf_loader.quantize_load_state_dict(transformer, sd, device="cpu")
            if load_device == "offload_device":
                transformer.to(offload_device)
            else:
                transformer.to(device)
        
        
        if fp8_fastmode:
           from .fp8_optimization import convert_fp8_linear
           convert_fp8_linear(transformer, vae_dtype)

        
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
            if "Pose" in model:
                pipe = CogVideoX_Fun_Pipeline_Control(vae, transformer, scheduler, pab_config=pab_config)
            else:
                pipe = CogVideoX_Fun_Pipeline_Inpaint(vae, transformer, scheduler, pab_config=pab_config)
        else:
            vae = AutoencoderKLCogVideoX.from_config(vae_config).to(vae_dtype).to(offload_device)
            vae.load_state_dict(vae_sd)
            pipe = CogVideoXPipeline(vae, transformer, scheduler, pab_config=pab_config)

        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        pipeline = {
            "pipe": pipe,
            "dtype": vae_dtype,
            "base_path": model,
            "onediff": True if compile == "onediff" else False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "scheduler_config": scheduler_config,
            "model_name": model
        }

        return (pipeline,)

class DownloadAndLoadCogVideoControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "TheDenk/cogvideox-2b-controlnet-hed-v1",
                        "TheDenk/cogvideox-2b-controlnet-canny-v1",
                    ],
                ),

            },
        }

    RETURN_TYPES = ("COGVIDECONTROLNETMODEL",)
    RETURN_NAMES = ("cogvideo_controlnet", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model):
        from .cogvideo_controlnet import CogVideoXControlnet

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        
        download_path = os.path.join(folder_paths.models_dir, 'CogVideo', 'ControlNet')
        base_path = os.path.join(download_path, (model.split("/")[-1]))
        
        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                ignore_patterns=["*text_encoder*", "*tokenizer*"],
                local_dir=base_path,
                local_dir_use_symlinks=False,
            )

        controlnet = CogVideoXControlnet.from_pretrained(base_path)

        return (controlnet,)
    
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

        vae._clear_fake_context_parallel_cache()
        
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
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, steps, cfg, seed, height, width, num_frames, scheduler, samples=None, 
                denoise_strength=1.0, image_cond_latents=None, context_options=None, controlnet=None):
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

        autocastcondition = not pipeline["onediff"]
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
                scheduler_name=scheduler,
                context_schedule=context_options["context_schedule"] if context_options is not None else None,
                context_frames=context_frames,
                context_stride= context_stride,
                context_overlap= context_overlap,
                freenoise=context_options["freenoise"] if context_options is not None else None,
                controlnet=controlnet
            )
        if not pipeline["cpu_offloading"]:
            pipe.transformer.to(offload_device)
        mm.soft_empty_cache()

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

        return (video,)

class CogVideoXFunSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "video_length": ("INT", {"default": 49, "min": 5, "max": 2048, "step": 4}),
                "base_resolution": ("INT", {"min": 64, "max": 1280, "step": 64, "default": 512, "tooltip": "Base resolution, closest training data bucket resolution is chosen based on the selection."}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (available_schedulers, {"default": 'DDIM'})
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
                "opt_empty_latent": ("LATENT",),
                "noise_aug_strength": ("FLOAT", {"default": 0.0563, "min": 0.0, "max": 1.0, "step": 0.001}),
                "context_options": ("COGCONTEXT", ),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline,  positive, negative, video_length, base_resolution, seed, steps, cfg, scheduler, 
                start_img=None, end_img=None, opt_empty_latent=None, noise_aug_strength=0.0563, context_options=None):
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
        log.info(f"Closest bucket size: {width}x{height}")
        
        # Load Sampler
        if context_options is not None and context_options["context_schedule"] == "temporal_tiling":
            logging.info("Temporal tiling enabled, changing scheduler to CogVideoXDDIM")
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

        generator= torch.Generator(device="cpu").manual_seed(seed)

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
                noise_aug_strength = noise_aug_strength,
                context_schedule=context_options["context_schedule"] if context_options is not None else None,
                context_frames=context_frames,
                context_stride= context_stride,
                context_overlap= context_overlap,
                freenoise=context_options["freenoise"] if context_options is not None else None
            )
        #if not pipeline["cpu_offloading"]:
        #     pipe.transformer.to(offload_device)
        mm.soft_empty_cache()

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
                "base_resolution": ("INT", {"min": 64, "max": 1280, "step": 64, "default": 512, "tooltip": "Base resolution, closest training data bucket resolution is chosen based on the selection."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (available_schedulers,
                    {
                        "default": 'DDIM'
                    }
                ),
                "denoise_strength": ("FLOAT", {"default": 0.70, "min": 0.05, "max": 1.00, "step": 0.01}),
                "validation_video": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, 
                validation_video):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]

        assert "fun" in base_path.lower(), "'Unfun' models not supported in 'CogVideoXFunSampler', use the 'CogVideoSampler'"
        assert "pose" not in base_path.lower(), "'Pose' models not supported in 'CogVideoXFunVid2VidSampler', use the 'CogVideoXFunControlSampler'"

        if not pipeline["cpu_offloading"]:
            pipe.enable_model_cpu_offload(device=device)

        mm.soft_empty_cache()

        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        validation_video = np.array(validation_video.cpu().numpy() * 255, np.uint8)
        original_width, original_height = Image.fromarray(validation_video[0]).size

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # Load Sampler
        scheduler_config = pipeline["scheduler_config"]
        if scheduler in scheduler_mapping:
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
            pipe.scheduler = noise_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        generator= torch.Generator(device).manual_seed(seed)

        autocastcondition = not pipeline["onediff"]
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()
        with autocast_context:
            video_length = int((video_length - 1) // pipe.vae.config.temporal_compression_ratio * pipe.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width))

            # for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
            #     pipeline = merge_lora(pipeline, _lora_path, _lora_weight)

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
            }

            latents = pipe(
                **common_params,
                video=input_video,
                mask_video=input_video_mask,
                strength=float(denoise_strength)
            )

            # for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
            #     pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)
        return (pipeline, {"samples": latents})

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
            logging.info("Temporal tiling enabled, changing scheduler to CogVideoXDDIM")
            scheduler="CogVideoXDDIM"
        if scheduler in scheduler_mapping:
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
            pipe.scheduler = noise_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        generator=torch.Generator(torch.device("cpu")).manual_seed(seed)

        autocastcondition = not pipeline["onediff"]
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

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": DownloadAndLoadCogVideoModel,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoDecode": CogVideoDecode,
    "CogVideoTextEncode": CogVideoTextEncode,
    "CogVideoDualTextEncode_311": CogVideoDualTextEncode_311,
    "CogVideoImageEncode": CogVideoImageEncode,
    "CogVideoXFunSampler": CogVideoXFunSampler,
    "CogVideoXFunVid2VidSampler": CogVideoXFunVid2VidSampler,
    "CogVideoXFunControlSampler": CogVideoXFunControlSampler,
    "CogVideoTextEncodeCombine": CogVideoTextEncodeCombine,
    "DownloadAndLoadCogVideoGGUFModel": DownloadAndLoadCogVideoGGUFModel,
    "CogVideoPABConfig": CogVideoPABConfig,
    "CogVideoTransformerEdit": CogVideoTransformerEdit,
    "CogVideoControlImageEncode": CogVideoControlImageEncode,
    "CogVideoLoraSelect": CogVideoLoraSelect,
    "CogVideoContextOptions": CogVideoContextOptions,
    "CogVideoControlNet": CogVideoControlNet,
    "DownloadAndLoadCogVideoControlNet": DownloadAndLoadCogVideoControlNet
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
    "CogVideoXFunControlSampler": "CogVideoXFun Control Sampler",
    "CogVideoTextEncodeCombine": "CogVideo TextEncode Combine",
    "DownloadAndLoadCogVideoGGUFModel": "(Down)load CogVideo GGUF Model",
    "CogVideoPABConfig": "CogVideo PABConfig",
    "CogVideoTransformerEdit": "CogVideo TransformerEdit",
    "CogVideoControlImageEncode": "CogVideo Control ImageEncode",
    "CogVideoLoraSelect": "CogVideo LoraSelect",
    "CogVideoContextOptions": "CogVideo Context Options",
    "DownloadAndLoadCogVideoControlNet": "(Down)load CogVideo ControlNet"
    }
