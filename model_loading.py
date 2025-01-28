import os
import json
import folder_paths
import comfy.model_management as mm
from typing import Union

def patched_write_atomic(
    path_: str,
    content: Union[str, bytes],
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    from pathlib import Path
    import os
    import shutil
    import threading
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    shutil.copy2(src=tmp_path, dst=path) #changed to allow overwriting cache files
    os.remove(tmp_path)
try:
    import torch._inductor.codecache
    torch._inductor.codecache.write_atomic = patched_write_atomic
except:
    pass

import torch
import torch.nn as nn

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler
from .custom_cogvideox_transformer_3d import CogVideoXTransformer3DModel
from .pipeline_cogvideox import CogVideoXPipeline
from contextlib import nullcontext

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from .utils import remove_specific_blocks, log
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

class CogVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("cogvideox_loras"), 
                {"tooltip": "LORA models are expected to be in ComfyUI/models/CogVideo/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("COGLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "fuse_lora": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LoRA weights into the transformer"}),
            }
        }

    RETURN_TYPES = ("COGLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/CogVideo/loras"

    def getlorapath(self, lora, strength, prev_lora=None, fuse_lora=False):
        cog_loras_list = []

        cog_lora = {
            "path": folder_paths.get_full_path("cogvideox_loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora
        }
        if prev_lora is not None:
            cog_loras_list.extend(prev_lora)
            
        cog_loras_list.append(cog_lora)
        print(cog_loras_list)
        return (cog_loras_list,)

class CogVideoLoraSelectComfy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"), 
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("COGLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "fuse_lora": ("BOOLEAN", {"default": False, "tooltip": "Fuse the LoRA weights into the transformer"}),
            }
        }

    RETURN_TYPES = ("COGLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, prev_lora=None, fuse_lora=False):
        cog_loras_list = []

        cog_lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora
        }
        if prev_lora is not None:
            cog_loras_list.extend(prev_lora)
            
        cog_loras_list.append(cog_lora)
        print(cog_loras_list)
        return (cog_loras_list,)
    
#region DownloadAndLoadCogVideoModel
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
                        "kijai/CogVideoX-5b-1.5-T2V",
                        "kijai/CogVideoX-5b-1.5-I2V",
                        "bertjiazheng/KoolCogVideoX-5b",
                        "kijai/CogVideoX-Fun-2b",
                        "kijai/CogVideoX-Fun-5b",
                        "kijai/CogVideoX-5b-Tora",
                        "alibaba-pai/CogVideoX-Fun-V1.1-2b-InP",
                        "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
                        "alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose",
                        "alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose",
                        "alibaba-pai/CogVideoX-Fun-V1.1-5b-Control",
                        "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP",
                        "feizhengcong/CogvideoX-Interpolation",
                        "NimVideo/cogvideox-2b-img2vid"
                    ],
                ),

            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16", "tooltip": "official recommendation is that 2b model should be fp16, 5b model should be bf16"}
                ),
                "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fastmode', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6"], {"default": 'disabled', "tooltip": "enabled casts the transformer to torch.float8_e4m3fn, fastmode is only for latest nvidia GPUs and requires torch 2.4.0 and cu124 minimum"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
                "block_edit": ("TRANSFORMERBLOCKS", {"default": None}),
                "lora": ("COGLORA", {"default": None}),
                "compile_args":("COMPILEARGS", ),
                "attention_mode": ([
                    "sdpa",
                    "fused_sdpa",
                    "sageattn",
                    "fused_sageattn", 
                    "sageattn_qk_int8_pv_fp8_cuda",
                    "sageattn_qk_int8_pv_fp16_cuda",
                    "sageattn_qk_int8_pv_fp16_triton",
                    "fused_sageattn_qk_int8_pv_fp8_cuda",
                    "fused_sageattn_qk_int8_pv_fp16_cuda",
                    "fused_sageattn_qk_int8_pv_fp16_triton",
                    "comfy"
                    ], {"default": "sdpa"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOMODEL", "VAE",)
    RETURN_NAMES = ("model", "vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Downloads and loads the selected CogVideo model from Huggingface to 'ComfyUI/models/CogVideo'"

    def loadmodel(self, model, precision, quantization="disabled", compile="disabled", 
                  enable_sequential_cpu_offload=False, block_edit=None, lora=None, compile_args=None, 
                  attention_mode="sdpa", load_device="main_device"):
        
        transformer = None

        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")
            if "qk_int8" in attention_mode:
                try:
                    from sageattention import sageattn_qk_int8_pv_fp16_cuda
                except Exception as e:
                    raise ValueError(f"Can't import SageAttention 2.0.0: {str(e)}")
        
        if precision == "fp16" and "1.5" in model:
            raise ValueError("1.5 models do not currently work in fp16")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        download_path = folder_paths.get_folder_paths("CogVideo")[0]
        
        if "Fun" in model:
            if "1.1" not in model and "1.5" not in model:
                repo_id = "kijai/CogVideoX-Fun-pruned"
                if "2b" in model:
                    base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-2b-InP") # location of the official model
                    if not os.path.exists(base_path):
                        base_path = os.path.join(download_path, "CogVideoX-Fun-2b-InP")
                elif "5b" in model:
                    base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", "CogVideoX-Fun-5b-InP") # location of the official model
                    if not os.path.exists(base_path):
                        base_path = os.path.join(download_path, "CogVideoX-Fun-5b-InP")
            else:
                repo_id = model
                base_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", (model.split("/")[-1])) # location of the official model
                if not os.path.exists(base_path):
                    base_path = os.path.join(download_path, (model.split("/")[-1]))
                download_path = base_path
            subfolder = "transformer"
            allow_patterns = ["*transformer*", "*scheduler*", "*vae*"]

        elif "2b" in model:
            if 'img2vid' in model:
                base_path = os.path.join(download_path, "cogvideox-2b-img2vid")
                download_path = base_path
                repo_id = model
            else:
                base_path = os.path.join(download_path, "CogVideo2B")
                download_path = base_path
                repo_id = model
            subfolder = "transformer"
            allow_patterns = ["*transformer*", "*scheduler*", "*vae*"]
        elif "1.5-T2V" in model or "1.5-I2V" in model:
            base_path = os.path.join(download_path, "CogVideoX-5b-1.5")
            download_path = base_path
            subfolder = "transformer_T2V" if "1.5-T2V" in model else "transformer_I2V"
            allow_patterns = [f"*{subfolder}*", "*vae*", "*scheduler*"]
            repo_id = "kijai/CogVideoX-5b-1.5"
        else:
            base_path = os.path.join(download_path, (model.split("/")[-1]))
            download_path = base_path
            repo_id = model
            subfolder = "transformer"
            allow_patterns = ["*transformer*", "*scheduler*", "*vae*"]

        if "2b" in model:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
        else:
            scheduler_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
        
        if not os.path.exists(base_path) or not os.path.exists(os.path.join(base_path, subfolder)):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=["*text_encoder*", "*tokenizer*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder=subfolder, attention_mode=attention_mode)
        transformer = transformer.to(dtype).to(transformer_load_device)

        if "1.5" in model and not "fun" in model:
            transformer.config.sample_height = 300
            transformer.config.sample_width = 300

        if block_edit is not None:
            transformer = remove_specific_blocks(transformer, block_edit)

        with open(scheduler_path) as f:
            scheduler_config = json.load(f)
        scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config)     

        # VAE
        vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae").to(dtype).to(offload_device)

        #pipeline
        pipe = CogVideoXPipeline(
            transformer, 
            scheduler, 
            dtype=dtype, 
            is_fun_inpaint="fun" in model.lower() and not ("pose" in model.lower() or "control" in model.lower())
            )
        if "cogvideox-2b-img2vid" in model:
            pipe.input_with_padding = False
        
        #LoRAs
        if lora is not None:
            dimensionx_loras = ["orbit", "dimensionx"] # for now dimensionx loras need scaling
            dimensionx_lora = False
            adapter_list = []
            adapter_weights = []
            for l in lora:
                if any(item in l["path"].lower() for item in dimensionx_loras):
                    dimensionx_lora = True
                fuse = True if l["fuse_lora"] else False
                lora_sd = load_torch_file(l["path"])
                lora_rank = None            
                for key, val in lora_sd.items():
                    if "lora_B" in key:
                        lora_rank = val.shape[1]
                        break
                if lora_rank is not None:
                    log.info(f"Merging rank {lora_rank} LoRA weights from {l['path']} with strength {l['strength']}")
                    adapter_name = l['path'].split("/")[-1].split(".")[0]
                    adapter_weight = l['strength']
                    pipe.load_lora_weights(l['path'], weight_name=l['path'].split("/")[-1], lora_rank=lora_rank, adapter_name=adapter_name)
                    
                    adapter_list.append(adapter_name)
                    adapter_weights.append(adapter_weight)
                else:
                    try: #Fun trainer LoRAs are loaded differently
                        from .lora_utils import merge_lora
                        log.info(f"Merging LoRA weights from {l['path']} with strength {l['strength']}")
                        pipe.transformer = merge_lora(pipe.transformer, l["path"], l["strength"], device=transformer_load_device, state_dict=lora_sd)
                    except:
                        raise ValueError(f"Can't recognize LoRA {l['path']}")
                del lora_sd
                mm.soft_empty_cache()
            if adapter_list:
                pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
                if fuse:
                    lora_scale = 1
                    if dimensionx_lora:
                        lora_scale = lora_scale / lora_rank
                    pipe.fuse_lora(lora_scale=lora_scale, components=["transformer"])
                    pipe.delete_adapters(adapter_list)
           

        if "fused" in attention_mode:
            from diffusers.models.attention import Attention
            pipe.transformer.fuse_qkv_projections = True
            for module in pipe.transformer.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=True)

        if compile_args is not None:
            pipe.transformer.to(memory_format=torch.channels_last)

        #fp8
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fastmode":
            params_to_keep = {"patch_embed", "lora", "pos_embedding", "time_embedding", "norm_k", "norm_q", "to_k.bias", "to_q.bias", "to_v.bias"}
            if "1.5" in model:
                    params_to_keep.update({"norm1.linear.weight", "ofs_embedding", "norm_final", "norm_out", "proj_out"}) 
            for name, param in pipe.transformer.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    param.data = param.data.to(torch.float8_e4m3fn)
        
            if quantization == "fp8_e4m3fn_fastmode":
                from .fp8_optimization import convert_fp8_linear
                if "1.5" in model:
                    params_to_keep.update({"ff"}) #otherwise NaNs
                convert_fp8_linear(pipe.transformer, dtype, params_to_keep=params_to_keep)
        
        # compilation
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            for i, block in enumerate(pipe.transformer.transformer_blocks):
                if "CogVideoXBlock" in str(block):
                    pipe.transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        if "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight
            )
            except:
                raise ImportError("torchao is not installed, please install torchao to use fp8dq")

            def filter_fn(module: nn.Module, fqn: str) -> bool:
                target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
                if any(sub in fqn for sub in target_submodules):
                    return isinstance(module, nn.Linear)
                return False
            
            if "fp6" in quantization: #slower for some reason on 4090
                quant_func = fpx_weight_only(3, 2)
            elif "fp8dq" in quantization: #very fast on 4090 when compiled
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()
        
            for i, block in enumerate(pipe.transformer.transformer_blocks):
                if "CogVideoXBlock" in str(block):
                    quantize_(block, quant_func, filter_fn=filter_fn)
                        
            manual_offloading = False # to disable manual .to(device) calls
        
        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
            manual_offloading = False

        # CogVideoXBlock(
        #     (norm1): CogVideoXLayerNormZero(
        #         (silu): SiLU()
        #         (linear): Linear(in_features=512, out_features=18432, bias=True)
        #         (norm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
        #     )
        #     (attn1): Attention(
        #         (norm_q): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #         (norm_k): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        #         (to_q): Linear(in_features=3072, out_features=3072, bias=True)
        #         (to_k): Linear(in_features=3072, out_features=3072, bias=True)
        #         (to_v): Linear(in_features=3072, out_features=3072, bias=True)
        #         (to_out): ModuleList(
        #         (0): Linear(in_features=3072, out_features=3072, bias=True)
        #         (1): Dropout(p=0.0, inplace=False)
        #         )
        #     )
        #     (norm2): CogVideoXLayerNormZero(
        #         (silu): SiLU()
        #         (linear): Linear(in_features=512, out_features=18432, bias=True)
        #         (norm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
        #     )
        #     (ff): FeedForward(
        #         (net): ModuleList(
        #         (0): GELU(
        #             (proj): Linear(in_features=3072, out_features=12288, bias=True)
        #         )
        #         (1): Dropout(p=0.0, inplace=False)
        #         (2): Linear(in_features=12288, out_features=3072, bias=True)
        #         (3): Dropout(p=0.0, inplace=False)
        #         )
        #     )
        #     )               
       
        # if compile == "onediff":
        #     from onediffx import compile_pipe
        #     os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
            
        #     pipe = compile_pipe(
        #     pipe,
        #     backend="nexfort",
        #     options= {"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": False, "triton.fuse_attention_allow_fp16_reduction": False}},
        #     ignores=["vae"],
        #     fuse_qkv_projections= False,
        #     )          
        
        pipeline = {
            "pipe": pipe,
            "dtype": dtype,
            "quantization": quantization,
            "base_path": base_path,
            "onediff": True if compile == "onediff" else False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "manual_offloading": manual_offloading,
            "scheduler_config": scheduler_config,
            "model_name": model,
        }

        return (pipeline, vae)
#region GGUF
class DownloadAndLoadCogVideoGGUFModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "CogVideoX_5b_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_I2V_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_1_5_I2V_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_1_1_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_fun_1_1_Pose_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_Interpolation_GGUF_Q4_0.safetensors",
                        "CogVideoX_5b_Tora_GGUF_Q4_0.safetensors",
                    ],
                ),
            "vae_precision": (["fp16", "fp32", "bf16"], {"default": "bf16", "tooltip": "VAE dtype"}),
            "fp8_fastmode": ("BOOLEAN", {"default": False, "tooltip": "only supported on 4090 and later GPUs, also requires torch 2.4.0 with cu124 minimum"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
            },
            "optional": {
                "block_edit": ("TRANSFORMERBLOCKS", {"default": None}),
                #"compile_args":("COMPILEARGS", ),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOMODEL", "VAE",)
    RETURN_NAMES = ("model", "vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, vae_precision, fp8_fastmode, load_device, enable_sequential_cpu_offload, 
                  block_edit=None, compile_args=None, attention_mode="sdpa"):
        
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        vae_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[vae_precision]
        download_path = os.path.join(folder_paths.models_dir, 'CogVideo', 'GGUF')
        gguf_path = os.path.join(folder_paths.models_dir, 'diffusion_models', model) # check MinusZone's model path first
        if not os.path.exists(gguf_path):
            gguf_path = os.path.join(download_path, model)
            if not os.path.exists(gguf_path):
                if "I2V" in model or "1_1" in model or "Interpolation" in model or "Tora" in model:
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


        from . import mz_gguf_loader
        import importlib
        importlib.reload(mz_gguf_loader)

        with mz_gguf_loader.quantize_lazy_load():
            if "fun" in model:
                if "Pose" in model:
                    transformer_config["in_channels"] = 32
                else:
                    transformer_config["in_channels"] = 33
            elif "I2V" in model or "Interpolation" in model:
                transformer_config["in_channels"] = 32
                if "1_5" in model:
                    transformer_config["ofs_embed_dim"] = 512
                    transformer_config["use_learned_positional_embeddings"] = False
                    transformer_config["patch_size_t"] = 2
                    transformer_config["patch_bias"] = False
                    transformer_config["sample_height"] = 300
                    transformer_config["sample_width"] = 300
            else:
                transformer_config["in_channels"] = 16

            transformer = CogVideoXTransformer3DModel.from_config(transformer_config, attention_mode=attention_mode)
            cast_dtype = vae_dtype
            params_to_keep = {"patch_embed", "pos_embedding", "time_embedding"}
            if "2b" in model:
                cast_dtype = torch.float16
            elif "1_5" in model:
                params_to_keep = {"norm1.linear.weight", "patch_embed", "time_embedding", "ofs_embedding", "norm_final", "norm_out", "proj_out"}
                cast_dtype = torch.bfloat16
            for name, param in transformer.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    param.data = param.data.to(torch.float8_e4m3fn)
                else:
                    param.data = param.data.to(cast_dtype)
            #for name, param in transformer.named_parameters():
             #       print(name, param.data.dtype)
           
            if block_edit is not None:
                transformer = remove_specific_blocks(transformer, block_edit)

        transformer.attention_mode = attention_mode
        
        if fp8_fastmode:
           params_to_keep = {"patch_embed", "lora", "pos_embedding", "time_embedding"}
           if "1.5" in model:
                params_to_keep.update({"ff","norm1.linear.weight", "norm_k", "norm_q","ofs_embedding", "norm_final", "norm_out", "proj_out"})   
           from .fp8_optimization import convert_fp8_linear
           convert_fp8_linear(transformer, vae_dtype, params_to_keep=params_to_keep)

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
        
        #VAE
        vae_sd = load_torch_file(vae_path)
        vae = AutoencoderKLCogVideoX.from_config(vae_config).to(vae_dtype).to(offload_device)
        vae.load_state_dict(vae_sd)
        del vae_sd
        pipe = CogVideoXPipeline(
            transformer, 
            scheduler, 
            dtype=vae_dtype,
            is_fun_inpaint="fun" in model.lower() and not ("pose" in model.lower() or "control" in model.lower())
            )

        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        sd = load_torch_file(gguf_path)
        pipe.transformer = mz_gguf_loader.quantize_load_state_dict(pipe.transformer, sd, device="cpu")
        del sd

        if load_device == "offload_device":
            pipe.transformer.to(offload_device)
        else:
            pipe.transformer.to(device)

        pipeline = {
            "pipe": pipe,
            "dtype": vae_dtype,
            "quantization": "GGUF",
            "base_path": model,
            "onediff": False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "scheduler_config": scheduler_config,
            "model_name": model,
            "manual_offloading": True,
        }

        return (pipeline, vae)

#region ModelLoader
class CogVideoXModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            
            "base_precision": (["fp16", "fp32", "bf16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
            },
            "optional": {
                "block_edit": ("TRANSFORMERBLOCKS", {"default": None}),
                "lora": ("COGLORA", {"default": None}),
                "compile_args":("COMPILEARGS", ),
                "attention_mode": ([
                    "sdpa",
                    "fused_sdpa",
                    "sageattn",
                    "fused_sageattn", 
                    "sageattn_qk_int8_pv_fp8_cuda",
                    "sageattn_qk_int8_pv_fp16_cuda",
                    "sageattn_qk_int8_pv_fp16_triton",
                    "fused_sageattn_qk_int8_pv_fp8_cuda",
                    "fused_sageattn_qk_int8_pv_fp16_cuda",
                    "fused_sageattn_qk_int8_pv_fp16_triton",
                    "comfy"
                    ], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("COGVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"

    def loadmodel(self, model, base_precision, load_device, enable_sequential_cpu_offload, 
                  block_edit=None, compile_args=None, lora=None, attention_mode="sdpa", quantization="disabled"):
        transformer = None
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        mm.soft_empty_cache()

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device)

        model_type = ""
        if sd["patch_embed.proj.weight"].shape == (3072, 33, 2, 2):
            model_type = "fun_5b"
        elif sd["patch_embed.proj.weight"].shape == (3072, 16, 2, 2):
            model_type = "5b"
        elif sd["patch_embed.proj.weight"].shape == (3072, 128):
            model_type = "5b_1_5"
        elif sd["patch_embed.proj.weight"].shape == (3072, 256):
            model_type = "5b_I2V_1_5"
        elif sd["patch_embed.proj.weight"].shape == (1920, 33, 2, 2):
            model_type = "fun_2b"
        elif sd["patch_embed.proj.weight"].shape == (1920, 32, 2, 2):
            model_type = "cogvideox-2b-img2vid"
        elif sd["patch_embed.proj.weight"].shape == (1920, 16, 2, 2):
            model_type = "2b"
        elif sd["patch_embed.proj.weight"].shape == (3072, 32, 2, 2):
            if "pos_embedding" in sd:
                model_type = "fun_5b_pose"
            else:
                model_type = "I2V_5b"
        else:
            raise Exception("Selected model is not recognized")
        log.info(f"Detected CogVideoX model type: {model_type}")

        if "5b" in model_type:
            scheduler_config_path = os.path.join(script_directory, 'configs', 'scheduler_config_5b.json')
            transformer_config_path = os.path.join(script_directory, 'configs', 'transformer_config_5b.json')
        elif "2b" in model_type:
            scheduler_config_path = os.path.join(script_directory, 'configs', 'scheduler_config_2b.json')
            transformer_config_path = os.path.join(script_directory, 'configs', 'transformer_config_2b.json')
    
        with open(transformer_config_path) as f:
            transformer_config = json.load(f)

            if model_type in ["I2V", "I2V_5b", "fun_5b_pose", "5b_I2V_1_5", "cogvideox-2b-img2vid"]:
                transformer_config["in_channels"] = 32
                if "1_5" in model_type:
                    transformer_config["ofs_embed_dim"] = 512
            elif "fun" in model_type:
                transformer_config["in_channels"] = 33
            else:
                transformer_config["in_channels"] = 16
            if "1_5" in model_type:
                    transformer_config["use_learned_positional_embeddings"] = False
                    transformer_config["patch_size_t"] = 2
                    transformer_config["patch_bias"] = False
                    transformer_config["sample_height"] = 300
                    transformer_config["sample_width"] = 300

        with init_empty_weights():
            transformer = CogVideoXTransformer3DModel.from_config(transformer_config, attention_mode=attention_mode)

        #load weights
        #params_to_keep = {}
        log.info("Using accelerate to load and assign model weights to device...")
        
        for name, param in transformer.named_parameters():
            #dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])
        del sd
        # TODO fix for transformer model patch_embed.pos_embedding dtype 
        #   or at add line  ComfyUI-CogVideoXWrapper/embeddings.py:129 code
        #   pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
        transformer = transformer.to(base_dtype).to(transformer_load_device)

        #scheduler
        with open(scheduler_config_path) as f:
            scheduler_config = json.load(f)
        scheduler = CogVideoXDDIMScheduler.from_config(scheduler_config, subfolder="scheduler")

        if block_edit is not None:
            transformer = remove_specific_blocks(transformer, block_edit)

        if "fused" in attention_mode:
            from diffusers.models.attention import Attention
            transformer.fuse_qkv_projections = True
            for module in transformer.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=True)
        transformer.attention_mode = attention_mode

        pipe = CogVideoXPipeline(
            transformer, 
            scheduler, 
            dtype=base_dtype, 
            is_fun_inpaint="fun" in model.lower() and not ("pose" in model.lower() or "control" in model.lower())
            )
        if "cogvideox-2b-img2vid" == model_type:
            pipe.input_with_padding = False
        if enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        #LoRAs
        if lora is not None:
            dimensionx_loras = ["orbit", "dimensionx"] # for now dimensionx loras need scaling
            dimensionx_lora = False
            adapter_list = []
            adapter_weights = []
            for l in lora:
                if any(item in l["path"].lower() for item in dimensionx_loras):
                    dimensionx_lora = True
                fuse = True if l["fuse_lora"] else False
                lora_sd = load_torch_file(l["path"])
                lora_rank = None            
                for key, val in lora_sd.items():
                    if "lora_B" in key:
                        lora_rank = val.shape[1]
                        break
                if lora_rank is not None:
                    log.info(f"Merging rank {lora_rank} LoRA weights from {l['path']} with strength {l['strength']}")
                    adapter_name = l['path'].split("/")[-1].split(".")[0]
                    adapter_weight = l['strength']
                    pipe.load_lora_weights(l['path'], weight_name=l['path'].split("/")[-1], lora_rank=lora_rank, adapter_name=adapter_name)
                    
                    adapter_list.append(adapter_name)
                    adapter_weights.append(adapter_weight)
                else:
                    try: #Fun trainer LoRAs are loaded differently
                        from .lora_utils import merge_lora
                        log.info(f"Merging LoRA weights from {l['path']} with strength {l['strength']}")
                        pipe.transformer = merge_lora(pipe.transformer, l["path"], l["strength"], device=transformer_load_device, state_dict=lora_sd)
                    except:
                        raise ValueError(f"Can't recognize LoRA {l['path']}")
            if adapter_list:
                pipe.set_adapters(adapter_list, adapter_weights=adapter_weights)
                if fuse:
                    lora_scale = 1
                    if dimensionx_lora:
                        lora_scale = lora_scale / lora_rank
                    pipe.fuse_lora(lora_scale=lora_scale, components=["transformer"])

        if compile_args is not None:
            pipe.transformer.to(memory_format=torch.channels_last)

        #quantization
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast":
            params_to_keep = {"patch_embed", "lora", "pos_embedding", "time_embedding", "norm_k", "norm_q", "to_k.bias", "to_q.bias", "to_v.bias"}
            if "1.5" in model:
                    params_to_keep.update({"norm1.linear.weight", "ofs_embedding", "norm_final", "norm_out", "proj_out"}) 
            for name, param in pipe.transformer.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    param.data = param.data.to(torch.float8_e4m3fn)
        
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                if "1.5" in model:
                    params_to_keep.update({"ff"}) #otherwise NaNs
                convert_fp8_linear(pipe.transformer, base_dtype, params_to_keep=params_to_keep)
        
        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            for i, block in enumerate(pipe.transformer.transformer_blocks):
                if "CogVideoXBlock" in str(block):
                    pipe.transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        if "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight
            )
            except:
                raise ImportError("torchao is not installed, please install torchao to use fp8dq")

            def filter_fn(module: nn.Module, fqn: str) -> bool:
                target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
                if any(sub in fqn for sub in target_submodules):
                    return isinstance(module, nn.Linear)
                return False
            
            if "fp6" in quantization: #slower for some reason on 4090
                quant_func = fpx_weight_only(3, 2)
            elif "fp8dq" in quantization: #very fast on 4090 when compiled
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()
        
            for i, block in enumerate(pipe.transformer.transformer_blocks):
                if "CogVideoXBlock" in str(block):
                    quantize_(block, quant_func, filter_fn=filter_fn)
            
            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")

        pipeline = {
            "pipe": pipe,
            "dtype": base_dtype,
            "quantization": quantization,
            "base_path": model,
            "onediff": False,
            "cpu_offloading": enable_sequential_cpu_offload,
            "scheduler_config": scheduler_config,
            "model_name": model,
            "manual_offloading": manual_offloading,
        }
        return (pipeline,)
    
#region VAE

class CogVideoXVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args":("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Loads CogVideoX VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        with open(os.path.join(script_directory, 'configs', 'vae_config.json')) as f:
            vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path)

        vae = AutoencoderKLCogVideoX.from_config(vae_config).to(dtype).to(offload_device)
        vae.load_state_dict(vae_sd)
        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        return (vae,)
    
#region Tora
class DownloadAndLoadToraModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "kijai/CogVideoX-5b-Tora",
                        "kijai/CogVideoX-5b-Tora-I2V",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("TORAMODEL",)
    RETURN_NAMES = ("tora_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoWrapper"
    DESCRIPTION = "Downloads and loads the the Tora model from Huggingface to 'ComfyUI/models/CogVideo/CogVideoX-5b-Tora'"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
  
        download_path = folder_paths.get_folder_paths("CogVideo")[0]

        from .tora.traj_module import MGF

        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            is_accelerate_available = True
        except:
            is_accelerate_available = False
            pass

        download_path = os.path.join(folder_paths.models_dir, 'CogVideo', "CogVideoX-5b-Tora")

        
        fuser_model = "fuser.safetensors" if not "I2V" in model else "fuser_I2V.safetensors"
        fuser_path = os.path.join(download_path, "fuser", fuser_model)
        if not os.path.exists(fuser_path):
            log.info(f"Downloading Fuser model to: {fuser_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                allow_patterns=[fuser_model],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        hidden_size = 3072
        num_layers = 42

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            fuser_list = nn.ModuleList([MGF(128, hidden_size) for _ in range(num_layers)])
        
        fuser_sd = load_torch_file(fuser_path)
        if is_accelerate_available:
            for key in fuser_sd:
                set_module_tensor_to_device(fuser_list, key, dtype=torch.float16, device=device, value=fuser_sd[key])
        else:
            fuser_list.load_state_dict(fuser_sd)
            for module in fuser_list:
                for param in module.parameters():
                    param.data = param.data.to(torch.bfloat16).to(device)
        del fuser_sd

        traj_extractor_model = "traj_extractor.safetensors" if not "I2V" in model else "traj_extractor_I2V.safetensors"
        traj_extractor_path = os.path.join(download_path, "traj_extractor", traj_extractor_model)
        if not os.path.exists(traj_extractor_path):
            log.info(f"Downloading trajectory extractor model to: {traj_extractor_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="kijai/CogVideoX-5b-Tora",
                allow_patterns=[traj_extractor_model],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        from .tora.traj_module import TrajExtractor
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            traj_extractor = TrajExtractor(
                vae_downsize=(4, 8, 8),
                patch_size=2,
                nums_rb=2,
                cin=16,
                channels=[128] * 42,
                sk=True,
                use_conv=False,
            )
    
        traj_sd = load_torch_file(traj_extractor_path)
        if is_accelerate_available:
            for key in traj_sd:
                set_module_tensor_to_device(traj_extractor, key, dtype=torch.float32, device=device, value=traj_sd[key])
        else:
            traj_extractor.load_state_dict(traj_sd)
            traj_extractor.to(torch.float32).to(device)

        toramodel = {
            "fuser_list": fuser_list,
            "traj_extractor": traj_extractor,
        }

        return (toramodel,)
#region controlnet
class DownloadAndLoadCogVideoControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "TheDenk/cogvideox-2b-controlnet-hed-v1",
                        "TheDenk/cogvideox-2b-controlnet-canny-v1",
                        "TheDenk/cogvideox-5b-controlnet-hed-v1",
                        "TheDenk/cogvideox-5b-controlnet-canny-v1"
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
    
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": DownloadAndLoadCogVideoModel,
    "DownloadAndLoadCogVideoGGUFModel": DownloadAndLoadCogVideoGGUFModel,
    "DownloadAndLoadCogVideoControlNet": DownloadAndLoadCogVideoControlNet,
    "DownloadAndLoadToraModel": DownloadAndLoadToraModel,
    "CogVideoLoraSelect": CogVideoLoraSelect,
    "CogVideoXVAELoader": CogVideoXVAELoader,
    "CogVideoXModelLoader": CogVideoXModelLoader,
    "CogVideoLoraSelectComfy": CogVideoLoraSelectComfy
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadCogVideoModel": "(Down)load CogVideo Model",
    "DownloadAndLoadCogVideoGGUFModel": "(Down)load CogVideo GGUF Model",
    "DownloadAndLoadCogVideoControlNet": "(Down)load CogVideo ControlNet",
    "DownloadAndLoadToraModel": "(Down)load Tora Model",
    "CogVideoLoraSelect": "CogVideo LoraSelect",
    "CogVideoXVAELoader": "CogVideoX VAE Loader",
    "CogVideoXModelLoader": "CogVideoX Model Loader",
    "CogVideoLoraSelectComfy": "CogVideo LoraSelect Comfy"
    }