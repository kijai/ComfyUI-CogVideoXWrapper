# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

import os
import json
import glob

import numpy as np
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.embeddings import apply_rotary_emb
from .embeddings import CogVideoXPatchEmbed

from .consis_id.models.local_facial_extractor import LocalFacialExtractor, PerceiverCrossAttention


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from sageattention import sageattn
    SAGEATTN_IS_AVAILABLE = True
except:
    SAGEATTN_IS_AVAILABLE = False

from comfy.ldm.modules.attention import optimized_attention

@torch.compiler.disable()
def sageattn_func(query, key, value, attn_mask=None, dropout_p=0.0,is_causal=False):
    return sageattn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,is_causal=is_causal)

def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft

class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_mode: Optional[str] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.to_q.weight.dtype == torch.float16 or attn.to_q.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(attn.to_q.weight.dtype)

        if attention_mode != "fused_sdpa" or attention_mode != "fused_sageattn":
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = torch.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                
        if attention_mode == "sageattn" or attention_mode == "fused_sageattn":
            hidden_states = sageattn_func(query, key, value, attn_mask=attention_mask, dropout_p=0.0,is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        elif attention_mode == "sdpa" or attention_mode == "fused_sdpa":
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        elif attention_mode == "comfy":
            hidden_states = optimized_attention(query, key, value, mask=attention_mask, heads=attn.heads, skip_reshape=True)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

#region Blocks
@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
       

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.cached_hidden_states = []
        self.cached_encoder_hidden_states = []
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_flow_feature: Optional[torch.Tensor] = None,
        fuser=None,
        block_use_fastercache=False,
        fastercache_counter=0,
        fastercache_start_step=15,
        fastercache_device="cuda:0",
        attention_mode="sdpa",
    ) -> torch.Tensor:
        #print("hidden_states in block: ", hidden_states.shape) #1.5: torch.Size([2, 3200, 3072]) 10.: torch.Size([2, 6400, 3072])
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        #print("norm_hidden_states in block: ", norm_hidden_states.shape) #torch.Size([2, 3200, 3072])
      
        # Tora Motion-guidance Fuser
        if video_flow_feature is not None:
            H, W = video_flow_feature.shape[-2:]
            T = norm_hidden_states.shape[1] // H // W
            h = rearrange(norm_hidden_states, "B (T H W) C -> (B T) C H W", H=H, W=W)
            h = fuser(h, video_flow_feature.to(h), T=T)
            norm_hidden_states = rearrange(h, "(B T) C H W ->  B (T H W) C", T=T)
            del h, fuser        
        
        #region fastercache
        if block_use_fastercache:
            B = norm_hidden_states.shape[0]
            if fastercache_counter >= fastercache_start_step + 3 and fastercache_counter%3!=0 and self.cached_hidden_states[-1].shape[0] >= B:
                attn_hidden_states = (
                    self.cached_hidden_states[1][:B] + 
                    (self.cached_hidden_states[1][:B] - self.cached_hidden_states[0][:B]) 
                    * 0.3
                    ).to(norm_hidden_states.device, non_blocking=True)
                attn_encoder_hidden_states = (
                    self.cached_encoder_hidden_states[1][:B] + 
                    (self.cached_encoder_hidden_states[1][:B] - self.cached_encoder_hidden_states[0][:B])
                    * 0.3
                    ).to(norm_hidden_states.device, non_blocking=True)
            else:
                attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    attention_mode=attention_mode,
                )
                if fastercache_counter == fastercache_start_step:
                    self.cached_hidden_states = [attn_hidden_states.to(fastercache_device), attn_hidden_states.to(fastercache_device)]
                    self.cached_encoder_hidden_states = [attn_encoder_hidden_states.to(fastercache_device), attn_encoder_hidden_states.to(fastercache_device)]
                elif fastercache_counter > fastercache_start_step:
                    self.cached_hidden_states[-1].copy_(attn_hidden_states.to(fastercache_device))
                    self.cached_encoder_hidden_states[-1].copy_(attn_encoder_hidden_states.to(fastercache_device))
        else:
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mode=attention_mode,
            )
        
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
       
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

#region Transformer
class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: int = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        is_train_face: bool = False,
        is_kps: bool = False,
        cross_attn_interval: int = 1,
        LFE_num_tokens: int = 32,
        LFE_output_dim: int = 768,
        LFE_heads: int = 12,
        local_face_scale: float = 1.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None

        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(ofs_embed_dim, ofs_embed_dim, timestep_activation_fn) # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

        self.fuser_list = None
        self.use_fastercache = False
        self.fastercache_counter = 0
        self.fastercache_start_step = 15
        self.fastercache_lf_step = 40
        self.fastercache_hf_step = 30
        self.fastercache_device = "cuda"
        self.fastercache_num_blocks_to_cache = len(self.transformer_blocks)
        self.attention_mode = "sdpa"

        if is_train_face:
            self.inner_dim = inner_dim
            self.cross_attn_interval = cross_attn_interval
            self.num_ca = num_layers // cross_attn_interval
            self.LFE_num_tokens = LFE_num_tokens
            self.LFE_output_dim = LFE_output_dim
            self.LFE_heads = LFE_heads
            self.LFE_final_output_dim = int(self.inner_dim / 3 * 2)
            self.local_face_scale = local_face_scale
            self._init_face_inputs()
        

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _init_face_inputs(self):
        device = self.device
        weight_dtype = next(self.transformer_blocks.parameters()).dtype
        self.local_facial_extractor = LocalFacialExtractor()
        self.local_facial_extractor.to(device, dtype=weight_dtype)
        self.perceiver_cross_attention = nn.ModuleList([
            PerceiverCrossAttention(dim=self.inner_dim, dim_head=128, heads=16, kv_dim=self.LFE_final_output_dim).to(device, dtype=weight_dtype) for _ in range(self.num_ca)
        ])
    @classmethod
    def from_pretrained_cus(cls, pretrained_model_path, subfolder=None, config_path=None, transformer_additional_kwargs={}):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, 'config.json')
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(config_path or pretrained_model_path, 'config.json')

        print(f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ...")

        # Check if config file exists
        if not os.path.isfile(config_file):
            raise RuntimeError(f"Configuration file '{config_file}' does not exist")

        # Load the configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if model.state_dict()['patch_embed.proj.weight'].size() != state_dict['patch_embed.proj.weight'].size():
            new_shape   = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).expand(new_shape).clone()
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if model.state_dict()['patch_embed.proj.weight'].size()[1] > state_dict['patch_embed.proj.weight'].size()[1]:
                    model.state_dict()['patch_embed.proj.weight'][:, :state_dict['patch_embed.proj.weight'].size()[1], :, :] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][:, state_dict['patch_embed.proj.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']
                else:
                    model.state_dict()['patch_embed.proj.weight'][:, :, :, :] = state_dict['patch_embed.proj.weight'][:, :model.state_dict()['patch_embed.proj.weight'].size()[1], :, :]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()['patch_embed.proj.weight']

        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        return model

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        controlnet_states: torch.Tensor = None,
        controlnet_weights: Optional[Union[float, int, list, np.ndarray, torch.FloatTensor]] = 1.0,
        video_flow_features: Optional[torch.Tensor] = None,
        consis_id: Optional[dict] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape
   
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None: #1.5 I2V
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        #print("hidden_states before patch_embedding", hidden_states.shape) #torch.Size([2, 4, 16, 60, 90])
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        #print("hidden_states after patch_embedding", hidden_states.shape) #1.5: torch.Size([2, 2926, 3072]) #1.0: torch.Size([2, 5626, 3072])
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        #print("hidden_states after split", hidden_states.shape) #1.5: torch.Size([2, 2700, 3072]) #1.0: torch.Size([2, 5400, 3072])

        # ConsisID: fuse clip and insightface
        if self.is_train_face:
            id_cond = consis_id["id_cond"]
            id_vit_hidden = consis_id["id_vit_hidden"]
            assert id_cond is not None and id_vit_hidden is not None
            valid_face_emb = self.local_facial_extractor(id_cond, id_vit_hidden)  # torch.Size([1, 1280]), list[5](torch.Size([1, 577, 1024]))  ->  torch.Size([1, 32, 2048])
      
        if self.use_fastercache:
            self.fastercache_counter+=1
        if self.fastercache_counter >= self.fastercache_start_step + 3 and self.fastercache_counter % 5 !=0:
            # 3. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states[:1],
                    encoder_hidden_states=encoder_hidden_states[:1],
                    temb=emb[:1],
                    image_rotary_emb=image_rotary_emb,
                    video_flow_feature=video_flow_features[i][:1] if video_flow_features is not None else None,
                    fuser = self.fuser_list[i] if self.fuser_list is not None else None,
                    block_use_fastercache = i <= self.fastercache_num_blocks_to_cache,
                    fastercache_counter = self.fastercache_counter,
                    fastercache_start_step = self.fastercache_start_step,
                    fastercache_device = self.fastercache_device,
                    attention_mode = self.attention_mode
                )

                if (controlnet_states is not None) and (i < len(controlnet_states)):
                    controlnet_states_block = controlnet_states[i]
                    controlnet_block_weight = 1.0
                    if isinstance(controlnet_weights, (list, np.ndarray)) or torch.is_tensor(controlnet_weights):
                        controlnet_block_weight = controlnet_weights[i]
                    elif isinstance(controlnet_weights, (float, int)):
                        controlnet_block_weight = controlnet_weights
                    
                    hidden_states = hidden_states + controlnet_states_block * controlnet_block_weight
                    
            if not self.config.use_rotary_positional_embeddings:
                # CogVideoX-2B
                hidden_states = self.norm_final(hidden_states)
            else:
                # CogVideoX-5B
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                hidden_states = hidden_states[:, text_seq_length:]

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb[:1])
            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            # Note: we use `-1` instead of `channels`:
            #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
            #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
            
            if p_t is None:
                output = hidden_states.reshape(1, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    1, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
            
            (bb, tt, cc, hh, ww) = output.shape
            cond = rearrange(output, "B T C H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            lf_c, hf_c = fft(cond.float())
            #lf_step = 40
            #hf_step = 30
            if self.fastercache_counter <= self.fastercache_lf_step:
                self.delta_lf = self.delta_lf * 1.1
            if self.fastercache_counter >= self.fastercache_hf_step:
                self.delta_hf = self.delta_hf * 1.1
   
            new_hf_uc = self.delta_hf + hf_c
            new_lf_uc = self.delta_lf + lf_c

            combine_uc = new_lf_uc + new_hf_uc
            combined_fft = torch.fft.ifftshift(combine_uc)
            recovered_uncond = torch.fft.ifft2(combined_fft).real
            recovered_uncond = rearrange(recovered_uncond.to(output.dtype), "(B T) C H W -> B T C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            output = torch.cat([output, recovered_uncond])
        else:
            ca_idx = 0
            for i, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    video_flow_feature=video_flow_features[i] if video_flow_features is not None else None,
                    fuser = self.fuser_list[i] if self.fuser_list is not None else None,
                    block_use_fastercache = i <= self.fastercache_num_blocks_to_cache,
                    fastercache_counter = self.fastercache_counter,
                    fastercache_start_step = self.fastercache_start_step,
                    fastercache_device = self.fastercache_device,
                    attention_mode = self.attention_mode
                )
                #has_nan = torch.isnan(hidden_states).any()
                #if has_nan:
                #    raise ValueError(f"block output hidden_states has nan: {has_nan}")

                #controlnet
                if (controlnet_states is not None) and (i < len(controlnet_states)):
                    controlnet_states_block = controlnet_states[i]
                    controlnet_block_weight = 1.0
                    if isinstance(controlnet_weights, (list, np.ndarray)) or torch.is_tensor(controlnet_weights):
                        controlnet_block_weight = controlnet_weights[i]
                        print(controlnet_block_weight)
                    elif isinstance(controlnet_weights, (float, int)):
                        controlnet_block_weight = controlnet_weights                    
                    hidden_states = hidden_states + controlnet_states_block * controlnet_block_weight

                # ConsisID
                if self.is_train_face:
                    if i % self.cross_attn_interval == 0 and valid_face_emb is not None:
                        hidden_states = hidden_states + self.local_face_scale * self.perceiver_cross_attention[ca_idx](valid_face_emb, hidden_states)  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])                        
                        ca_idx += 1
                    
            if not self.config.use_rotary_positional_embeddings:
                # CogVideoX-2B
                hidden_states = self.norm_final(hidden_states)
            else:
                # CogVideoX-5B
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                hidden_states = hidden_states[:, text_seq_length:]

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb)
            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            # Note: we use `-1` instead of `channels`:
            #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
            #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
           
            if p_t is None:
                output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

            if self.fastercache_counter >= self.fastercache_start_step + 1: 
                (bb, tt, cc, hh, ww) = output.shape
                cond = rearrange(output[0:1].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)
                uncond = rearrange(output[1:2].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)

                lf_c, hf_c = fft(cond)
                lf_uc, hf_uc = fft(uncond)

                self.delta_lf = lf_uc - lf_c
                self.delta_hf = hf_uc - hf_c

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

