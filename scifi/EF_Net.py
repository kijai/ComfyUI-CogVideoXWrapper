from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput, CogVideoXBlock
from diffusers.utils import is_torch_version
from diffusers.loaders import  PeftAdapterMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class EF_Net(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        vae_channels: int = 16,
        in_channels: int = 3,
        downscale_coef: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 1,
        patch_size: int = 2,
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
        out_proj_dim = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        out_proj_dim = inner_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
            
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=vae_channels,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=49,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.patch_embed_first = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=vae_channels,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)
        self.weights = nn.ModuleList([nn.Linear(inner_dim, 13) for _ in range(num_layers)])
        self.first_weights = nn.ModuleList([nn.Linear(2*inner_dim, inner_dim) for _ in range(num_layers)])

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

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

        self.out_projectors = None
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        
        if out_proj_dim is not None:
            self.out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, out_proj_dim) for _ in range(num_layers)]
            )
        
        self.gradient_checkpointing = False
        
    def _set_gradient_checkpointing(self, enable=False, gradient_checkpointing_func=None):
        self.gradient_checkpointing = enable

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        EF_Net_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = EF_Net_states.shape
        o_hidden_states = hidden_states
        hidden_states = EF_Net_states
        encoder_hidden_states_ = encoder_hidden_states
        
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        o_hidden_states = self.patch_embed_first(encoder_hidden_states_, o_hidden_states)
        o_hidden_states = self.embedding_dropout(o_hidden_states)

        text_seq_length = encoder_hidden_states_.shape[1]
        o_hidden_states = o_hidden_states[:, text_seq_length:]
        
        EF_Net_hidden_states = ()
        # 2. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            #if self.training and self.gradient_checkpointing:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            
            if self.out_projectors is not None:
                coff = self.weights[i](hidden_states)
                temp_list = []
                for j in range(coff.shape[2]):
                    temp_list.append(hidden_states*coff[:,:,j:(j+1)])
                out = torch.concat(temp_list, dim=1)
                out = torch.concat([out, o_hidden_states], dim=2)
                out = self.first_weights[i](out)
                out = self.relu(out)
                out = self.out_projectors[i](out)
                EF_Net_hidden_states += (out,)
            else:
                out = torch.concat([weight*hidden_states for weight in self.weights], dim=1)
                EF_Net_hidden_states += (out,)
              
        if not return_dict:
            return (EF_Net_hidden_states,)
        return Transformer2DModelOutput(sample=EF_Net_hidden_states)

