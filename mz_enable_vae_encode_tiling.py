# thanks to MinusZoneAI: https://github.com/MinusZoneAI/ComfyUI-CogVideoX-MZ/blob/b98b98bd04621e4c85547866c12de2ec723ae98a/mz_enable_vae_encode_tiling.py
from typing import Optional
import torch
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput


@apply_forward_hook
def encode(
    self, x: torch.Tensor, return_dict: bool = True
):
    """
    Encode a batch of images into latents.
    Args:
        x (`torch.Tensor`): Input batch of images.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.
    Returns:
            The latent representations of the encoded videos. If `return_dict` is True, a
            [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
    """
    if self.use_slicing and x.shape[0] > 1:
        encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
        h = torch.cat(encoded_slices)
    else:
        h = self._encode(x)
    posterior = DiagonalGaussianDistribution(h)

    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)


def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
    r"""Encode a batch of images using a tiled encoder.
    When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
    steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
    different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
    tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
    output, but they should be much less noticeable.
    Args:
        x (`torch.Tensor`): Input batch of videos.
    Returns:
        `torch.Tensor`:
            The latent representation of the encoded videos.
    """
    # For a rough memory estimate, take a look at the `tiled_decode` method.
    batch_size, num_channels, num_frames, height, width = x.shape
    overlap_height = int(self.tile_sample_min_height *
                         (1 - self.tile_overlap_factor_height))
    overlap_width = int(self.tile_sample_min_width *
                        (1 - self.tile_overlap_factor_width))
    blend_extent_height = int(
        self.tile_latent_min_height * self.tile_overlap_factor_height)
    blend_extent_width = int(
        self.tile_latent_min_width * self.tile_overlap_factor_width)
    row_limit_height = self.tile_latent_min_height - blend_extent_height
    row_limit_width = self.tile_latent_min_width - blend_extent_width
    frame_batch_size = 4
    # Split x into overlapping tiles and encode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
            num_batches = num_frames // frame_batch_size if num_frames > 1 else 1
            time = []
            for k in range(num_batches):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * k + \
                    (0 if k == 0 else remaining_frames)
                end_frame = frame_batch_size * (k + 1) + remaining_frames
                tile = x[
                    :,
                    :,
                    start_frame:end_frame,
                    i: i + self.tile_sample_min_height,
                    j: j + self.tile_sample_min_width,
                ]
                
                tile = self.encoder(tile)
                if not isinstance(tile, tuple):
                    tile = (tile,)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                time.append(tile[0])
            try:
                self._clear_fake_context_parallel_cache()
            except:
                pass
            row.append(torch.cat(time, dim=2))
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = self.blend_v(
                    rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent_width)
            result_row.append(
                tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))
    enc = torch.cat(result_rows, dim=3)
    return enc


def _encode(
    self, x: torch.Tensor, return_dict: bool = True
):
    batch_size, num_channels, num_frames, height, width = x.shape

    if self.use_encode_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
        return self.tiled_encode(x)

    if num_frames == 1:
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
    else:
        frame_batch_size = 4
        h = []
        for i in range(num_frames // frame_batch_size):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + \
                (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = x[:, :, start_frame:end_frame]
            z_intermediate = self.encoder(z_intermediate)
            if self.quant_conv is not None:
                z_intermediate = self.quant_conv(z_intermediate)
            h.append(z_intermediate)
        try:
            self._clear_fake_context_parallel_cache()
        except:
            pass
        h = torch.cat(h, dim=2)
    return h


def enable_encode_tiling(
    self,
    tile_sample_min_height: Optional[int] = None,
    tile_sample_min_width: Optional[int] = None,
    tile_overlap_factor_height: Optional[float] = None,
    tile_overlap_factor_width: Optional[float] = None,
) -> None:
    r"""
    Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
    compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
    processing larger images.

    Args:
        tile_sample_min_height (`int`, *optional*):
            The minimum height required for a sample to be separated into tiles across the height dimension.
        tile_sample_min_width (`int`, *optional*):
            The minimum width required for a sample to be separated into tiles across the width dimension.
        tile_overlap_factor_height (`int`, *optional*):
            The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
            no tiling artifacts produced across the height dimension. Must be between 0 and 1. Setting a higher
            value might cause more tiles to be processed leading to slow down of the decoding process.
        tile_overlap_factor_width (`int`, *optional*):
            The minimum amount of overlap between two consecutive horizontal tiles. This is to ensure that there
            are no tiling artifacts produced across the width dimension. Must be between 0 and 1. Setting a higher
            value might cause more tiles to be processed leading to slow down of the decoding process.
    """
    self.use_encode_tiling = True
    self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
    self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
    self.tile_latent_min_height = int(
        self.tile_sample_min_height /
        (2 ** (len(self.config.block_out_channels) - 1))
    )
    self.tile_latent_min_width = int(
        self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
    self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
    self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width


from types import MethodType


def enable_vae_encode_tiling(vae):
    vae.encode = MethodType(encode, vae)
    setattr(vae, "_encode", MethodType(_encode, vae))
    setattr(vae, "tiled_encode", MethodType(tiled_encode, vae))
    setattr(vae, "use_encode_tiling", True)
    
    setattr(vae, "enable_encode_tiling", MethodType(enable_encode_tiling, vae))
    vae.enable_encode_tiling()
    return vae
