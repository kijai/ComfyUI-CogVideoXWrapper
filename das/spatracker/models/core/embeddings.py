# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    B, S, N, _ = grid.shape
    gridx = grid[..., 0].view(B*S*N).detach().cpu().numpy()
    gridy = grid[..., 1].view(B*S*N).detach().cpu().numpy()
    gridz = grid[..., 2].view(B*S*N).detach().cpu().numpy()

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridx)  # (N, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridy)  # (N, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, gridz)  # (N, D/3)


    emb = np.concatenate([emb_h, emb_w, emb_z], axis=1)  # (N, D)
    emb = torch.from_numpy(emb).to(grid.device)
    return emb.view(B, S, N, embed_dim)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_embedding(xy, C, cat_coords=True):
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # B, N, C*3+3
    return pe


def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert D == 3

    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (
        torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2)  # B, N, C*3+3
    return pe


def get_4d_embedding(xyzw, C, cat_coords=True):
    B, N, D = xyzw.shape
    assert D == 4

    x = xyzw[:, :, 0:1]
    y = xyzw[:, :, 1:2]
    z = xyzw[:, :, 2:3]
    w = xyzw[:, :, 3:4]
    div_term = (
        torch.arange(0, C, 2, device=xyzw.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xyzw.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyzw.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyzw.device, dtype=torch.float32)
    pe_w = torch.zeros(B, N, C, device=xyzw.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe_w[:, :, 0::2] = torch.sin(w * div_term)
    pe_w[:, :, 1::2] = torch.cos(w * div_term)

    pe = torch.cat([pe_x, pe_y, pe_z, pe_w], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyzw], dim=2)  # B, N, C*3+3
    return pe

import torch.nn as nn
class Embedder_Fourier(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder_Fourier, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self,
                input: torch.Tensor,
                rescale: float = 1.0):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)
        out = []
        if self.include_input:
            out.append(input/rescale)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out