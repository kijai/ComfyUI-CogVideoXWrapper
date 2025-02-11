# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
from sklearn.cluster import SpectralClustering
from .blocks import Lie
import matplotlib.pyplot as plt
import cv2

import torch.nn.functional as F
from .blocks import (
    BasicEncoder,
    CorrBlock,
    EUpdateFormer,
    FusionFormer,
    pix2cam,
    cam2pix,
    edgeMat,
    VitEncoder,
    DPTEnc,
    Dinov2
)

from .feature_net import (
    LocalSoftSplat
)

from ..model_utils import (
    meshgrid2d, bilinear_sample2d, smart_cat, sample_features5d, vis_PCA
)
from ..embeddings import (
    get_2d_embedding,
    get_3d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed_from_grid,
    Embedder_Fourier,
)
import numpy as np
from .softsplat import softsplat 

torch.manual_seed(0)
from comfy.utils import ProgressBar
from tqdm import tqdm

def get_points_on_a_grid(grid_size, interp_shape,
                          grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, 
                             interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


def sample_pos_embed(grid_size, embed_dim, coords):
    if coords.shape[-1] == 2:
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim,
                                             grid_size=grid_size)
        pos_embed = (
            torch.from_numpy(pos_embed)
            .reshape(grid_size[0], grid_size[1], embed_dim)
            .float()
            .unsqueeze(0)
            .to(coords.device)
        )
        sampled_pos_embed = bilinear_sample2d(
            pos_embed.permute(0, 3, 1, 2), 
            coords[:, 0, :, 0], coords[:, 0, :, 1]
        )
    elif coords.shape[-1] == 3:
        sampled_pos_embed = get_3d_sincos_pos_embed_from_grid(
            embed_dim, coords[:, :1, ...]
        ).float()[:,0,...].permute(0, 2, 1)

    return sampled_pos_embed


class SpaTracker(nn.Module):
    def __init__(
        self,
        S=8,
        stride=8,
        add_space_attn=True,
        num_heads=8,
        hidden_size=384,
        space_depth=12,
        time_depth=12,
        args=edict({})
    ):
        super(SpaTracker, self).__init__()

        # step1: config the arch of the model
        self.args=args
        # step1.1: config the default value of the model
        if getattr(args, "depth_color", None) == None:
            self.args.depth_color = False
        if getattr(args, "if_ARAP", None) == None:
            self.args.if_ARAP = True
        if getattr(args, "flash_attn", None) == None:
            self.args.flash_attn = True
        if getattr(args, "backbone", None) == None:
            self.args.backbone = "CNN"
        if getattr(args, "Nblock", None) == None:
            self.args.Nblock = 0  
        if getattr(args, "Embed3D", None) == None:
            self.args.Embed3D = True

        # step1.2: config the model parameters
        self.S = S
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = latent_dim = 128
        self.b_latent_dim = self.latent_dim//3
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.lie = Lie()

        # step2: config the model components
        # @Encoder
        self.fnet = BasicEncoder(input_dim=3,
            output_dim=self.latent_dim, norm_fn="instance", dropout=0, 
            stride=stride, Embed3D=False
        )

        # conv head for the tri-plane features
        self.headyz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))
        
        self.headxz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))

        # @UpdateFormer
        self.updateformer = EUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=456, 
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=latent_dim + 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            flash=getattr(self.args, "flash_attn", True)
        )
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1

        self.norm = nn.GroupNorm(1, self.latent_dim)
       
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatyz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatxz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )

        #TODO @NeuralArap: optimize the arap
        self.embed_traj = Embedder_Fourier(
            input_dim=5, max_freq_log2=5.0, N_freqs=3, include_input=True
        )
        self.embed3d = Embedder_Fourier(
            input_dim=3, max_freq_log2=10.0, N_freqs=10, include_input=True
        )
        self.embedConv = nn.Conv2d(self.latent_dim+63,
                            self.latent_dim, 3, padding=1)
        
        # @Vis_predictor
        self.vis_predictor = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.embedProj = nn.Linear(63, 456)
        self.zeroMLPflow = nn.Linear(195, 130)

    def prepare_track(self, rgbds, queries):
        """
        NOTE:
        Normalized the rgbs and sorted the queries via their first appeared time
        Args: 
            rgbds: the input rgbd images (B T 4 H W) 
            queries: the input queries (B N 4)
        Return:
            rgbds: the normalized rgbds (B T 4 H W)
            queries: the sorted queries (B N 4)
            track_mask:         
        """
        assert (rgbds.shape[2]==4) and (queries.shape[2]==4)
        #Step1: normalize the rgbs input
        device = rgbds.device
        rgbds[:, :, :3, ...] = 2 * (rgbds[:, :, :3, ...] / 255.0) - 1.0
        B, T, C, H, W = rgbds.shape
        B, N, __ = queries.shape
        self.traj_e = torch.zeros((B, T, N, 3), device=device)
        self.vis_e = torch.zeros((B, T, N), device=device)

        #Step2: sort the points via their first appeared time
        first_positive_inds = queries[0, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds, dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[sort_inds]
        # check if can be inverse
        assert torch.allclose(
            first_positive_inds, first_positive_inds[sort_inds][inv_sort_inds]
        )

        # filter those points never appear points during 1 - T
        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= 
                      first_positive_inds[None, None, :]).unsqueeze(-1)
        
        # scale the coords_init 
        coords_init = queries[:, :, 1:].reshape(B, 1, N, 3).repeat(
            1, self.S, 1, 1
        ) 
        coords_init[..., :2] /= float(self.stride)

        #Step3: initial the regular grid   
        gridx = torch.linspace(0, W//self.stride - 1, W//self.stride)
        gridy = torch.linspace(0, H//self.stride - 1, H//self.stride)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        gridxy = torch.stack([gridx, gridy], dim=-1).to(rgbds.device).permute(
            2, 1, 0
        )
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        # Step4: initial traj for neural arap
        T_series = torch.linspace(0, 5, T).reshape(1, T, 1 , 1).cuda() # 1 T 1 1
        T_series = T_series.repeat(B, 1, N, 1)
        # get the 3d traj in the camera coordinates
        intr_init = self.intrs[:,queries[0,:,0].long()]
        Traj_series = pix2cam(queries[:,:,None,1:].double(), intr_init.double())
        #torch.inverse(intr_init.double())@queries[:,:,1:,None].double() # B N 3 1
        Traj_series = Traj_series.repeat(1, 1, T, 1).permute(0, 2, 1, 3).float()
        Traj_series = torch.cat([T_series, Traj_series], dim=-1)
        # get the indicator for the neural arap
        Traj_mask = -1e2*torch.ones_like(T_series)
        Traj_series = torch.cat([Traj_series, Traj_mask], dim=-1)

        return (
            rgbds, 
            first_positive_inds, 
            first_positive_sorted_inds,
            sort_inds, inv_sort_inds, 
            track_mask, gridxy, coords_init[..., sort_inds, :].clone(),
            vis_init, Traj_series[..., sort_inds, :].clone()
            )

    def sample_trifeat(self, t, 
                       coords, 
                       featMapxy,
                       featMapyz,
                       featMapxz):
        """
        Sample the features from the 5D triplane feature map 3*(B S C H W)
        Args:
            t: the time index
            coords: the coordinates of the points B S N 3
            featMapxy: the feature map B S C Hx Wy
            featMapyz: the feature map B S C Hy Wz
            featMapxz: the feature map B S C Hx Wz
        """
        # get xy_t yz_t xz_t
        queried_t = t.reshape(1, 1, -1, 1)
        xy_t = torch.cat(
            [queried_t, coords[..., [0,1]]],
            dim=-1
            )
        yz_t = torch.cat(
            [queried_t, coords[..., [1, 2]]],
            dim=-1
            ) 
        xz_t = torch.cat(
            [queried_t, coords[..., [0, 2]]],
            dim=-1
            )
        featxy_init = sample_features5d(featMapxy, xy_t)
    
        featyz_init = sample_features5d(featMapyz, yz_t)
        featxz_init = sample_features5d(featMapxz, xz_t)
        
        featxy_init = featxy_init.repeat(1, self.S, 1, 1)
        featyz_init = featyz_init.repeat(1, self.S, 1, 1)
        featxz_init = featxz_init.repeat(1, self.S, 1, 1)

        return featxy_init, featyz_init, featxz_init

    def neural_arap(self, coords, Traj_arap, intrs_S, T_mark):
        """ calculate the ARAP embedding and offset
        Args:
            coords: the coordinates of the current points   1 S N' 3
            Traj_arap: the trajectory of the points   1 T N' 5
            intrs_S: the camera intrinsics B S 3 3
        
        """
        coords_out = coords.clone()
        coords_out[..., :2] *= float(self.stride)
        coords_out[..., 2] = coords_out[..., 2]/self.Dz
        coords_out[..., 2] = coords_out[..., 2]*(self.d_far-self.d_near) + self.d_near
        intrs_S = intrs_S[:, :, None, ...].repeat(1, 1, coords_out.shape[2], 1, 1)
        B, S, N, D = coords_out.shape
        if S != intrs_S.shape[1]:
            intrs_S = torch.cat(
                [intrs_S, intrs_S[:, -1:].repeat(1, S - intrs_S.shape[1],1,1,1)], dim=1
            )
            T_mark = torch.cat(
                [T_mark, T_mark[:, -1:].repeat(1, S - T_mark.shape[1],1)], dim=1
            )
        xyz_ = pix2cam(coords_out.double(), intrs_S.double()[:,:,0])
        xyz_ = xyz_.float()
        xyz_embed = torch.cat([T_mark[...,None], xyz_,
                               torch.zeros_like(T_mark[...,None])], dim=-1)

        xyz_embed = self.embed_traj(xyz_embed)
        Traj_arap_embed = self.embed_traj(Traj_arap)
        d_xyz,traj_feat = self.arapFormer(xyz_embed, Traj_arap_embed)
        # update in camera coordinate
        xyz_ = xyz_ + d_xyz.clamp(-5, 5)
        # project back to the image plane
        coords_out = cam2pix(xyz_.double(), intrs_S[:,:,0].double()).float()
        # resize back
        coords_out[..., :2] /= float(self.stride)
        coords_out[..., 2] = (coords_out[..., 2] - self.d_near)/(self.d_far-self.d_near)
        coords_out[..., 2] *= self.Dz

        return xyz_, coords_out, traj_feat
    
    def gradient_arap(self, coords, aff_avg=None, aff_std=None, aff_f_sg=None,
                      iter=0, iter_num=4, neigh_idx=None, intr=None, msk_track=None):
        with torch.enable_grad():
            coords.requires_grad_(True)
            y = self.ARAP_ln(coords, aff_f_sg=aff_f_sg, neigh_idx=neigh_idx,
                              iter=iter, iter_num=iter_num, intr=intr,msk_track=msk_track)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=coords,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]

        return gradients.detach()
            
    def forward_iteration(
        self,
        fmapXY,
        fmapYZ,
        fmapXZ,
        coords_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
        intrs_S=None,
    ):
        B, S_init, N, D = coords_init.shape
        assert D == 3
        assert B == 1
        B, S, __, H8, W8 = fmapXY.shape
        device = fmapXY.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)],
                dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            intrs_S = torch.cat(
                [intrs_S, intrs_S[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

        fcorr_fnXY = CorrBlock(
            fmapXY, num_levels=self.corr_levels, radius=self.corr_radius
        )
        fcorr_fnYZ = CorrBlock(
            fmapYZ, num_levels=self.corr_levels, radius=self.corr_radius
        )
        fcorr_fnXZ = CorrBlock(
            fmapXZ, num_levels=self.corr_levels, radius=self.corr_radius
        )
        
        ffeats = torch.split(feat_init.clone(), dim=-1, split_size_or_sections=1)
        ffeats = [f.squeeze(-1) for f in ffeats]

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=456,
            coords=coords[..., :2],
        )
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)
        
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        coord_predictions = []
        attn_predictions = []
        Rot_ln = 0
        support_feat = self.support_features

        comfy_pbar = ProgressBar(iters)

        for __ in tqdm(range(iters), desc="Processing iterations", leave=True):
            coords = coords.detach()
            # if self.args.if_ARAP == True:
            #     # refine the track with arap
            #     xyz_pred, coords, flows_cat0 = self.neural_arap(coords.detach(),
            #                                                    Traj_arap.detach(),
            #                                                    intrs_S, T_mark)
            with torch.no_grad():
                fcorrsXY = fcorr_fnXY.corr_sample(ffeats[0], coords[..., :2])
                fcorrsYZ = fcorr_fnYZ.corr_sample(ffeats[1], coords[..., [1,2]])
                fcorrsXZ = fcorr_fnXZ.corr_sample(ffeats[2], coords[..., [0,2]])
            # fcorrs = fcorrsXY 
            fcorrs = fcorrsXY + fcorrsYZ + fcorrsXZ           
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, 64, cat_coords=True)
            flows_cat =  self.zeroMLPflow(flows_cat)
            

            ffeats_xy = ffeats[0].permute(0, 
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_yz = ffeats[1].permute(0, 
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_xz = ffeats[2].permute(0, 
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_ = ffeats_xy + ffeats_yz + ffeats_xz
            
            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, 2)
            )

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)

            if transformer_input.shape[-1] < pos_embed.shape[-1]:
            # padding the transformer_input to the same dimension as pos_embed
                transformer_input = F.pad(
                    transformer_input, (0, pos_embed.shape[-1] - transformer_input.shape[-1]),
                    "constant", 0
                )

            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta, AttnMap, so3_dist, delta_se3F, so3 = self.updateformer(x, support_feat)
            support_feat = support_feat + delta_se3F[0]/100
            delta = rearrange(delta, " b n t d -> (b n) t d")
            d_coord = delta[:, :, :3]
            d_feats = delta[:, :, 3:]
            
            ffeats_xy = self.ffeat_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_xy.reshape(-1, self.latent_dim)
            ffeats_yz = self.ffeatyz_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_yz.reshape(-1, self.latent_dim)
            ffeats_xz = self.ffeatxz_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_xz.reshape(-1, self.latent_dim)
            ffeats[0] = ffeats_xy.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            ffeats[1] = ffeats_yz.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            ffeats[2] = ffeats_xz.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            coords = coords + d_coord.reshape(B, N, S, 3).permute(0, 2, 1, 3)
            if torch.isnan(coords).any():
                import ipdb; ipdb.set_trace()

            coords_out = coords.clone()
            coords_out[..., :2] *= float(self.stride)
            
            coords_out[..., 2] = coords_out[..., 2]/self.Dz
            coords_out[..., 2] = coords_out[..., 2]*(self.d_far-self.d_near) + self.d_near

            coord_predictions.append(coords_out)
            attn_predictions.append(AttnMap)
            comfy_pbar.update(1)

        ffeats_f = ffeats[0] + ffeats[1] + ffeats[2]
        vis_e = self.vis_predictor(ffeats_f.reshape(B * S * N, self.latent_dim)).reshape(
            B, S, N
        )
        self.support_features = support_feat.detach()
        return coord_predictions, attn_predictions, vis_e, feat_init, Rot_ln


    def forward(self, rgbds, queries, iters=4, feat_init=None,
                is_train=False, intrs=None, wind_S=None):
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1
        self.is_train=is_train
        B, T, C, H, W = rgbds.shape
        # set the intrinsic or simply initialized
        if intrs is None:
            intrs = torch.from_numpy(np.array([[W, 0.0, W//2],
                                              [0.0, W, H//2],
                                              [0.0, 0.0, 1.0]]))
            intrs = intrs[None,
                         None,...].repeat(B, T, 1, 1).float().to(rgbds.device)
        self.intrs = intrs

        # prepare the input for tracking
        (
          rgbds, 
          first_positive_inds, 
          first_positive_sorted_inds, sort_inds, 
          inv_sort_inds, track_mask, gridxy,
          coords_init, vis_init, Traj_arap
          ) = self.prepare_track(rgbds.clone(), queries)
        coords_init_ = coords_init.clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()
        
        depth_all = rgbds[:, :, 3,...]
        d_near = self.d_near = depth_all[depth_all>0.01].min().item()
        d_far = self.d_far = depth_all[depth_all>0.01].max().item()
        
        if wind_S is not None:
            self.S = wind_S

        B, N, __ = queries.shape
        self.Dz = Dz = W//self.stride
        w_idx_start = 0
        p_idx_end = 0
        p_idx_start = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        attn_predictions = []
        p_idx_end_list = []
        Rigid_ln_total = 0
        while w_idx_start < T - self.S // 2:
            curr_wind_points = torch.nonzero(
                first_positive_sorted_inds < w_idx_start + self.S)
            if curr_wind_points.shape[0] == 0:
                w_idx_start = w_idx_start + self.S // 2
                continue
            p_idx_end = curr_wind_points[-1] + 1
            p_idx_end_list.append(p_idx_end)
            # the T may not be divided by self.S
            rgbds_seq = rgbds[:, w_idx_start:w_idx_start + self.S].clone()
            S = S_local = rgbds_seq.shape[1]
            if S < self.S:
                rgbds_seq = torch.cat(
                    [rgbds_seq, 
                     rgbds_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbds_seq.shape[1]
            
            rgbs_ = rgbds_seq.reshape(B * S, C, H, W)[:, :3]
            depths = rgbds_seq.reshape(B * S, C, H, W)[:, 3:].clone()
            # open the mask 
            # Traj_arap[:, w_idx_start:w_idx_start + self.S, :p_idx_end, -1] = 0
            #step1: normalize the depth map

            depths = (depths - d_near)/(d_far-d_near)            
            depths_dn = nn.functional.interpolate(
                    depths, scale_factor=1.0 / self.stride, mode="nearest")
            depths_dnG = depths_dn*Dz
            
            #step2: normalize the coordinate
            coords_init_[:, :, p_idx_start:p_idx_end, 2] = (
                coords_init[:, :, p_idx_start:p_idx_end, 2] - d_near
                )/(d_far-d_near)
            coords_init_[:, :, p_idx_start:p_idx_end, 2] *= Dz

            # efficient triplane splatting 
            gridxyz = torch.cat([gridxy[None,...].repeat(
                                depths_dn.shape[0],1,1,1), depths_dnG], dim=1)
            Fxy2yz = gridxyz[:,[1, 2], ...] - gridxyz[:,:2]
            Fxy2xz = gridxyz[:,[0, 2], ...] - gridxyz[:,:2]
            if getattr(self.args, "Embed3D", None) == True:
                gridxyz_nm = gridxyz.clone()
                gridxyz_nm[:,0,...] = (gridxyz_nm[:,0,...]-gridxyz_nm[:,0,...].min())/(gridxyz_nm[:,0,...].max()-gridxyz_nm[:,0,...].min())
                gridxyz_nm[:,1,...] = (gridxyz_nm[:,1,...]-gridxyz_nm[:,1,...].min())/(gridxyz_nm[:,1,...].max()-gridxyz_nm[:,1,...].min())
                gridxyz_nm[:,2,...] = (gridxyz_nm[:,2,...]-gridxyz_nm[:,2,...].min())/(gridxyz_nm[:,2,...].max()-gridxyz_nm[:,2,...].min())
                gridxyz_nm = 2*(gridxyz_nm-0.5)
                _,_,h4,w4 = gridxyz_nm.shape
                gridxyz_nm = gridxyz_nm.permute(0,2,3,1).reshape(S*h4*w4, 3)
                featPE = self.embed3d(gridxyz_nm).view(S, h4, w4, -1).permute(0,3,1,2)
                if fmaps_ is None:
                    fmaps_ = torch.cat([self.fnet(rgbs_),featPE], dim=1) 
                    fmaps_ = self.embedConv(fmaps_)
                else:
                    fmaps_new = torch.cat([self.fnet(rgbs_[self.S // 2 :]),featPE[self.S // 2 :]], dim=1) 
                    fmaps_new = self.embedConv(fmaps_new)
                    fmaps_ = torch.cat(
                        [fmaps_[self.S // 2 :], fmaps_new], dim=0
                    )
            else:        
                if fmaps_ is None:
                    fmaps_ = self.fnet(rgbs_)
                else:
                    fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                    )

            fmapXY = fmaps_[:, :self.latent_dim].reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )

            fmapYZ = softsplat(fmapXY[0], Fxy2yz, None,
                            strMode="avg", tenoutH=self.Dz, tenoutW=H//self.stride)
            fmapXZ = softsplat(fmapXY[0], Fxy2xz, None,
                                strMode="avg", tenoutH=self.Dz, tenoutW=W//self.stride)

            fmapYZ = self.headyz(fmapYZ)[None, ...]
            fmapXZ = self.headxz(fmapXZ)[None, ...]

            if p_idx_end - p_idx_start > 0:
                queried_t = (first_positive_sorted_inds[p_idx_start:p_idx_end]
                                                        - w_idx_start)
                (featxy_init,
                 featyz_init,
                 featxz_init) = self.sample_trifeat(
                     t=queried_t,featMapxy=fmapXY,
                     featMapyz=fmapYZ,featMapxz=fmapXZ,
                     coords=coords_init_[:, :1, p_idx_start:p_idx_end]
                     )
                # T, S, N, C, 3
                feat_init_curr = torch.stack([featxy_init, 
                                              featyz_init, featxz_init], dim=-1)
                feat_init = smart_cat(feat_init, feat_init_curr, dim=2)
            
            if p_idx_start > 0:
                # preprocess the coordinates of last windows
                last_coords = coords[-1][:, self.S // 2 :].clone()
                last_coords[..., :2] /= float(self.stride)
                last_coords[..., 2:] = (last_coords[..., 2:]-d_near)/(d_far-d_near)
                last_coords[..., 2:] = last_coords[..., 2:]*Dz                        
                
                coords_init_[:, : self.S // 2, :p_idx_start] = last_coords
                coords_init_[:, self.S // 2 :, :p_idx_start] = last_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)
                
                last_vis = vis[:, self.S // 2 :].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :p_idx_start] = last_vis
                vis_init_[:, self.S // 2 :, :p_idx_start] = last_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            coords, attns, vis, __, Rigid_ln = self.forward_iteration(
                fmapXY=fmapXY,
                fmapYZ=fmapYZ,
                fmapXZ=fmapXZ,
                coords_init=coords_init_[:, :, :p_idx_end],
                feat_init=feat_init[:, :, :p_idx_end],
                vis_init=vis_init_[:, :, :p_idx_end],
                track_mask=track_mask[:, w_idx_start : w_idx_start + self.S, :p_idx_end],
                iters=iters,
                intrs_S=self.intrs[:, w_idx_start : w_idx_start + self.S],
                )
            
            Rigid_ln_total+=Rigid_ln
            
            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                attn_predictions.append(attns)

            self.traj_e[:, w_idx_start:w_idx_start+self.S, :p_idx_end] = coords[-1][:, :S_local]
            self.vis_e[:, w_idx_start:w_idx_start+self.S, :p_idx_end] = vis[:, :S_local]

            track_mask[:, : w_idx_start + self.S, :p_idx_end] = 0.0
            w_idx_start = w_idx_start + self.S // 2

            p_idx_start = p_idx_end
        
        self.traj_e = self.traj_e[:, :, inv_sort_inds]
        self.vis_e = self.vis_e[:, :, inv_sort_inds]

        self.vis_e = torch.sigmoid(self.vis_e)
        train_data = (
            (vis_predictions, coord_predictions, attn_predictions,
             p_idx_end_list, sort_inds, Rigid_ln_total)
        )
        if self.is_train:
            return self.traj_e, feat_init, self.vis_e, train_data
        else:
            return self.traj_e, feat_init, self.vis_e

