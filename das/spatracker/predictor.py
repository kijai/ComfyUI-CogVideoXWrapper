# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import time

from tqdm import tqdm
from .models.core.spatracker.spatracker import get_points_on_a_grid
from .models.core.model_utils import smart_cat
from .models.build_spatracker import (
    build_spatracker,
)
from .models.core.model_utils import (
    meshgrid2d, bilinear_sample2d, smart_cat
)

from comfy.utils import ProgressBar

class SpaTrackerPredictor(torch.nn.Module):
    def __init__(
        self, checkpoint="cotracker/checkpoints/cotracker_stride_4_wind_8.pth",
        interp_shape=(384, 512),
        seq_length=16
    ):
        super().__init__()
        self.interp_shape = interp_shape
        self.support_grid_size = 6
        model = build_spatracker(checkpoint, seq_length=seq_length)

        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,  # (1, T, 3, H, W)
        video_depth = None, # (T, 1, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (1, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
        depth_predictor=None,
        wind_length: int = 8,
        progressive_tracking: bool = False,
    ):
        if queries is None and grid_size == 0:
            tracks, visibilities, T_Firsts = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                video_depth=video_depth,
                depth_predictor=depth_predictor,
                wind_length=wind_length,
            )
        else:
            tracks, visibilities, T_Firsts = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=False, #(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                video_depth=video_depth,
                depth_predictor=depth_predictor,
                wind_length=wind_length,
            )
        
        return tracks, visibilities, T_Firsts

    def _compute_dense_tracks(
        self, video, grid_query_frame, grid_size=30, backward_tracking=False,
        depth_predictor=None, video_depth=None, wind_length=8
    ):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = T_Firsts = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        for offset in tqdm(range(grid_step * grid_step)):
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = (
                torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            )
            grid_pts[0, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step, T_First_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
                wind_length=wind_length,
                video_depth=video_depth,
                depth_predictor=depth_predictor,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)
            T_Firsts = smart_cat(T_Firsts, T_First_step, dim=1)


        return tracks, visibilities, T_Firsts

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
        depth_predictor=None,
        video_depth=None,
        wind_length=8,
    ):
        B, T, C, H, W = video.shape
        assert B == 1

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear")
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            queries = queries.clone()
            B, N, D = queries.shape
            assert D == 3
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts_extra = grid_pts[:, point_mask]
            else:
                grid_pts_extra = None
            if grid_pts_extra is not None:
                total_num = int(grid_pts_extra.shape[1])
                total_num = min(800, total_num)
                pick_idx = torch.randperm(grid_pts_extra.shape[1])[:total_num]
                grid_pts_extra = grid_pts_extra[:, pick_idx]
                queries_extra = torch.cat(
                    [
                        torch.ones_like(grid_pts_extra[:, :, :1]) * grid_query_frame,
                        grid_pts_extra,
                    ],
                    dim=2,
                )

            queries = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts],
                dim=2,
            )

        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            queries = torch.cat([queries, grid_pts], dim=1)

        ## ----------- estimate the video depth -----------##
        if video_depth is None:
            with torch.no_grad():
                if video[0].shape[0]>30:
                    vidDepths = [] 
                    for i in range(video[0].shape[0]//30+1):
                        if (i+1)*30 > video[0].shape[0]:
                            end_idx = video[0].shape[0]
                        else:
                            end_idx = (i+1)*30
                        if end_idx == i*30:
                            break
                        video_ = video[0][i*30:end_idx]
                        vidDepths.append(depth_predictor.infer(video_/255))

                    video_depth = torch.cat(vidDepths, dim=0)

                else:
                    video_depth = depth_predictor.infer(video[0]/255)
        video_depth = F.interpolate(video_depth,
                                     tuple(self.interp_shape), mode="nearest")
        
        # from PIL import Image
        # import numpy
        # depth_frame = video_depth[0].detach().cpu()
        # depth_frame = depth_frame.squeeze(0)
        # print(depth_frame)
        # print(depth_frame.min(), depth_frame.max())
        # depth_img = (depth_frame * 255).numpy().astype(numpy.uint8)
        # depth_img = Image.fromarray(depth_img, mode='L')
        # depth_img.save('outputs/depth_map.png')

        # frame = video[0, 0].detach().cpu()
        # frame = frame.permute(1, 2, 0)
        # frame = (frame * 255).numpy().astype(numpy.uint8)
        # frame = Image.fromarray(frame, mode='RGB')
        # frame.save('outputs/frame.png')

        depths = video_depth
        rgbds = torch.cat([video, depths[None,...]], dim=2)
        # get the 3D queries
        comfy_pbar = ProgressBar(queries.shape[1])
        depth_interp=[]
        for i in tqdm(range(queries.shape[1]), desc="Processing queries"):
            depth_interp_i = bilinear_sample2d(video_depth[queries[:, i:i+1, 0].long()], 
                                queries[:, i:i+1, 1], queries[:, i:i+1, 2])
            depth_interp.append(depth_interp_i)
            comfy_pbar.update(1)

        depth_interp = torch.cat(depth_interp, dim=1)
        queries = smart_cat(queries, depth_interp,dim=-1)

        #NOTE: free the memory of depth_predictor
        del depth_predictor
        torch.cuda.empty_cache()
        t0 = time.time()
        tracks, __, visibilities = self.model(rgbds=rgbds, queries=queries, iters=6, wind_S=wind_length)
        print("Time taken for inference: ", time.time()-t0)

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                rgbds, queries, tracks, visibilities
            )
            if add_support_grid:
                queries[:, -self.support_grid_size ** 2 :, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size ** 2]
            visibilities = visibilities[:, :, : -self.support_grid_size ** 2]
        thr = 0.9
        visibilities = visibilities > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        
        for i in tqdm(range(len(queries)), desc="Processing queries", leave=False):
            queries_t = queries[i, :tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True
            
        T_First = queries[..., :tracks.size(2), 0].to(torch.uint8)
        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        return tracks, visibilities, T_First

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, __, inv_visibilities = self.model(
            rgbds=inv_video, queries=queries, iters=6
        )

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)

        mask = tracks == 0

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities