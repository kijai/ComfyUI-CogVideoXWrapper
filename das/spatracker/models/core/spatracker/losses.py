# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from ..model_utils import reduce_masked_mean
from .blocks import (
        pix2cam
)
from ..model_utils import (
    bilinear_sample2d
)

EPS = 1e-6
import torchvision.transforms.functional as TF

sigma = 3
x_grid = torch.arange(-7,8,1)
y_grid = torch.arange(-7,8,1)
x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
gridxy = torch.stack([x_grid, y_grid], dim=-1).float()
gs_kernel = torch.exp(-torch.sum(gridxy**2, dim=-1)/(2*sigma**2))


def balanced_ce_loss(pred, gt, valid=None):
    total_balanced_loss = 0.0
    for j in range(len(gt)):
        B, S, N = gt[j].shape
        # pred and gt are the same shape
        for (a, b) in zip(pred[j].size(), gt[j].size()):
            assert a == b  # some shape mismatch!
        # if valid is not None:
        for (a, b) in zip(pred[j].size(), valid[j].size()):
            assert a == b  # some shape mismatch!

        pos = (gt[j] > 0.95).float()
        neg = (gt[j] < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred[j]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        pos_loss = reduce_masked_mean(loss, pos * valid[j])
        neg_loss = reduce_masked_mean(loss, neg * valid[j])
        balanced_loss = pos_loss + neg_loss
        total_balanced_loss += balanced_loss / float(N)
    import ipdb; ipdb.set_trace()
    return total_balanced_loss


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8,
                  intr=None, trajs_g_all=None):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0

    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        # assert D == 3
        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(flow_preds[j])
        if intr is not None:
            intr_i = intr[j]
        flow_loss = 0.0  
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i][..., -N:, :D]
            flow_gt_j = flow_gt[j].clone()
            if intr is not None:
                xyz_j_gt = pix2cam(flow_gt_j, intr_i)
            try:
                i_loss = (flow_pred - flow_gt_j).abs()  # B, S, N, 3
            except:
                import ipdb; ipdb.set_trace()
            if D==3:
                i_loss[...,2]*=30
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            flow_loss += i_weight * (reduce_masked_mean(i_loss, valids[j])) 
        
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss / float(N)
        
    
    return total_flow_loss
