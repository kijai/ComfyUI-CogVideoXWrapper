import os
import numpy as np
from os.path import isfile
import torch
import torch.nn.functional as F
EPS = 1e-6
import copy

def sub2ind(height, width, y, x):
    return y*width + x

def ind2sub(height, width, ind):
    y = ind // width
    x = ind % width
    return y, x
    
def get_lr_str(lr):
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    return lrn
    
def strnum(x):
    s = '%g' % x
    if '.' in s:
        if x < 1.0:
            s = s[s.index('.'):]
        s = s[:min(len(s),4)]
    return s

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)

def print_stats(name, tensor):
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)

def print_stats_py(name, tensor):
    shape = tensor.shape
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)

def print_(name, tensor):
    tensor = tensor.detach().cpu().numpy()
    print(name, tensor, tensor.shape)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out

def hard_argmax2d(tensor):
    B, C, Y, X = list(tensor.shape)
    assert(C==1)

    # flatten the Tensor along the height and width axes
    flat_tensor = tensor.reshape(B, -1)
    # argmax of the flat tensor
    argmax = torch.argmax(flat_tensor, dim=1)

    # convert the indices into 2d coordinates
    argmax_y = torch.floor(argmax / X) # row
    argmax_x = argmax % X # col

    argmax_y = argmax_y.reshape(B)
    argmax_x = argmax_x.reshape(B)
    return argmax_y, argmax_x

def argmax2d(heat, hard=True):
    B, C, Y, X = list(heat.shape)
    assert(C==1)

    if hard:
        # hard argmax
        loc_y, loc_x = hard_argmax2d(heat)
        loc_y = loc_y.float()
        loc_x = loc_x.float()
    else:
        heat = heat.reshape(B, Y*X)
        prob = torch.nn.functional.softmax(heat, dim=1)

        grid_y, grid_x = meshgrid2d(B, Y, X)

        grid_y = grid_y.reshape(B, -1)
        grid_x = grid_x.reshape(B, -1)
        
        loc_y = torch.sum(grid_y*prob, dim=1)
        loc_x = torch.sum(grid_x*prob, dim=1)
        # these are B
        
    return loc_y, loc_x

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert(x.size() == mask.size())
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda', on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X
    
    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    # if cuda:
    #     grid_z = grid_z.cuda()
    #     grid_y = grid_y.cuda()
    #     grid_x = grid_x.cuda()
        
    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x
    
def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    
    return grid_z, grid_y, grid_x

def gridcloud2d(B, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy

def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

import re
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data    

def normalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin / float(H)
    ymax = ymax / float(H)
    xmin = xmin / float(W)
    xmax = xmax / float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin * float(H)
    ymax = ymax * float(H)
    xmin = xmin * float(W)
    xmax = xmax * float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_box2d(box2d, H, W):
    return unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def normalize_box2d(box2d, H, W):
    return normalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def get_gaussian_kernel_2d(channels, kernel_size=3, sigma=2.0, mid_one=False):
    C = channels
    xy_grid = gridcloud2d(C, kernel_size, kernel_size) # C x N x 2

    mean = (kernel_size - 1)/2.0
    variance = sigma**2.0

    gaussian_kernel = (1.0/(2.0*np.pi*variance)**1.5) * torch.exp(-torch.sum((xy_grid - mean)**2.0, dim=-1) / (2.0*variance)) # C X N
    gaussian_kernel = gaussian_kernel.view(C, 1, kernel_size, kernel_size) # C x 1 x 3 x 3
    kernel_sum = torch.sum(gaussian_kernel, dim=(2,3), keepdim=True)

    gaussian_kernel = gaussian_kernel / kernel_sum # normalize

    if mid_one:
        # normalize so that the middle element is 1
        maxval = gaussian_kernel[:,:,(kernel_size//2),(kernel_size//2)].reshape(C, 1, 1, 1)
        gaussian_kernel = gaussian_kernel / maxval

    return gaussian_kernel

def gaussian_blur_2d(input, kernel_size=3, sigma=2.0, reflect_pad=False, mid_one=False):
    B, C, Z, X = input.shape
    kernel = get_gaussian_kernel_2d(C, kernel_size, sigma, mid_one=mid_one)
    if reflect_pad:
        pad = (kernel_size - 1)//2
        out = F.pad(input, (pad, pad, pad, pad), mode='reflect')
        out = F.conv2d(out, kernel, padding=0, groups=C)
    else:
        out = F.conv2d(input, kernel, padding=(kernel_size - 1)//2, groups=C)
    return out

def gradient2d(x, absolute=False, square=False, return_sum=False):
    # x should be B x C x H x W
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    zeros = torch.zeros_like(x)
    zero_h = zeros[:, :, 0:1, :]
    zero_w = zeros[:, :, :, 0:1]
    dh = torch.cat([dh, zero_h], axis=2)
    dw = torch.cat([dw, zero_w], axis=3)
    if absolute:
        dh = torch.abs(dh)
        dw = torch.abs(dw)
    if square:
        dh = dh ** 2
        dw = dw ** 2
    if return_sum:
        return dh+dw
    else: 
        return dh, dw
