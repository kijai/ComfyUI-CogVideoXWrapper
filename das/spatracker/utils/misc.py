import torch
import numpy as np
import math
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def posemb_sincos_2d_xy(xy, C, temperature=10000, dtype=torch.float32, cat_coords=False):
    device = xy.device
    dtype = xy.dtype
    B, S, D = xy.shape
    assert(D==2)
    x = xy[:,:,0]
    y = xy[:,:,1]
    assert (C % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(C // 4, device=device) / (C // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) 
    pe = pe.reshape(B,S,C).type(dtype)
    if cat_coords:
        pe = torch.cat([pe, xy], dim=2) # B,N,C+2
    return pe

class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        self.items = []
        
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size=1):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = min_size
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/float(len(self.items))
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/float(len(self.items))
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self, with_replacement=True):
        idx = np.random.randint(len(self.items))
        if with_replacement:
            return self.items[idx]
        else:
            return self.items.pop(idx)
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = len(self.items)==self.pool_size
        return full
    
    def empty(self):
        self.items = []
            
    def update(self, items):
        for item in items:
            if len(self.items) < self.pool_size:
                # the pool is not full, so let's add this in
                self.items.append(item)
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
                # add to the back
                self.items.append(item)
        return self.items

def farthest_point_sample(xyz, npoint, include_ends=False, deterministic=False):
    """
    Input:
        xyz: pointcloud data, [B, N, C], where C is probably 3
        npoint: number of samples
    Return:
        inds: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.float()
    inds = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if deterministic:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        if include_ends:
            if i==0:
                farthest = 0
            elif i==1:
                farthest = N-1
        inds[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

        if npoint > N:
            # if we need more samples, make them random
            distance += torch.randn_like(distance)
    return inds

def farthest_point_sample_py(xyz, npoint):
    N,C = xyz.shape
    inds = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1,C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        if npoint > N:
            # if we need more samples, make them random
            distance += np.random.randn(*distance.shape)
    return inds
    
