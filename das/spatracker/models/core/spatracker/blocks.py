# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
import collections
from functools import partial
from itertools import repeat
import torchvision.models as tvm

from .vit.encoder import ImageEncoderViT as vitEnc
from .dpt.models import DPTEncoder
from .loftr import LocalFeatureTransformer
# from models.monoD.depth_anything.dpt import DPTHeadEnc, DPTHead

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None,
                  num_heads=8, dim_head=48, qkv_bias=False, flash=False):
        super().__init__()
        inner_dim = self.inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.flash = flash

        self.qkv = nn.Linear(query_dim, inner_dim*3, bias=qkv_bias)
        self.proj = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, _ = x.shape
        C = self.inner_dim
        h = self.heads
        # q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        # k, v = self.to_kv(context).chunk(2, dim=-1)
        # context = default(context, x)

        qkv = self.qkv(x).reshape(B, N1, 3, h, C // h)
        q, k, v = qkv[:,:, 0], qkv[:,:, 1], qkv[:,:, 2]
        N2 = x.shape[1]

        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        q = q.reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        if self.flash==False:
            sim = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                sim = sim + attn_bias
            attn = sim.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        else:
            input_args = [x.half().contiguous() for x in [q, k, v]]
            x = F.scaled_dot_product_attention(*input_args).permute(0,2,1,3).reshape(B,N1,-1)  # type: ignore

        # return self.to_out(x.float())
        return self.proj(x.float())

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(
        self, input_dim=3, output_dim=128, stride=8, norm_fn="batch", dropout=0.0,
        Embed3D=False
    ):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 64

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            if Embed3D:
                self.conv_fuse = nn.Conv2d(64+63, 
                                           self.in_planes, kernel_size=3, padding=1)
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(
                128 + 128 + 96 + 64,
                output_dim * 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                                 nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, feat_PE=None):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            if feat_PE is not None:
                x = self.conv_fuse(torch.cat([x, feat_PE], dim=1))
                a = self.layer1(x)
            else:
                a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            d = F.interpolate(
                d,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x

class VitEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=128, stride=4):
        super(VitEncoder, self).__init__()
        self.vit = vitEnc(img_size=512, 
                     depth=6, num_heads=8, in_chans=input_dim,
                     out_chans=output_dim,embed_dim=384).cuda()
        self.stride = stride
    def forward(self, x):
        T, C, H, W = x.shape
        x_resize = F.interpolate(x.view(-1, C, H, W), size=(512, 512),
                                  mode='bilinear', align_corners=False)
        x_resize = self.vit(x_resize)
        x = F.interpolate(x_resize, size=(H//self.stride, W//self.stride),
                            mode='bilinear', align_corners=False)
        return x

class DPTEnc(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=2):
        super(DPTEnc, self).__init__()
        self.dpt = DPTEncoder()
        self.stride = stride
    def forward(self, x):
        T, C, H, W = x.shape
        x = (x-0.5)/0.5
        x_resize = F.interpolate(x.view(-1, C, H, W), size=(384, 384),
                                  mode='bilinear', align_corners=False)
        x_resize = self.dpt(x_resize)
        x = F.interpolate(x_resize, size=(H//self.stride, W//self.stride),
                            mode='bilinear', align_corners=False)
        return x

# class DPT_DINOv2(nn.Module):
#     def __init__(self, encoder='vits', features=64, out_channels=[48, 96, 192, 384],
#                   use_bn=True, use_clstoken=False, localhub=True, stride=2, enc_only=True):
#         super(DPT_DINOv2, self).__init__()
#         self.stride = stride
#         self.enc_only = enc_only
#         assert encoder in ['vits', 'vitb', 'vitl']
        
#         if localhub:
#             self.pretrained = torch.hub.load('models/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
#         else:
#             self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
#         state_dict = torch.load("models/monoD/zoeDepth/ckpts/dinov2_vits14_pretrain.pth")
#         self.pretrained.load_state_dict(state_dict, strict=True)
#         self.pretrained.requires_grad_(False)
#         dim = self.pretrained.blocks[0].attn.qkv.in_features
#         if enc_only == True:
#             out_channels=[128, 128, 128, 128]
        
#         self.DPThead = DPTHeadEnc(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        
#     def forward(self, x):
#         mean_ = torch.tensor([0.485, 0.456, 0.406], 
#                              device=x.device).view(1, 3, 1, 1)
#         std_ = torch.tensor([0.229, 0.224, 0.225],
#                             device=x.device).view(1, 3, 1, 1)
#         x = (x+1)/2
#         x = (x - mean_)/std_
#         h, w = x.shape[-2:]
#         h_re, w_re = 560, 560
#         x_resize = F.interpolate(x, size=(h_re, w_re),
#                                   mode='bilinear', align_corners=False)
#         with torch.no_grad():
#             features = self.pretrained.get_intermediate_layers(x_resize, 4, return_class_token=True)
#         patch_h, patch_w = h_re // 14, w_re // 14
#         feat = self.DPThead(features, patch_h, patch_w, self.enc_only)
#         feat = F.interpolate(feat, size=(h//self.stride, w//self.stride), mode="bilinear", align_corners=True)

#         return feat


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = True, amp_dtype = torch.float16):
        super().__init__()
        # in case the Internet connection is not stable, please load the DINOv2 locally
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main',
                                          'dinov2_{:}14'.format("vitl"), source='local', pretrained=False)
        
        state_dict = torch.load("models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth")
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)


        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid

class Dinov2(nn.Module):
    def __init__(self, amp = True, amp_dtype = torch.float16):
        super().__init__()
        # in case the Internet connection is not stable, please load the DINOv2 locally
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main',
                                          'dinov2_{:}14'.format("vitl"), source='local', pretrained=False)
        
        state_dict = torch.load("models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth")
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)

        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            self.dinov2_vitl14 = self.dinov2_vitl14.to(self.amp_dtype)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        mean_ = torch.tensor([0.485, 0.456, 0.406], 
                             device=x.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        x = (x+1)/2
        x = (x - mean_)/std_
        h_re, w_re = 560, 560
        x_resize = F.interpolate(x, size=(h_re, w_re),
                                  mode='bilinear', align_corners=True)
        if not upsample:
            with torch.no_grad():
                dinov2_features_16 = self.dinov2_vitl14.forward_features(x_resize.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,h_re//14, w_re//14)
                del dinov2_features_16
        features_16 = F.interpolate(features_16, size=(H//8, W//8), mode="bilinear", align_corners=True)
        return features_16

class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                  flash=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flash=flash

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, flash=flash,
            **block_kwargs
        )        
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0,
                 flash=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)

        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, 
            num_heads=num_heads, qkv_bias=True, **block_kwargs, flash=flash

        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context):
        with autocast():
            x = x + self.cross_attn(
                self.norm1(x), self.norm_context(context)
            )
        x = x + self.mlp(self.norm2(x))
        return x


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4, depths_dnG=None):
        B, S, C, H_prev, W_prev = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H_prev, W_prev

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.depth_pyramid = []
        self.fmaps_pyramid.append(fmaps)
        if depths_dnG is not None:
           self.depth_pyramid.append(depths_dnG) 
        for i in range(self.num_levels - 1):
            if depths_dnG is not None:
                depths_dnG_ = depths_dnG.reshape(B * S, 1, H_prev, W_prev)
                depths_dnG_ = F.avg_pool2d(depths_dnG_, 2, stride=2)
                _, _, H, W = depths_dnG_.shape
                depths_dnG = depths_dnG_.reshape(B, S, 1, H, W)
                self.depth_pyramid.append(depths_dnG)
            fmaps_ = fmaps.reshape(B * S, C, H_prev, W_prev)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            H_prev = H
            W_prev = W
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)
    
    def corr_sample(self, targets, coords, coords_dp=None):
        B, S, N, C = targets.shape
        r = self.radius
        Dim_c = (2*r+1)**2
        assert C == self.C
        assert S == self.S

        out_pyramid = []
        out_pyramid_dp = []
        for i in range(self.num_levels): 
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            fmaps = self.fmaps_pyramid[i]
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B*S, C, H, W)
            if len(self.depth_pyramid)>0:
                depths_dnG_i = self.depth_pyramid[i]
                depths_dnG_i = depths_dnG_i.view(B*S, 1, H, W)
                dnG_sample = bilinear_sampler(depths_dnG_i, coords_lvl.view(B*S,1,N*Dim_c,2))
                dp_corrs = (dnG_sample.view(B*S,N,-1) - coords_dp[0]).abs()/coords_dp[0]
                out_pyramid_dp.append(dp_corrs)
            fmap2s_sample = bilinear_sampler(fmap2s, coords_lvl.view(B*S,1,N*Dim_c,2))
            fmap2s_sample = fmap2s_sample.permute(0, 3, 1, 2) # B*S, N*Dim_c, C, -1
            corrs = torch.matmul(targets.reshape(B*S*N, 1, -1), fmap2s_sample.reshape(B*S*N, Dim_c, -1).permute(0, 2, 1))
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        if len(self.depth_pyramid)>0:
            out_dp = torch.cat(out_pyramid_dp, dim=-1)
            self.fcorrD = out_dp.contiguous().float()
        else:
            self.fcorrD = torch.zeros_like(out).contiguous().float()
        return out.contiguous().float()


class EUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        vq_depth=3,
        add_space_attn=True,
        add_time_attn=True,
        flash=True
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flash = flash
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_size, output_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=True)
        )

        cross_attn_kwargs = {
            "d_model": 384,
            "nhead": 4,
            "layer_names": ['self', 'cross'] * 3,
        }
        self.gnn = LocalFeatureTransformer(cross_attn_kwargs)
        
        # Attention Modules in the temporal dimension         
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash) if add_time_attn else nn.Identity()
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, flash=flash)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_blocks)

        # Placeholder for the rigid transformation
        self.RigidProj = nn.Linear(self.hidden_size, 128, bias=True)
        self.Proj = nn.Linear(self.hidden_size, 128, bias=True)

        self.se3_dec = nn.Linear(384, 3, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, se3_feature):
        """ Updating with Transformer

        Args:
            input_tensor: B, N, T, C
            arap_embed: B, N, T, C
        """
        B, N, T, C = input_tensor.shape
        x = self.input_transform(input_tensor)
        tokens = x
        K = 0
        j = 0
        for i in range(len(self.time_blocks)):
            tokens_time = rearrange(tokens, "b n t c -> (b n) t c", b=B, t=T, n=N+K)
            tokens_time = self.time_blocks[i](tokens_time)
            tokens = rearrange(tokens_time, "(b n) t c -> b n t c ", b=B, t=T, n=N+K)
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_blocks)) == 0
            ):
                tokens_space = rearrange(tokens, "b n t c -> (b t) n c ", b=B, t=T, n=N)
                tokens_space = self.space_blocks[j](tokens_space)
                tokens = rearrange(tokens_space, "(b t) n c -> b n t c  ", b=B, t=T, n=N)
                j += 1

        B, N, S, _ = tokens.shape
        feat0, feat1 = self.gnn(tokens.view(B*N*S, -1)[None,...], se3_feature[None, ...])

        so3 = F.tanh(self.se3_dec(feat0.view(B*N*S, -1)[None,...].view(B, N, S, -1))/100)
        flow = self.flow_head(feat0.view(B,N,S,-1))

        return flow, _, _, feat1, so3


class FusionFormer(nn.Module):
    """ 
    Fuse the feature tracks info with the low rank motion tokens
    """
    def __init__(
        self,
        d_model=64,
        nhead=8,
        attn_iters=4,
        mlp_ratio=4.0,
        flash=False,
        input_dim=35,
        output_dim=384+3,
    ):
        super().__init__()
        self.flash = flash
        self.in_proj = nn.ModuleList(
            [
                nn.Linear(input_dim, d_model)
                for _ in range(2)
            ]
        )
        self.out_proj = nn.Linear(d_model, output_dim, bias=True)
        self.time_blocks = nn.ModuleList(
            [
                CrossAttnBlock(d_model, d_model, nhead, mlp_ratio=mlp_ratio)
                for _ in range(attn_iters)
            ]
        )
        self.space_blocks = nn.ModuleList(
            [
                AttnBlock(d_model, nhead, mlp_ratio=mlp_ratio, flash=self.flash)
                for _ in range(attn_iters)
            ]
        )
    
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.out_proj.weight.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, x, token_cls):
        """ Fuse the feature tracks info with the low rank motion tokens

        Args:
            x: B, S, N, C
            Traj_whole: B T N C
        
        """
        B, S, N, C = x.shape
        _, T, _, _ = token_cls.shape
        x = self.in_proj[0](x)
        token_cls = self.in_proj[1](token_cls)
        token_cls = rearrange(token_cls, 'b t n c -> (b n) t c')

        for i in range(len(self.space_blocks)):
            x = rearrange(x, 'b s n c -> (b n) s c')
            x = self.time_blocks[i](x, token_cls)
            x = self.space_blocks[i](x.permute(1,0,2))
            x = rearrange(x, '(b s) n c -> b s n c', b=B, s=S, n=N)

        x = self.out_proj(x)
        delta_xyz = x[..., :3]
        feat_traj = x[..., 3:]
        return delta_xyz, feat_traj   

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    


def pix2cam(coords, 
            intr):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    coords=coords.detach()
    B, S, N, _, = coords.shape
    xy_src = coords.reshape(B*S*N, 3)
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B*S*N, 3, 3)
    xy_src = torch.cat([xy_src[..., :2], torch.ones_like(xy_src[..., :1])], dim=-1)
    xyz_src = (torch.inverse(intr)@xy_src[...,None])[...,0]
    dp_pred = coords[..., 2]
    xyz_src_ = (xyz_src*(dp_pred.reshape(S*N, 1)))
    xyz_src_ = xyz_src_.reshape(B, S, N, 3)
    return xyz_src_

def cam2pix(coords,
            intr):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    coords=coords.detach()
    B, S, N, _, = coords.shape
    xy_src = coords.reshape(B*S*N, 3).clone()
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B*S*N, 3, 3)
    xy_src = xy_src / (xy_src[..., 2:]+1e-5)
    xyz_src = (intr@xy_src[...,None])[...,0]
    dp_pred = coords[..., 2]
    xyz_src[...,2] *= dp_pred.reshape(S*N) 
    xyz_src = xyz_src.reshape(B, S, N, 3)
    return xyz_src

def edgeMat(traj3d):
    """ 
    Args:
        traj3d: [B, T, N, 3]
    """
    B, T, N, _ = traj3d.shape
    traj3d = traj3d
    traj3d = traj3d.view(B, T, N, 3)
    traj3d = traj3d[..., None, :] - traj3d[..., None, :, :] # B, T, N, N, 3
    edgeMat = traj3d.norm(dim=-1)  # B, T, N, N 
    return edgeMat