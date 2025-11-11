import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange
from timm.layers import trunc_normal_
import math

from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import torchvision

# to see "model.py" module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model import PatchMerging, PatchEmbedding, BasicLayer


class CondPatchEmbedding(nn.Module):
    """
    Same as regular PatchEmbedding - but extended for various input_dim

    Parameters:
        input_dim (int): number of channels in the input tensor. 
        img_size (int): size of input image (supposed it comes in a square form)
        patch_size (int): length of patch's side in pixels
        emb_dim (int): dimension of aquired patch embeddings
        norm_layer (nn.Module, optional): normalization to apply afterwards
    """

    def __init__(self, input_dim=3, img_size=224, patch_size=4, emb_dim=16, norm_layer=None):
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.norm_layer = norm_layer(emb_dim) if norm_layer is not None else None
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

    def forward(self, x):                                    # x -> (B, C, H, W)
        assert len(x.shape) == 4, "only batched 3d tensors supported"
        # assert x.shape[1] == 3, "input tensor has to be in (B, 3, H, W) format"
        assert x.shape[2] % self.patch_size == 0, "tensor size has to be divisible by patch_size"

        res = self.patch_conv(x)                             # x -> (B, C, H, W)
        res = res.permute(0, 2, 3, 1).contiguous()           # -> (B, H, W, C)
        if self.norm_layer is not None:
            res = self.norm_layer(res)

        return res
    

class PatchExpanding(nn.Module):
    """
    Patch Expanding (upsampling method).

    Parameters:
        input_dim (int): number of channels in input tensor
        norm_layer (nn.Module, optional): normalization to apply before merging
    """

    def __init__(self, input_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, 2 * input_dim, bias=False)    # was 4 * input_dim
        self.norm_layer = norm_layer(input_dim)

    def forward(self, x):                        # x -> (B, H, W, C)
        x = self.norm_layer(x)
        b, h, w, c = x.shape
        x = rearrange(x, "b h w c -> (b h w) c")
        x = rearrange(self.projection(x), "(b h w) c -> b h w c", b=b, h=h, w=w)
        x = rearrange(x, "b h w (c h1 w1) -> b (h h1) (w w1) c", h1=2, w1=2)

        return x


class DualUpsample(nn.Module):
    """
    Combines PatchExpanding with interpolation (upsampling method).
    """

    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim // 2

        self.expand = PatchExpanding(in_dim)
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.align_bilinear = nn.Conv2d(in_dim, out_dim, 1)
        self.out_conv = nn.Conv2d(out_dim * 2, out_dim, 1)

    def forward(self, x):
        x1 = self.expand(x).permute(0, 3, 1, 2).contiguous()
        x2 = self.bilinear(x.permute(0, 3, 1, 2).contiguous())
        x_cat = torch.cat([x1, self.align_bilinear(x2)], dim=1)

        return self.out_conv(x_cat).permute(0, 2, 3, 1).contiguous()
    

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims).contiguous()
    


# ------------------------------------------- UNet-shaped diffusion model -------------------------------------------- #

class SwinTransformerDiffusion(nn.Module):
    """
    SWIN Transformer (same model as in ~/model.py) + saving hierarchial feature maps in forward.
    (Also, the classification head is excluded as useless)

    """

    def __init__(self, img_size=224, patch_size=4, input_dim=3, n_classes=1000,
                 emb_dim=96, depths=[2, 2, 6, 2], n_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 pos_drop=0., mlp_drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, time_emb_dim=256,
                 condition_dim=None, weights=None, **kwargs):
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.emb_dim = emb_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.num_features = int(emb_dim * 2 ** (self.n_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.weights = weights
        self.input_dim = input_dim

        self.patch_embed = CondPatchEmbedding(input_dim=input_dim, patch_size=patch_size, emb_dim=emb_dim,
                                              norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        num_patches = self.patch_embed.num_patches

        # absolute position embedding (APE)
        if self.ape:
            self.absolute_pos_emb = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
            trunc_normal_(self.absolute_pos_emb, std=.02)

        self.pos_drop = nn.Dropout(p=pos_drop)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        # basic layers
        self.layers = nn.ModuleList([
            BasicLayer(input_dim=int(emb_dim * 2 ** i),
                       input_resolution=(patches_resolution[0] // (2 ** i),
                                         patches_resolution[1] // (2 ** i)),
                       depth=depths[i],
                       n_heads=n_heads[i],
                       window_size=window_size,
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       mlp_drop=mlp_drop, attn_drop=attn_drop,
                       drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                       norm_layer=norm_layer,
                       downsample=PatchMerging if (i < self.n_layers - 1) else None)
            for i in range(self.n_layers)
        ])

        # # to project time embedding to patch embedding size
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(time_emb_dim, self.emb_dim),
        #     nn.GELU(),
        #     nn.Linear(self.emb_dim, self.emb_dim)
        # )
        self.time_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_emb_dim, l.input_dim),
                nn.GELU(),
                nn.Linear(l.input_dim, l.input_dim)
            )
            for l in self.layers
        ])

        self._init_weights()

    def forward(self, x, t_emb):
        x = self.patch_embed(x)                                    # -> (B, H, W, C)
        out_features = [x,]
        # t_proj = self.time_mlp(t_emb).unsqueeze(1).unsqueeze(1)    # -> (B, 1, 1, C)
        # x = x + t_proj

        if self.ape:
            x = x + self.absolute_pos_emb
        x = self.pos_drop(x)

        for i in range(len(self.layers)):
            t_proj = self.time_mlps[i](t_emb).unsqueeze(1).unsqueeze(1)    # -> (B, 1, 1, C)
            x = x + t_proj
            x = self.layers[i](x)
            out_features.append(x)

        return out_features

    def _init_weights(self):
        # if no pretrained weights - basic initialization
        if self.weights == None:
            for name, m in self.named_modules():
                if "downsampling" in name:
                    for layer in m.children():
                        if isinstance(layer, nn.Conv2d):
                            trunc_normal_(layer.weight, std=.02)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0)
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            if self.input_dim > 3:
                # init first 3 filters of PatchEmbedding conv by pretrained weights, other - Xavier 
                old_patch_conv_w = self.weights.pop('patch_embed.patch_conv.weight')
                old_patch_conv_b = self.weights.pop('patch_embed.patch_conv.bias')

                with torch.no_grad():
                    self.patch_embed.patch_conv.weight[:, :3, ...] = old_patch_conv_w
                    trunc_normal_(self.patch_embed.patch_conv.weight[:, 3:, ...], std=.02)
                    self.patch_embed.patch_conv.bias[:3] = old_patch_conv_b[:3]
                    nn.init.constant_(self.patch_embed.patch_conv.bias[3:], 0)

            self.load_state_dict(self.weights, strict=False)   # strict=False 'cos we may have deleted a key above


class TimeEmbedding(nn.Module):
    """
    Time embedding module.
    Acqures an SPE and passes it throuth an MLP.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def spe(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad if dim is odd
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=1)

        return emb

    def forward(self, t):
        spe_emb = self.spe(t)

        return self.mlp(spe_emb)
    

class UpFirstBasicLayer(nn.Module):
    """
    Basic layer but with applying upsampling before Swin Transformer blocks.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.basic_layer = BasicLayer(**kwargs)
        dim = self.basic_layer.input_dim
        self.patch_expanding = PatchExpanding(dim * 2)
        self.align_conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)

    def forward(self, x, skip):
        x_upsampled = self.patch_expanding(x).permute(0, 3, 1, 2)
        skip = skip.permute(0, 3, 1, 2)
        x_fused = self.align_conv(torch.cat((x_upsampled, skip), axis=1))
        out = self.basic_layer(x_fused.permute(0, 2, 3, 1))

        return out
    

class SwinUNetDiffusion(nn.Module):
    """
    Main diffusuion UNet-shaped model.

    Parameters:
        swin_params (dict): dict with named parameter for underlying Swin Transformer
        mid_depth (int): number of Swin Transformer blocks in the middle stage of UNet
        time_emb_dim (int): embedding dimention to encode timestamp
        condition (bool): whether it's the case of conditional generation
        condition_dim (int): number of channels in condition tensor
      
    """

    def __init__(self, swin_params, out_dim=1, mid_depth=2, time_emb_dim=256,
                 weights=None, condition=False, condition_dim=None):
        super().__init__()

        self.depths = swin_params['depths']
        self.n_layers = len(self.depths)
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.condition = condition

        # if conditioned -> will concat condition tensor along C axis
        if condition and condition_dim:
            swin_params['input_dim'] += condition_dim

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, swin_params['drop_path'], sum(self.depths))]

        # pretrained encoder layers
        self.downs = SwinTransformerDiffusion(**swin_params, time_emb_dim=time_emb_dim, weights=weights)
        patches_resolution = self.downs.patch_embed.patches_resolution

        # middle stage
        self.middle = BasicLayer(
            input_dim=int(swin_params['emb_dim'] * 2 ** 3),
            input_resolution=(patches_resolution[0] // (2 ** ((self.n_layers - 1))),
                              patches_resolution[1] // (2 ** ((self.n_layers - 1)))),
            depth=mid_depth,
            n_heads=swin_params['n_heads'][3],
            window_size=swin_params['window_size'],
            mlp_ratio=swin_params['mlp_ratio'],
            qkv_bias=swin_params['qkv_bias'], qk_scale=swin_params['qk_scale'],
            mlp_drop=swin_params['mlp_drop'], attn_drop=swin_params['attn_drop'],
            drop_path=0.,
            norm_layer=swin_params['norm_layer'],
            downsample=None
        )

        # decoder input parameters
        input_dims = [int(swin_params['emb_dim'] * 2 ** ((self.n_layers - 2 - i))) for i in range(self.n_layers - 1)]
        input_resolutions = [
            (patches_resolution[0] // (2 ** ((self.n_layers - 2 - i))),
            patches_resolution[1] // (2 ** ((self.n_layers - 2 - i))))
            for i in range(self.n_layers - 1)
        ]

        # decoder layers
        self.ups = nn.ModuleList([
            UpFirstBasicLayer(input_dim=input_dims[i],
                            input_resolution=input_resolutions[i],
                            depth=self.depths[i],
                            n_heads=swin_params['n_heads'][i],
                            window_size=swin_params['window_size'],
                            mlp_ratio=swin_params['mlp_ratio'],
                            qkv_bias=swin_params['qkv_bias'], qk_scale=swin_params['qk_scale'],
                            mlp_drop=swin_params['mlp_drop'], attn_drop=swin_params['attn_drop'],
                            drop_path=dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])],
                            norm_layer=swin_params['norm_layer'],
                            downsample=None)
                            # downsample=PatchExpanding)
            for i in range(self.n_layers - 1)
        ])

        # output projection --> shall be smth like inverse to patch_embed
        self.out_proj = nn.Sequential(
            DualUpsample(96),    # -> (112, 112, 48)
            nn.LayerNorm(48),
            nn.GELU(),
            DualUpsample(48),    # -> (224, 224, 24)
            nn.LayerNorm(24),
            nn.GELU(),
            Permute((0, 3, 1, 2)),
            nn.Conv2d(24, out_dim, 3, 1, 1)
        )


    def forward(self, x, t, x_cond=None):       # x -> (B, C, H, W), t -> (B,), cond -> (B, C1, H, W)
        if self.condition:
            # x_cond = default(cond, torch.zeros_like(x))    # to use "0" class by default
            x = torch.cat((x, x_cond), dim=1)

        t_emb = self.time_emb(t)
        features = self.downs(x, t_emb)
        x = self.middle(features[-1])

        for i in range(self.n_layers - 1):
            x = self.ups[i](x, features[-(i + 3)])

        x = self.out_proj(x + features[0])

        return x, features


class MetaConditionedModel(nn.Module):  
    """
    Model to execute class-conditioned generation with SwinUnetDiffusion model.
    """
 
    def __init__(self, swin_params=None, num_classes=10, weights=None, cfg=None, **kwargs):
        super().__init__()

        self.condition_dim = cfg['model']['condition_dim']
        self.image_size = cfg['model']['image_size']
        self.class_emb = nn.Embedding(num_classes, cfg['model']['condition_emb_dim'])
        self.class_mlp = nn.Sequential(
            nn.Linear(cfg['model']['condition_emb_dim'], 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, cfg['model']['condition_dim']),
        )

        self.model = SwinUNetDiffusion(
            swin_params=swin_params,
            weights=weights,
            **kwargs
        )


    def forward(self, x, t, class_labels):

        b_size, c, h, w = x.shape

        class_cond = self.class_emb(class_labels)
        class_cond = self.class_mlp(class_cond)
        # class_cond = self.class_mlp(class_cond).view(-1, self.condition_dim, 16, 16)
        class_cond = class_cond.view(b_size, class_cond.shape[1], 1, 1).expand(
            b_size, class_cond.shape[1], h, w
        )
        # class_cond = F.interpolate(class_cond, size=(self.image_size, self.image_size),
        #                            mode='bilinear', align_corners=False)

        output, _ = self.model(x, t, class_cond)
        return output
