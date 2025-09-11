import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, CIFAR10
from torchvision import transforms
from timm.layers import DropPath, trunc_normal_

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from torch import einsum

from ..model import window_partition, window_merging, build_mask
from ..model import PatchEmbedding, MLP, WMSA, PatchMerging, SwinTransformer


# ------------------------------------------------------ ResNet backbones ------------------------------------------------------ #

class ResNet50Backbone(torch.nn.Module):
    def __init__(self, img_size=224, weights="IMAGENET_V2"):
        super().__init__()

        # load pure ResNet50
        base_model = resnet50(weights=None)
        # load pretrained weights
        if weights == "IMAGENET_V2":
            state_dict = torch.load("../saved_weights/resnet50_statedict.pth", weights_only=True)
            base_model.load_state_dict(state_dict)

        # channel dims of output feature maps (for convenience in SSD)
        self.out_dims = [512, 1024, 2048, 512, 256, 256]

        # copy base layers (except for fc head)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2  
        self.layer3 = base_model.layer3 
        self.layer4 = base_model.layer4

        self.aux1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.aux2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aux3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1 if img_size == 224 else 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self._init_auxilary_layers()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)    # (224x224) -> 512x28x28      (300x300) -> 512x38x38
        c4 = self.layer3(c3)    # (224x224) -> 1024x14x14     (300x300) -> 1024x19x19
        c5 = self.layer4(c4)    # (224x224) -> 2048x7x7       (300x300) -> 2048x10x10
    
        c6 = self.aux1(c5)      # (224x224) -> 512x4x4        (300x300) -> 512x5x5
        c7 = self.aux2(c6)      # (224x224) -> 256x2x2        (300x300) -> 256x3x3
        c8 = self.aux3(c7)      # (224x224) -> 256x1x1        (300x300) -> 256x1x1

        return c3, c4, c5, c6, c7, c8
    

    def _init_auxilary_layers(self):
        for name, module in self.named_modules():
            if "aux" in name:
                for m in module.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


class ResNet50Backbone_Deeper(torch.nn.Module):
    def __init__(self, img_size=300, weights="IMAGENET_V2"):
        super().__init__()

        if img_size != 300:
            raise NotImplementedError("Tailored only to 300x300 input.")

        # load pure ResNet50
        base_model = resnet50(weights=None)
        # load pretrained weights
        if weights == "IMAGENET_V2":
            state_dict = torch.load("../saved_weights/resnet50_statedict.pth", weights_only=True)
            base_model.load_state_dict(state_dict)

        # channel dims of output feature maps (for convenience in SSD)
        self.out_dims = [1024, 512, 512, 256, 256, 256]

        # copy base layers (except for fc head)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2  
        self.layer3 = base_model.layer3 

        # change stride to 1 to avoid downsampling
        self.layer3[0].conv1.stride = (1, 1)
        self.layer3[0].conv2.stride = (1, 1)
        self.layer3[0].downsample[0].stride = (1, 1)        

        self.aux1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.aux2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.aux3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aux4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aux5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self._init_auxilary_layers()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)     
        c4 = self.layer3(c3)     # (300x300) -> 1024x38x38
    
        c5 = self.aux1(c4)       # (300x300) -> 512x19x19
        c6 = self.aux2(c5)       # (300x300) -> 512x10x10
        c7 = self.aux3(c6)       # (300x300) -> 256x5x5
        c8 = self.aux4(c7)       # (300x300) -> 256x3x3
        c9 = self.aux5(c8)       # (300x300) -> 256x1x1

        return c4, c5, c6, c7, c8, c9
    

    def _init_auxilary_layers(self):
        for name, module in self.named_modules():
            if "aux" in name:
                for m in module.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = ResNet50Backbone()
    features = model(torch.randn(1, 3, 224, 224))

    print([k.size() for k in features])


# -------------------------------------------- Swin adapted structural blocks -------------------------------------------- #
# (Adaptation of the architecture for 336x336 input size - closest of all the fitting ones to 300x300)

class SwinTransformerBlock_336(nn.Module):
    """
    Swin Transformer Block.
    (Applies normalization, WMSA and MLP)

    Parameters:
        input_dim (int): number of channels in input tensor
        input_resolution (Tuple[int, int]): input resulotion
        n_heads (int): number of attentions heads
        window_size (int): window size
        shift_size (int): shift size for shifting in WMSA
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): if True, add a learnable bias to query, key, value
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set
        mlp_drop (float, optional): dropout rate
        attn_drop (float, optional): attention dropout rate
        drop_path (float, optional): stochastic depth rate
        act_layer (nn.Module, optional): activation layer
        norm_layer (nn.Module, optional): normalization layer
    """

    def __init__(self, input_dim, input_resolution, n_heads,
                 window_size=7, shift_size=2, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, mlp_drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.input_resolution = input_resolution
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # if window is larger than the image - we work with only one window and do no shifts
        if min(self.input_resolution) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # when resolution is 10x10 - we pad it to 14x14
        if self.input_resolution == (10, 10):
            self.input_resolution = (14, 14)

        self.norm1 = norm_layer(input_dim)
        self.attn = WMSA(input_dim=input_dim, window_size=(window_size, window_size),
                         n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, output_drop=mlp_drop)
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = MLP(in_features=input_dim, hid_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=mlp_drop)


        # building mask
        if self.shift_size > 0:
            self.attn_mask = build_mask(resolution=self.input_resolution,
                                        winsize=self.window_size,
                                        shift=self.shift_size)
        else:
            self.attn_mask = None

        self.drop_path = DropPath(drop_path)

    def forward(self, x):    # x -> (B, H, W, C)
        B, H, W, C = x.shape

        if self.input_resolution == 14:
            x = F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0)

        residual = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x

        # window partition
        x_windows = window_partition(x_shifted, self.window_size)    # -> (bW, Wh, Ww, C)

        # WMSA with shifts
        attn_windows = self.attn(x_windows, mask=self.attn_mask)     # -> (bW, Wh, Ww, C)
        attn_windows = attn_windows.contiguous()

        # reverse cyclic shift
        if self.shift_size > 0:
            x_shifted = window_merging(attn_windows, self.window_size, H, W)    # -> (B, H, W, C)
            x = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_shifted = window_merging(attn_windows, self.window_size, H, W)    # -> (B, H, W, C)
            x = x_shifted

        x = x + residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer_336(nn.Module):
    """
    Basic Swin Transformer layer.
    (Stacks several Swin Blocks and puts the downsampler afterwords)

    Parameters:
        input_dim (int): number of input channels
        input_resolution (Tuple[int, int]): input resolution
        depth (int): number of blocks
        n_heads (int): number of attention heads
        window_size (int): local window size
        mlp_ratio (float): ratio of MLP hidden dim to embedding dim
        qkv_bias (bool, optional): if True, add a learnable bias to query, key, value
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set
        mlp_drop (float, optional): MLP dropout rate
        attn_drop (float, optional): attention dropout rate
        drop_path (float | Tuple[float], optional): stochastic depth rate
        norm_layer (nn.Module, optional): normalization layer
        downsample (nn.Module, optional): downsample layer at the end of the layer
    """

    def __init__(self, input_dim, input_resolution, depth, n_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, mlp_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.input_dim = input_dim
        self.input_resolution = input_resolution
        self.depth = depth

        # SWIN blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_336(
                input_dim=input_dim, input_resolution=input_resolution,
                n_heads=n_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                mlp_drop=mlp_drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # downsampling
        if downsample is not None:
            self.downsample = downsample(input_dim=input_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)                     # -> (B, H, W, C)
        if self.downsample is not None:
            x = self.downsample(x)           # -> (B, H, W, C)

        return x


class SwinTransformer_336(nn.Module):
    """
    SWIN Transformer.
    (Everything together)

    Parameters:
        img_size (int): image size
        patch_size (int): patch size
        input_dim (int): number of channels in input image
        n_classes (int): number of classes for classification
        emb_dim (int): embedding dimension
        depths (List[int]): number of Swin Blocks in each Basic Layer
        n_heads (List[int]): number of attn heads in each Basic Layer
        window_size (int): window size
        mlp_ratio (float): ratio of MLP hidden dim to embedding dim
        qkv_bias (bool, optional): if True, add a learnable bias to query, key, value
        qk_scale (float, optional): override default qk scale of head_dim ** -0.5 if set
        pos_drop (float, optional): dropout rate for APE
        mlp_drop (float, optional): MLP dropout rate
        attn_drop (float, optional): attention dropout rate
        drop_path (float | Tuple[float], optional): stochastic depth rate
        norm_layer (nn.Module, optional): normalization to apply after feature extraction
        ape (bool, optional): whether to use APE
        patch_norm (nn.Module, optional): normalization for PatchEmbedding layer

    """

    def __init__(self, img_size=224, patch_size=4, input_dim=3, n_classes=1000,
                 emb_dim=96, depths=[2, 2, 6, 2], n_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 pos_drop=0., mlp_drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.emb_dim = emb_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.num_features = int(emb_dim * 2 ** (self.n_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, emb_dim=emb_dim,
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
            BasicLayer_336(input_dim=int(emb_dim * 2 ** i),
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

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, n_classes) if n_classes > 0 else nn.Identity()

        self._init_weights()

    def extract_features(self, x):
        x = self.patch_embed(x)                    # -> (B, H, W, C)

        if self.ape:
            x = x + self.absolute_pos_emb
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                           # -> (B, H, W, C)
        x = self.avgpool(x.permute(0, 3, 1, 2))    # -> (B, C, 1, 1)
        x = x.squeeze()                            # -> (B, C)

        return x

    def forward(self, x):
        x = self.extract_features(x)               # -> (B, C)
        x = self.head(x)                           # -> (B, head_output_dim)

        return x

    def _init_weights(self):
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

# ---------------------------------------------------- Swin backbone ----------------------------------------------------- #

class SwinTBackbone(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # Load pretrained torchvision ResNet50
        base_model = SwinTransformer_336(img_size=336)
        if weights:
            base_model.load_state_dict(weights)


        # copy base layers (except for fc head)
        self.patch_embed = base_model.patch_embed
        self.swin_block_0 = base_model.layers[0]
        self.swin_block_1 = base_model.layers[1]
        self.swin_block_2 = base_model.layers[2]

        # channel dims of output feature maps (for convenience in SSD)
        self.out_dims = [192, 384, 768, 512, 256, 256]

        self.aux1 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # output: 5x5x512 (approximate due to stride 2)

        self.aux2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # output: 3x3x256

        self.aux3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True)
        )  # output: 1x1x256

    def forward(self, x):
        out_features = []

        x = self.patch_embed(x)
        x = self.swin_block_0(x)
        out_features.append(x.permute(0, 3, 1, 2).contiguous())
        x = self.swin_block_1(x)
        out_features.append(x.permute(0, 3, 1, 2).contiguous())
        x = self.swin_block_2(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        
        out_features.append(x)
        x = self.aux1(x)
        out_features.append(x)
        x = self.aux2(x)
        out_features.append(x)
        x = self.aux3(x)
        out_features.append(x)

        return out_features


if __name__ == "__main__":

    # Example usage
    model = SwinTBackbone(weights=None)
    features = model(torch.randn(1, 3, 336, 336))

    print([k.size() for k in features])