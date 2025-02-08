import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet, CIFAR10
from torchvision import transforms
from timm.layers import DropPath, trunc_normal_

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from torch import einsum


def window_partition(x, window_size=4):
    """
    Devides patched tensor into windows.

    Parameters:
        x (torch.Tensor): tensor of patches -> (B, H, W, C)
        window_size (int): size of window

    Returns:
        torch.Tensor: tensor of windows -> (n * B, window_size, window_size, C)
    """

    assert len(x.shape) == 4, "suppports only batched tensors"
    assert (x.shape[1] % window_size == 0) and (x.shape[2] % window_size == 0)

    windows = rearrange(x, 'b (h h2) (w w2) c -> (b h w) h2 w2 c', h2=window_size, w2=window_size)

    return windows


def window_merging(x, window_size, h, w):
    """
    Assembles tensor of windows into single patched tensor.

    Parameters:
        x (torch.Tensor): tensor of windows -> (n * B, window_size, window_size, C)
        window_size (int): size of window

    Returns:
        torch.Tensor: tensor of patches -> (B, H, W, C)
    """

    assert len(x.shape) == 4, "suppports only batched tensors"
    assert (x.shape[1] == window_size) and (x.shape[2] == window_size)

    h_ = h // window_size
    w_ = w // window_size

    patches = rearrange(x, '(b h w) h2 w2 c -> b (h h2) (w w2) c', h=h_, w=w_)

    return patches


def build_mask(resolution, winsize, shift):
    """
    Constructs boolean mask for WMSA.

    Parameters:
        resolution (Tuple[int, int]): height and width of input tensor
        winsize (int): size of window
        shift (int): number of patches to move up and left
    """
    h, w = resolution
    mask = torch.zeros((1, h, w, 1))

    h_stamps = (0, -winsize, -shift)
    w_stamps = (0, -winsize, -shift)

    for hs in h_stamps:
        mask[:, hs:, :, :] += 1
    for ws in w_stamps:
        mask[:, :, ws:, :] += 3

    mask = window_partition(mask, window_size=winsize)    # -> (nW, Wh, Ww, 1)
    mask = mask.flatten(start_dim=1)                      # -> (nW, Wh*Ww)
    attn_mask = mask.unsqueeze(1) - mask.unsqueeze(2)     # -> (nW, (Wh*ww)*(Wh*Ww))
    attn_mask = (attn_mask != 0).float() * -100.0

    return attn_mask


# ----------------------------------------------- Basic Computational Blocks ---------------------------------------------------- #


class PatchEmbedding(nn.Module):
    """
    Splits input image into patches -> maps patches to embedding vectors.

    Parameters:
        img_size (int): size of input image (supposed it comes in a square form)
        patch_size (int): length of patch's side in pixels
        emb_dim (int): dimension of aquired patch embeddings
        norm_layer (nn.Module, optional): normalization to apply afterwards
    """

    def __init__(self, img_size=224, patch_size=4, emb_dim=16, norm_layer=None):
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patch_size = patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.emb_dim = emb_dim
        self.norm_layer = norm_layer(emb_dim) if norm_layer is not None else None

    def forward(self, x):
        assert len(x.shape) == 4, "only batched 3d tensors supported"
        assert x.shape[3] == 3, "input tensor has to be in (B, H, W, 3) format"
        assert x.shape[1] % self.patch_size == 0, "tensor size has to be divisible by patch_size"

        permuted_x = x.permute(0, 3, 1, 2)      # -> (B, C, H, W)
        res = self.patch_conv(permuted_x)
        res = res.permute(0, 2, 3, 1)           # -> (B, H, W, C)
        if self.norm_layer is not None:
            res = self.norm_layer(res)

        return res


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.

    Parameters:
        in_features (int): number of input tensor's features
        hid_features (int): number of neurons at hidden layer
        out_features (int): number of neurons at output layer
        act_layer (nn.Module): activation layer
        drop (float): dropout proabbility -> (0, 1)
    """

    def __init__(self, in_features, hid_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hid_features = hid_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hid_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hid_features, out_features)
        self.drop_layer = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop_layer(x)
        x = self.fc2(x)
        x = self.drop_layer(x)

        return x


class WMSA(nn.Module):
    """
    Window-based Multi-head Self Attention.

    Parameters:
        input _dim (int): number of input channels
        window_size (tuple[int]): the height and width of the window
        n_heads (int): Number of attention heads
        qkv_bias (bool, optional):  if True, add a learnable bias to query, key, value
        qk_scale (float, optional): override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): dropout ratio of attention weight
        output_drop (float, optional): dropout ratio of output
    """

    def __init__(self, input_dim, window_size, n_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., output_drop=0.):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.n_heads = n_heads
        assert input_dim % n_heads == 0    # -> to avoid errors in rearranging
        self.head_dim = input_dim // n_heads
        self.qk_scale = qk_scale or self.head_dim ** -0.5

        self.to_qkv = nn.Conv2d(input_dim, self.n_heads * self.head_dim * 3, 1, bias=qkv_bias)
        self.to_out = nn.Conv2d(self.n_heads * self.head_dim, input_dim, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop)

        # define a parameter table of Relative Position Bias
        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads)   # -> (2*Wh-1 * 2*Ww-1, nH)
        )

        # constructing Relative Position Bias index matrix
        # absolute coords:
        x_coords = torch.arange(window_size[0])
        y_coords = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='ij'), axis=0)
        coords_flat = coords.flatten(start_dim=1)
        # relative coords:
        coords_rel = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)
        coords_rel = coords_rel.permute(1, 2, 0).contiguous()
        coords_rel[:, :, 0] += window_size[0] - 1
        coords_rel[:, :, 1] += window_size[1] - 1
        coords_rel[:, :, 0] *= 2 * window_size[1] - 1
        self.relative_position_index = coords_rel.sum(-1)


    def forward(self, x, mask=None):
        x = x.permute(0, 3, 1, 2)               # -> (bW, C, Wh, Ww)
        bW, c, Wh, Ww = x.shape                 # bW = B * nW

        qkv = self.to_qkv(x).chunk(3, dim=1)    # -> 3 x (bW, C, Wh, Ww)

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h d (x y)", h=self.n_heads), qkv
        )
        q = q * self.qk_scale                                # -> (bW, nH, C, Wh*Ww)  for q, k, v

        sim = einsum("b h d i, b h d j -> b h i j", q, k)    # -> (bW, nH, Wh*Ww, Wh*Ww)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        rpb = self.rpb_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1   # -> (Wh*Ww, Wh*Ww, nH)
        )
        rpb = rpb.permute(2, 0, 1).contiguous()           # -> (nH, Wh*Ww, Wh*Ww)

        sim = sim + rpb.unsqueeze(0)                      # rpb -> (1, nH, Wh*Ww, Wh*Ww)
        if mask is not None:
            nW = mask.shape[0]                            # -> n of windows in single image
            mask = mask.to(sim.device)                    # moving mask to compatible device
            sim = rearrange(sim, "(b nW) nH i j -> b nW nH i j", nW=nW)
            sim = sim + mask.unsqueeze(1).unsqueeze(0)    # mask -> (1, nW, 1, Wh*Ww, Wh*Ww)
            attn = sim.softmax(dim=-1)
            attn = rearrange(attn, "b nW nH i j -> (b nW) nH i j")
        else:
            attn = sim.softmax(dim=-1)

        output = einsum("b h i j, b h d j -> b h i d", attn, v)                 # -> (bW, nH, Wh*Ww, C)
        output = rearrange(output, "b h (x y) d -> b (h d) x y", x=Wh, y=Ww)    # -> (bW, C, Wh, Ww)
        output = self.to_out(output)
        output = output.permute(0, 2, 3, 1)                                     # -> (bW, Wh, Ww, C)

        return output


class PatchMerging(nn.Module):
    """
    Patch Merging (downsampling method).

    Parameters:
        input_dim (int): number of channels in input tensor
        norm_layer (nn.Module, optional): normalization to apply before merging
    """

    def __init__(self, input_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.reduction = nn.Conv2d(input_dim, 2 * input_dim, kernel_size=2, stride=2, bias=False)
        self.norm_layer = norm_layer(input_dim)

    def forward(self, x):           # x -> (B, H, W, C)
        x = self.norm_layer(x)
        x = x.permute(0, 3, 1, 2)   # -> (B, C, H, W)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1)   # -> (B, H, W, C)

        return x
        

# ----------------------------------------------- Architectural Blocks ---------------------------------------------------- #

class SwinTransformerBlock(nn.Module):
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

        residual = x
        x = self.norm1(x.view(B, H*W, C))
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x

        # window partition
        x_windows = window_partition(x_shifted, self.window_size)    # -> (bW, Wh, Ww, C)

        # WMSA with shifts
        attn_windows = self.attn(x_windows, mask=self.attn_mask)     # -> (bW, Wh, Ww, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x_shifted = window_merging(attn_windows, self.window_size, H, W)    # -> (B, H, W, C)
            x = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_merging(attn_windows, self.window_size, H, W)    # -> (B, H, W, C)
            x = shifted_x

        x = x + residual
        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C)

        return x


class BasicLayer(nn.Module):
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
            SwinTransformerBlock(
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


class SwinTransformer(nn.Module):
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
        mlp_drop (float, optional): MLP dropout rate
        attn_drop (float, optional): attention dropout rate
        drop_path (float | Tuple[float], optional): stochastic depth rate
        norm_layer (nn.Module, optional): normalization to apply after feature extraction
        patch_norm (nn.Module, optional): normalization for PatchEmbedding layer

    """

    def __init__(self, img_size=224, patch_size=4, input_dim=3, n_classes=1000,
                 emb_dim=96, depths=[2, 2, 6, 2], n_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 mlp_drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.emb_dim = emb_dim
        self.patch_norm = patch_norm
        self.num_features = int(emb_dim * 2 ** (self.n_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedding(patch_size=patch_size, emb_dim=emb_dim,
                                          norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution

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

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def extract_features(self, x):
        x = self.patch_embed(x)                    # -> (B, H, W, C)
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
