import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FPN(nn.Module):
    """
    Fearure Pyramid Network.

    Parameters:
        dim (int): hidden aligned channel dimension for pyramid layers.
        fm_dims (List[int]): channel dimensions of imput feature maps.
        fm_spacials (List[int]): spacial dimensions of input feature maps.
    """

    def __init__(self, dim=256, fm_dims=[1024, 512, 512, 256, 256, 256],
                                fm_spacials=[38, 19, 10, 5, 3, 1]):

        super().__init__()

        self.fm_dims = fm_dims
        self.fm_spacials = fm_spacials
        self.dim = dim
        self.out_dims = [dim] * 6

        self.align_convs = nn.ModuleList([
            nn.Conv2d(fm_dims[i], dim, kernel_size=1)
            for i in range(len(fm_dims))
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            for i in range(len(fm_dims))
        ])

        self._init_convs()

    def forward(self, input_fmaps):
        assert len(input_fmaps) == len(self.fm_dims) == len(self.fm_spacials)

        n_fmaps = len(input_fmaps)
        aligned_fmaps = []
        output_fmaps = []

        for i in range(n_fmaps):
            x = self.align_convs[i](input_fmaps[i])
            aligned_fmaps.append(x)

        x = aligned_fmaps[-1]
        output_fmaps.append(x)
        for i in range(n_fmaps - 1, 0, -1):
            x = F.interpolate(x, size=self.fm_spacials[i - 1], mode="bilinear", align_corners=False)
            x += aligned_fmaps[i - 1]
            output_fmaps.append(x)
        
        output_fmaps = [c(fm) for c, fm in zip(self.smooth_convs, output_fmaps)]

        return output_fmaps[::-1]

    def _init_convs(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class PAN(nn.Module):
    """
    Path Aggregation Network.

    Parameters:
        fpn_dim (int): hidden aligned channel dimension for pyramid layers.
        fm_dims (List[int]): channel dimensions of imput feature maps.
        fm_spacials (List[int]): spacial dimensions of input feature maps.
    """

    def __init__(self, fpn_dim=256, fm_dims=[1024, 512, 512, 256, 256, 256],
                                    fm_spacials=[38, 19, 10, 5, 3, 1]):
        
        super().__init__()

        self.fm_dims = fm_dims
        self.fm_spacials = fm_spacials

        assert len(fm_dims) == len(fm_spacials)
        self.n_fmaps = len(fm_dims)

        self.fpn_dim = fpn_dim
        self.fpn = FPN(self.fpn_dim, self.fm_dims, self.fm_spacials)
        self.out_dims = [fpn_dim] * 6

        self.down_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, stride=2, 
                      padding=0 if (i == self.n_fmaps - 2) else 1)
            for i in range(self.n_fmaps - 1)
        ])

        self._init_convs()
    

    def forward(self, input_fmaps):
        output_fmaps = []
        fpn_fmaps = self.fpn(input_fmaps)

        assert self.n_fmaps == len(fpn_fmaps)

        x = fpn_fmaps[0]
        output_fmaps.append(x)
        for i in range(self.n_fmaps - 1):
            x = self.down_convs[i](x)
            x += fpn_fmaps[i + 1]
            output_fmaps.append(x)
        
        return output_fmaps
    
    
    def _init_convs(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  


class DenseFPN(nn.Module):
    """
    FPN enhancement inspired by DenseNet paradigm.

    Parameters:
        dim (int): hidden aligned channel dimension for pyramid layers.
        fm_dims (List[int]): channel dimensions of imput feature maps.
        fm_spacials (List[int]): spacial dimensions of input feature maps.
    """

    def __init__(self, dim=256, fm_dims=[1024, 512, 512, 256, 256, 256],
                                fm_spacials=[38, 19, 10, 5, 3, 1]):

        super().__init__()

        self.fm_dims = fm_dims[::-1]
        self.fm_spacials = fm_spacials[::-1]
        self.out_dims = [dim] * 6

        self.align_convs = nn.ModuleList([
            nn.Conv2d(self.fm_dims[i], dim, kernel_size=1)
            for i in range(len(self.fm_dims))
        ])

        self.act = nn.ReLU()

        self.dense_convs = nn.ModuleList([
            nn.Conv2d(dim * (i + 1), dim, kernel_size=3, padding=1)
            for i in range(1, len(self.fm_dims))
        ])

        self._init_convs()


    def forward(self, input_fmaps):
        """
        Parameters:
            input_fmaps (list[tensor]): feature maps in high-to-low resolution order, ex. [C3 -> C8]
        Returns:
            (list[tensor]): fused feature maps in high-to-low resolution order, ex. [P3 -> P8]
        """
        assert len(input_fmaps) == len(self.fm_dims) == len(self.fm_spacials)

        input_fmaps = input_fmaps[::-1]
        n_fmaps = len(input_fmaps)
        aligned_fmaps = []
        dense_fmaps = []

        for i in range(n_fmaps):
            x = self.align_convs[i](input_fmaps[i])
            aligned_fmaps.append(x)

        for i in range(n_fmaps):
            if i == 0:
                dense_fmaps.append(aligned_fmaps[i])
            else:
                upsampled_fmaps = [
                    F.interpolate(x, size=self.fm_spacials[i], mode="bilinear", align_corners=False)
                    for x in dense_fmaps
                ]

                concat_fmaps = torch.cat(upsampled_fmaps + [aligned_fmaps[i]], axis=1)
                dense_fmaps.append(self.dense_convs[i - 1](concat_fmaps))    # first dense conv is for aligned_fmap[1]


        return dense_fmaps[::-1]

    def _init_convs(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
