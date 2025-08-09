import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FPN(nn.Module):
    """
    Fearure Pyramid Network.

    Parameters:
        out_dim (int): hidden aligned channel dimension for pyramid layers.
        fm_dims (List[int]): channel dimensions of imput feature maps.
        fm_spacials (List[int]): spacial dimensions of input feature maps.
    """

    def __init__(self, out_dim=512, fm_dims=[512, 1024, 2048, 512, 256, 256],
                                fm_spacials=[38, 19, 10, 5, 3, 1]):

        super().__init__()

        self.fm_dims = fm_dims
        self.fm_spacials = fm_spacials

        self.align_convs = nn.ModuleList([
            nn.Conv2d(fm_dims[i], out_dim, kernel_size=1)
            for i in range(len(fm_dims))
        ])

        self.act = nn.ReLU()

        self.pyramid_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
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
        for i in range(n_fmaps - 1, -1, -1):
            if i < n_fmaps - 1:
                x += aligned_fmaps[i + 1]
            x = F.interpolate(x, size=self.fm_spacials[i], mode="bilinear", align_corners=False)
            x = self.act(self.pyramid_convs[i](x))

            output_fmaps.append(x)

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

    def __init__(self, fpn_dim, fm_dims=[512, 1024, 2048, 512, 256, 256],
                                fm_spacials=[38, 19, 10, 5, 3, 1]):
        
        super().__init__()

        self.fm_dims = fm_dims
        self.fm_spacials = fm_spacials

        assert len(fm_dims) == len(fm_spacials)
        self.n_fmaps = len(fm_dims)

        self.fpn_dim = fpn_dim
        self.fpn = FPN(self.fpn_dim, self.fm_dims, self.fm_spacials)

        self.down_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, stride=2, 
                      padding=0 if (i == self.n_fmaps - 2) else 1)
            for i in range(self.n_fmaps)
        ])

        self._init_convs()
    

    def forward(self, input_fmaps):
        output_fmaps = []
        fpn_fmaps = self.fpn(input_fmaps)[::-1]

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