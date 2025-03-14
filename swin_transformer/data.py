import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup  # donnow 'bout muxup yet... 
from timm.data import create_transform
from timm.data.transforms import _pil_interp


def build_loaders(cfg):
    train_set = build_dataset(is_train=True, cfg=cfg['data'])
    val_set = build_dataset(is_train=False, cfg=cfg['data'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    
    train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'],
                              shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'],
                            shuffle=False, sampler=val_sampler)

    return train_loader, val_loader


def build_dataset(cfg, is_train=True):
    transform = build_transform(cfg, is_train)    # ... compose transforms
    if is_train:
        dataset = ImageFolder(root=cfg['root'] + "/train", transform=transform)
    else:
        dataset = ImageFolder(root=cfg['root'] + "/val", transform=transform)

    num_classes = 1000

    return dataset, num_classes


def build_transform(cfg, is_train=True):
    # train transforms
    if is_train:
        transform = create_transform(
            input_size=cfg["img_size"],
            is_training=True,
            color_jitter=cfg["aug"]['color_jitter'] if cfg["aug"]['color_jitter'] > 0 else None,
            auto_augment=cfg["aug"]['auto_augment'] if cfg["aug"]['auto_augment'] != 'none' else None,
            re_prob=cfg["aug"]['reprob'],
            re_mode=cfg["aug"]['remode'],
            re_count=cfg["aug"]['recount'],
            interpolation=cfg['interpolation'],
        )
        
        return transform

    # test transforms
    t = []
    
    if cfg['test_crop']:
        size = int((256 / 224) * cfg['img_size'])
        t.append(transforms.Resize(size, interpolation=_pil_interp(cfg['interpolation'])))
        t.append(transforms.CenterCrop(cfg['img_size']))
    else:
        t.append(transforms.Resize((cfg['img_size'], cfg['img_size']),
                                    interpolation=_pil_interp(cfg['interpolation'])))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)