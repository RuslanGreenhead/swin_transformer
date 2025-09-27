import os
import numpy as np
import cv2
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import albumentations as A


class COCODetection(Dataset):
    """
    COCO dataset class.

    Parameters:
        img_dir (str): path to directory with images
        ann_file (str): path to .json with annotations
        transform (albumentations transfom): transforms for images and boxes
    """
    
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file) as f:
            self.coco = json.load(f)
        self.imgs = self.coco['images']
        self.anns = self.coco['annotations']
        self.categories = self.coco['categories']

        self.img_id_to_anns = {}
        for ann in self.anns:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        # mapping original labels to a continuous range of indexes
        # + storing labels' names
        self.label_map = {}
        self.label_names = {}
        cnt = 0
        self.label_names[cnt] = "background"
        for cat in self.categories:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_names[cnt] = cat["name"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        anns = self.img_id_to_anns.get(img_info['id'], [])
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [self.label_map[ann['category_id']] for ann in anns]    # map to index

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        target = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(category_ids, dtype=torch.int64)
        }

        return image, target


def collate_coco(batch):
    images = [item[0] for item in batch]
    images = torch.stack(images)

    bboxes = [item[1]['bboxes'] for item in batch]
    labels = [item[1]['labels'] for item in batch]

    return images, bboxes, labels


def build_loaders(cfg):
    train_set, _ = build_dataset(is_train=True, cfg=cfg['dataset'])
    val_set, _ = build_dataset(is_train=False, cfg=cfg['dataset'])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, shuffle=cfg['dataset']['test_shuffle']
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_coco
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_coco
    )

    return train_loader, val_loader


def build_dataset(cfg, is_train=True):
    transform = build_transform(cfg, is_train)    # ... compose transforms
    if is_train:
        dataset = COCODetection(img_dir=cfg['root'] + "/train2017/train2017",
                                ann_file=cfg["root"] + "/annotations_trainval2017/annotations/instances_train2017.json",
                                transform=transform)
    else:
        dataset = COCODetection(img_dir=cfg['root'] + "/val2017/val2017",
                                ann_file=cfg["root"] + "/annotations_trainval2017/annotations/instances_val2017.json",
                                transform=transform)

    num_classes = 80    # here we imply only ACTUAL classes (w.o. "background")

    return dataset, num_classes


def build_transform(cfg, train=True, pattern=1):
    img_size = cfg['image_size']

    if train:
        # enhanced pattern (potentially requires longer training)
        if pattern == 0:

            transform = A.Compose([
                # photometric transforms
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.ToGray(p=0.05),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.ChannelShuffle(p=0.1),

                # blur and noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ], p=0.4),

                # geometric transforms
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=10, border_mode=0, value=(104, 117, 123), p=0.7),
                A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=0, pad_val=(104, 117, 123), p=0.3),
                
                # padding & cropping & flipping
                A.PadIfNeeded(min_height=int(img_size * 1.5), min_width=int(img_size * 1.5),
                            border_mode=0, value=(104, 117, 123), p=0.5),
                A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, p=0.7),
                A.HorizontalFlip(p=0.5),

                # cutout for occlusion robustness
                A.CoarseDropout(max_holes=5, max_height=int(img_size*0.1), max_width=int(img_size*0.1), 
                                min_holes=1, min_height=int(img_size*0.05), min_width=int(img_size*0.05),
                                fill_value=(104, 117, 123), p=0.5),

                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format='coco', clip=True, min_area=1, min_visibility=0.2, label_fields=['category_ids'])
            )

        # baseline pattern 
        elif pattern == 1:
    
            transform = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                    A.ToGray(p=0.05),
                ], p=1.0),
                A.PadIfNeeded(min_height=int(img_size * 1.5), min_width=int(img_size * 1.5),
                            border_mode=0, value=(104, 117, 123), p=0.5),
                A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, p=0.7),
                A.HorizontalFlip(p=0.5),

                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format='coco', clip=True, min_area=1, min_visibility=0.2, label_fields=['category_ids']))

        # no-augmentation pattern
        elif pattern == 2:
        
            transform = A.Compose([
                A.OneOf([
                    A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.7),
                    A.Resize(img_size, img_size, p=0.3),
                ], p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            bbox_params=A.BboxParams(format='coco', clip=True, min_area=1, min_visibility=0.2, label_fields=['category_ids'])
            ) 

    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params=A.BboxParams(format='coco', clip=True, min_area=1, min_visibility=0.0, label_fields=['category_ids'])
        )
    
    return transform
