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
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file) as f:
            self.coco = json.load(f)
        self.imgs = self.coco['images']
        self.anns = self.coco['annotations']

        self.img_id_to_anns = {}
        for ann in self.anns:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        anns = self.img_id_to_anns.get(img_info['id'], [])
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]

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
    
    """ train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, shuffle=cfg['dataset']['test_shuffle']
    ) """
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_coco
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        shuffle=False,
        # sampler=val_sampler,
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


def build_transform(cfg, train=True):
    if train:
        transform = A.Compose([
            A.OneOf([
                A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.7),
                A.Resize(224, 224, p=0.3),
            ], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(p=0.3),
            A.ToGray(p=0.05),
            A.CoarseDropout(
                max_holes=1,  # always one hole
                max_height=16, max_width=16,  # size of the hole
                min_holes=1, min_height=16, min_width=16,
                fill_value=0  # fill with black
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params=A.BboxParams(format='coco', min_area=1, min_visibility=0.2, label_fields=['category_ids'])
        )
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        bbox_params=A.BboxParams(format='coco', min_area=1, min_visibility=0.0, label_fields=['category_ids'])
        )
    
    return transform
