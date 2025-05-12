import torch 
from torch import nn
import numpy as np
import yaml

from data import build_loaders
from ssd import SSD, MultiBoxLoss
from backbones import ResNet50Backbone

CONFIG_PATH = "../configs/detection_default.yaml"

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    train_loader, val_loader = build_loaders(cfg)
    model = SSD(backbone=ResNet50Backbone(weights=cfg['model']['backbone_weights']), n_classes=cfg['model']['n_classes'])
    model.train()
    criterion = MultiBoxLoss(model.priors_cxcy)

    for i, (images, boxes, labels) in enumerate(train_loader):
        print("--> iteraion ", i)
        if i > 2:
            print("--> break")
            break
        if i == 0:
            torch.save(images, "batch_images.pth")
            torch.save(boxes, "batch_boxes.pth")
            torch.save(labels, "batch_labels.pth")

        print("Batch loaded & saved")
        pred_locs, pred_scores = model(images)
        if i == 0:
            torch.save(pred_locs, "pred_locs.pth")
            torch.save(pred_scores, "pred_scores.pth")
        print("Inference succeeded! Outputs saved!")
        loss = criterion(pred_locs, pred_scores, boxes, labels)
        print("Loss calculated")
        loss.backward()
        print("Backward performed!")
