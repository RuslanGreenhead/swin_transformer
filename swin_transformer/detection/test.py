import torch 
from torch import nn
import numpy as np
import yaml

from data import build_loaders
from ssd import SSD, MultiBoxLoss
from backbones import ResNet50Backbone
from utils import calculate_coco_mAP, coco_to_xy, gcxgcy_to_cxcy, cxcy_to_xy

CONFIG_PATH = "../configs/detection_ssd_resnet50.yaml"

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    train_loader, val_loader = build_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = SSD(backbone=ResNet50Backbone(weights=cfg['model']['backbone_weights'], img_size=300), 
    #             n_classes=cfg['model']['n_classes'],
    #             input_size=cfg['model']['image_size'])
    # model = model.to(device)
    # model.load_state_dict(torch.load("final_model.pth"))
    # model.eval()

    # criterion = MultiBoxLoss(model.priors_cxcy)

    # named_params = list(model.named_parameters())
    # for i in range(129, 159):
    #     if i < len(named_params):
    #         name, param = named_params[i]
    #         print(f"Index {i}: {name}, shape={param.shape}, requires_grad={param.requires_grad}")
    #     else:
    #         print(f"Index {i} out of range (total params: {len(named_params)})")
    
    # print("Done!")


    # for i, (images, boxes, labels) in enumerate(val_loader):
    #     images = images.to(device)
    #     boxes = [img_boxes.to(device) for img_boxes in boxes]
    #     labels = [model.id_to_idx[img_labels].to(device) for img_labels in labels]

    #     pred_locs, pred_scores = model(images)
    #     det_boxes, det_labels, det_scores = model.detect_objects(
    #         pred_locs, pred_scores, 
    #         min_score=0.5, max_overlap=0.5, top_k=5
    #     )

    #     torch.save({
    #         "images": images,
    #         "boxes": boxes,
    #         "labels": labels,
    #         "pred_locs": pred_locs,
    #         "pred_scores": pred_scores,
    #         "det_boxes": det_boxes,
    #         "det_labels": det_labels,
    #         "det_scores": det_scores,
    #     }, "batch_val_detections.pth")

    #     print('Batch detections saved')
    #     break

    # model.train()

    # for i, (images, boxes, labels) in enumerate(train_loader):
    #     images = images.to(device)
    #     boxes = [img_boxes.to(device) for img_boxes in boxes]
    #     labels = [model.id_to_idx[img_labels].to(device) for img_labels in labels]

    #     pred_locs, pred_scores = model(images)
    #     det_boxes, det_labels, det_scores = model.detect_objects(
    #         pred_locs, pred_scores, 
    #         min_score=0.5, max_overlap=0.5, top_k=5
    #     )

    #     torch.save({
    #         "images": images,
    #         "boxes": boxes,
    #         "labels": labels,
    #         "pred_locs": pred_locs,
    #         "pred_scores": pred_scores,
    #         "det_boxes": det_boxes,
    #         "det_labels": det_labels,
    #         "det_scores": det_scores,
    #     }, "batch_train_detections.pth")

    #     print('Batch detections saved')
    #     break

        
