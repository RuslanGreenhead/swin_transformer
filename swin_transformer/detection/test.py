import torch 
from torch import nn
import numpy as np
import yaml
import itertools

from tqdm import tqdm

from data import build_loaders
from ssd import SSD, MultiBoxLoss
from backbones import ResNet50Backbone, ResNet50Backbone_Deeper
from utils import calculate_coco_mAP, coco_to_xy, gcxgcy_to_cxcy, cxcy_to_xy, normalize_boxes, calculate_coco_mAP_pcct
from necks import FPN, PAN, DenseFPN

CONFIG_PATH = "../configs/detection_ssd_resnet50.yaml"

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    train_loader, val_loader = build_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SSD(backbone=ResNet50Backbone_Deeper(weights=cfg['model']['backbone_weights'], img_size=300), 
                n_classes=cfg['model']['n_classes'] + 1,
                input_size=cfg['model']['image_size'],
                neck=DenseFPN())
    model = model.to(device)
    model.load_state_dict(torch.load("final_model.pth"))
    model.eval()

    # min_scores = [0.01, 0.05, 0.1, 0.2]
    # max_overlaps = [0.4, 0.45, 0.5]
    # top_ks = [100, 200]

    # min_score=0.01, max_overlap=0.5, top_k=325: mAP 0.24960956272687362
    # min_score=0.01, max_overlap=0.5, top_k=350: mAP 0.24965571734650552
    # min_score=0.01, max_overlap=0.5, top_k=375: mAP 0.24967939809724807
    # min_score=0.01, max_overlap=0.5, top_k=400: mAP 0.24969983537950133
    # min_score=0.01, max_overlap=0.5, top_k=425: mAP 0.24973226237818316
    # min_score=0.01, max_overlap=0.5, top_k=450: mAP 0.24975926823983483
    # min_score=0.01, max_overlap=0.5, top_k=500: mAP 0.24980386843240363
    # min_score=0.01, max_overlap=0.5, top_k=525: mAP 0.24984077061572849
    # min_score=0.01, max_overlap=0.5, top_k=550: mAP 0.24986831674034457
    # min_score=0.01, max_overlap=0.5, top_k=600: mAP 0.24990524081367135

    # Full validation for ResNet+FPN (exp. 10): mAP=0.2502355349360508
    # Full validation for ResNet+PAN (exp. 11): mAP=0.25220221483815125
    



    min_scores = [0.05]
    max_overlaps = [0.45]
    top_ks = [200]

    for param_set in itertools.product(min_scores, max_overlaps, top_ks):

        all_det_boxes, all_det_labels, all_det_scores = [], [], []
        all_true_boxes, all_true_labels = [], []

        with torch.no_grad():
            for i, (images, boxes, labels) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                boxes = [img_boxes.to(device) for img_boxes in boxes]
                labels = [img_labels.to(device) for img_labels in labels]
                
                pred_locs, pred_scores = model(images)
                # loss = self.criterion(pred_locs, pred_scores, boxes, labels)
                # val_loss_meter.update(loss.item())

                det_boxes, det_labels, det_scores = model.detect_objects(
                    pred_locs, pred_scores, 
                    min_score=param_set[0], max_overlap=param_set[1], top_k=param_set[2]
                )

                # normalize GT boxes and cast them to "xyxy" format
                boxes_n_xyxy = [normalize_boxes(coco_to_xy(boxes_n_img),
                                                img_width=cfg['model']['image_size'],
                                                img_height=cfg['model']['image_size'],
                                                box_format="xy") for boxes_n_img in boxes]

                all_true_boxes.extend(boxes_n_xyxy)
                all_true_labels.extend(labels)
                all_det_boxes.extend(det_boxes)
                all_det_labels.extend(det_labels)
                all_det_scores.extend(det_scores)
            
        # calculating COCO-style mAP
        # mAP = calculate_coco_mAP(all_det_boxes, all_det_labels, all_det_scores, all_true_boxes, all_true_labels)
        mAP_pycoco, _ = calculate_coco_mAP_pcct(all_det_boxes, all_det_labels, all_det_scores, all_true_boxes, all_true_labels)
        print(f"    min_score={param_set[0]}, max_overlap={param_set[1]}, top_k={param_set[2]}: mAP {mAP_pycoco}")




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

    #     torch.save(images, "batch_train_augm.pth")

    #     print('Train images saved')
    #     break