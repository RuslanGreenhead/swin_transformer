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
    """ model = SSD(backbone=ResNet50Backbone(weights=cfg['model']['backbone_weights']), n_classes=cfg['model']['n_classes'])
    model.train()
    criterion = MultiBoxLoss(model.priors_cxcy) """

    """ for i, (images, boxes, labels) in enumerate(train_loader):
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
        print("Backward performed!") """
    
    print("Validation:")
    # model.eval()

    no_objects_images = 0
    all_images = 0

    for i, (images, boxes, labels) in enumerate(train_loader):

        boxes_lens = torch.tensor([len(img_boxes) for img_boxes in boxes])
        no_objects_images += (boxes_lens == 0).sum()
        all_images += len(boxes_lens)
        print(f"{no_objects_images=}    batch {i+1}/{len(train_loader)}", flush=True)


        """ if i == 0: continue
        elif i == 3: break

        pred_locs, pred_scores = model(images)
        batch_boxes, batch_labels, batch_scores = model.detect_objects(pred_locs, pred_scores, 
                                                                       min_score=0.5, max_overlap=0.5, top_k=5)
        
        boxes = [coco_to_xy(img_boxes) for img_boxes in boxes]                            # boxes to xy format
        print(f"{batch_boxes[0].shape=}, {boxes[0].shape=}")
        labels = [model.id_to_idx[img_labels] for img_labels in labels]                   # labels to continious range

        torch.save(batch_boxes, f"batch_boxes_{i}.pth")
        torch.save(batch_labels, f"batch_labels_{i}.pth")
        torch.save(batch_scores, f"batch_scores_{i}.pth")
        torch.save(boxes, f"boxes_{i}.pth")
        torch.save(labels, f"lablels_{i}.pth") 
        mAP = calculate_coco_mAP(batch_boxes, batch_labels, batch_scores, boxes, labels)

        print("Manually calculated mAP: ", mAP) """

    print("Train Images with no boxes:", no_objects_images)
    print("Total train images:", all_images)

        
