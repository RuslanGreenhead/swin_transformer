import torch
from torch import nn
import numpy as np


def build_coco_label_index():
    '''
    Maps COCO category ID ([1-90] with gaps) to a continuous range [0-79]
    '''
    excluded_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
    coco_ids = list(set(range(1, 91)) - set(excluded_ids))

    map_dict =  {id: idx for idx, id in enumerate(coco_ids)}
    map_tensor = torch.zeros(max(map_dict.keys()) + 1, dtype=torch.long)    #  == int64
    for k, v in map_dict.items():
        map_tensor[k] = v
    
    return map_tensor


def xy_to_cxcy(xy):

    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):

    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),      # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def coco_to_xy(coco):
    '''
    (x_min, y_min, w, h) -> (x_min, y_min, x_max, y_max)
    '''

    return torch.cat([coco[:, :2],                          # x_min, y_min
                      coco[:, :2] + coco[:, 2:]], 1)        # x_max, y_max


def coco_to_cxcy(coco):
    '''
    (x_min, y_min, w, h) -> (cx, cy, w, h)
    '''

    return torch.cat([coco[:, :2] + coco[:, 2:] / 2,        # cx, cy
                      coco[:, 2:]], 1)                      # w, h


def cxcy_to_gcxgcy(cxcy, priors_cxcy):

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def calculate_iou(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


class AverageMeter:
    """
    Tracker for various metrics & statistics.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.mean = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count


def calculate_AP(recalls, precisions):
    """
    Compute Average Precision (AP) using 11-point interpolation.
    recall, precision are 1D tensors sorted by recall ascending.
    """
    
    recall_levels = torch.linspace(0, 1, 11)
    ap = 0.0
    for t in recall_levels:
        p = precisions[recalls >= t].max() if (recalls >= t).any() else 0
        ap += p / 11

    return ap.item()


def calculate_mAP(det_boxes, det_labels, det_scores,
                  true_boxes, true_labels, iou_threshold=0.5, n_classes=80):
    """
    Calculate mAP for COCO-style detection.

    Arguments:
        det_boxes: list of tensors of shape (n_detections, 4), each describing single image
        det_labels: list of tensors of shape (n_detections,), each describing single image  
        det_scores: list of tensors of shape (n_detections,), each describing single image  
        true_boxes: list of tensors of shape (n_objects, 4), each describing single image
        true_labels: list of tensors of shape (n_objects,), each describing single image

    Returns:
        (float): mAP value
        (dict): [c: v] -> AP value (v) for each class (c)
    """
    if n_classes is None:
        n_classes = int(torch.cat(true_labels).max().item()) + 1  
    device = det_boxes[0].device  

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

    # create list with "image_ids" for each ground truth box:
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)
    # flatten true boxes and labels over images:
    true_boxes = torch.cat(true_boxes, dim=0)         # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)       # (n_objects)

    # create list with "image_ids" for each detected box:
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)       # (n_detections)
    # flatten detected boxes, labels and scores over images:
    det_boxes = torch.cat(det_boxes, dim=0)                    # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)                  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)                  # (n_detections)

    ap_per_class = {}

    for c in range(n_classes):
        # gather all targets and detections of class (c)
        true_class_images = true_images[true_labels == c]      # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]        # (n_class_objects, 4)
        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)

        det_class_images = det_images[det_labels == c]     # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]       # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]     # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]                                      # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]                                        # (n_class_detections, 4)

        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)   # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]                      # (), scalar

            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = calculate_iou(this_detection_box, object_boxes)    # (1, n_class_objects_in_img)
            max_overlap, idx = torch.max(overlaps.squeeze(0), dim=0)      # (), () - scalars
            # Нам нужно дополнрительно хранить оригинальный ID (ID в общем пуле детекций), так как считали IoU и выбирали из результатов максимальный
            # мы только из детекций одного конкретного изображение (и, соответствено, получили ID в рамках этого изображения):
            original_idx = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][idx]

            if max_overlap.item() > iou_threshold:
                # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_idx] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_idx] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative sums of TP and FP
        tp_cum = torch.cumsum(true_positives, dim=0)
        fp_cum = torch.cumsum(false_positives, dim=0)

        recalls = tp_cum / max(len(true_class_boxes), 1)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-10)

        # Add (0,1) point for recall and precision to complete curve
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        precisions = torch.cat((torch.tensor([1.0]), precisions))

        # Compute AP for this class
        ap = calculate_AP(recalls, precisions)
        ap_per_class[c] = ap

    # Calculate mean AP
    mean_ap = sum(ap_per_class.values()) / len(ap_per_class) if len(ap_per_class) > 0 else 0.0

    return mean_ap, ap_per_class


def calculate_coco_mAP(det_boxes, det_labels, det_scores,
                       true_boxes, true_labels, n_classes=80):
    """
    Calculate mean mAP over 10 IoU thresholds within range (0.5, 0.95)
    """

    ious = torch.linspace(0.5, 0.95, 10)
    aps = []

    for t in ious:
        mean_ap, ap_per_class = calculate_mAP(det_boxes, det_labels, det_scores,
                                              true_boxes, true_labels, iou_threshold=t, n_classes=80)
        aps.append(mean_ap)

    coco_map = sum(aps) / len(aps)
    # print(f"COCO-style mAP @[0.5:0.95]: {coco_map:.4f}")

    return coco_map
