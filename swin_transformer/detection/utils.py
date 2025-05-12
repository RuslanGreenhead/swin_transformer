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
