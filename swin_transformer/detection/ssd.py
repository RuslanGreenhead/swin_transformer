import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import cxcy_to_xy, xy_to_cxcy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils import coco_to_cxcy, coco_to_xy, normalize_boxes
from utils import calculate_iou


class SSD(nn.Module):
    """
    Single-Shot multibox Detector.
    
    Parameters:
        backbone (nn.Module): feature-extracting module
        n_classes (int): number of classes (including "background" index)
        input_size (int): spacial size of input square images
        neck (nn.Module, optional): module for additional processing of exctracted features

    """

    def __init__(self, backbone, n_classes, input_size=300, neck=None):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.input_size = input_size
        self.neck = neck

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in C3
        nn.init.constant_(self.rescale_factors, 20)

        self.location_convs = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(2048, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])

        self.score_convs = nn.ModuleList([
            nn.Conv2d(512, 4 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(2048, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)
        ])

        self.priors_cxcy = self.generate_priors(image_size=input_size)
        self._init_head_convs()


    def forward(self, x):
        feature_maps = list(self.backbone(x))   # -> [C3 - C8]
        feature_maps[0] = F.normalize(feature_maps[0], p=2, dim=1) * self.rescale_factors    # rescale C3

        if self.neck is not None:
            feature_maps = self.neck(feature_maps)

        batch_size = feature_maps[0].size(0)
        bbox_offsets = []
        bbox_scores = []

        for fmap, loc_conv in zip(feature_maps, self.location_convs):
            l = loc_conv(fmap).permute(0, 2, 3, 1).contiguous()
            # l = l.view(batch_size, -1, 4)
            # print(f"{l.shape=}")
            # bbox_offsets.append(l)
            bbox_offsets.append(l.view(batch_size, -1, 4))
        for fmap, score_conv in zip(feature_maps, self.score_convs):
            s = score_conv(fmap).permute(0, 2, 3, 1).contiguous()
            bbox_scores.append(s.view(batch_size, -1, self.n_classes))

        bbox_offsets = torch.cat(bbox_offsets, axis=1)
        bbox_scores = torch.cat(bbox_scores, axis=1)

        return bbox_offsets, bbox_scores


    def generate_priors(self, image_size):
        if image_size == 224:
            feature_map_dims = [28, 14, 7, 4, 2, 1]     # n_priors = 4722
        elif image_size == 300:
            feature_map_dims = [38, 19, 10, 5, 3, 1]    # n_priors = 8732
        else:
            raise NotImplementedError("Unexpected image_size for anchor generation!")

        # scales for each feature map (as fractions of image size)
        scales = np.empty(len(feature_map_dims))
        scales[0] = 0.1
        scales[1:] = np.linspace(0.2, 0.9, len(feature_map_dims) - 1)

        # aspect ratios per feature map
        aspect_ratios = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 3.0, 0.5, 1.0/3],
            [1.0, 2.0, 3.0, 0.5, 1.0/3],
            [1.0, 2.0, 3.0, 0.5, 1.0/3],
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5],
        ]

        priors = []
        for k, fm_dim in enumerate(feature_map_dims):
            scale = scales[k]
            scale_next = scales[k + 1] if k + 1 < len(scales) else 1.0

            for i in range(fm_dim):
                for j in range(fm_dim):
                    cx = (j + 0.5) / fm_dim
                    cy = (i + 0.5) / fm_dim

                    for ar in aspect_ratios[k]:
                        w = scale * np.sqrt(ar)
                        h = scale / np.sqrt(ar)
                        priors.append([cx, cy, w, h])

                        # for aspect ratio 1, add an extra prior with scale = sqrt(s_k * s_{k+1})
                        if ar == 1.0:
                            if k < len(feature_map_dims):
                                extra_scale = np.sqrt(scale * scale_next)
                                priors.append([cx, cy, extra_scale, extra_scale])

        priors = torch.FloatTensor(np.clip(priors, 0.0, 1.0))

        return priors  # (n_priors, 4)


    def detect_objects(self, pred_locs, pred_scores, min_score, max_overlap, top_k):
        """
        Decipher (n_priors) locations and class scores (output of ths SSD) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        Parameters:
            pred_locs (Tensor): predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
            pred_scores (Tensor): class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            min_score (float): minimum threshold for a box to be considered a match for a certain class
            max_overlap (float): maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k (int): if there are a lot of resulting detection across all classes, keep only the top 'k'
        Returns:
            detections (boxes, labels, and scores) -> lists of (batch_size) length
        """

        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        pred_scores = F.softmax(pred_scores, dim=2)  # (N, n_priors, n_classes)
        device = pred_locs.device

        # lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(pred_locs[i], self.priors_cxcy.to(device)))  # (n_priors, 4)

            # lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = pred_scores[i].max(dim=1)  # (4721)

            # for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = pred_scores[i][:, c]                     # (n_priors)
                score_above_min_score = class_scores > min_score        # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]        # (n_qualified), n_min_score <= n_priors
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = calculate_iou(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)
                keep = []

                # # Consider each box in order of decreasing scores
                # for box in range(class_decoded_locs.size(0)):
                #     # If this box is already marked for suppression
                #     if suppress[box] == 1:
                #         continue

                #     # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                #     # Find such boxes and update suppress indices
                #     suppress = torch.max(suppress, overlap[box] > max_overlap)
                #     # The max operation retains previously suppressed boxes, like an 'OR' operation

                #     # Don't suppress this box, even though it has an overlap of 1 with itself
                #     suppress[box] = 0

                for box in range(class_decoded_locs.size(0)):
                    if box in keep:
                        continue
                    keep.append(box)

                    # suppress boxes with IoU > max_overlap that come after box i (lower scores)
                    suppress_indices = (overlap[box] > max_overlap).nonzero(as_tuple=False).squeeze(1)
                    # only suppress boxes with index > i (i.e., lower ranked boxes)
                    suppress_indices = suppress_indices[suppress_indices > box]

                    for idx in suppress_indices:
                        if idx not in keep:
                            suppress[idx] = 1

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # if no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)    # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]            # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]    # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


    def _init_head_convs(self):
        for name, module in self.named_modules():
            if ("score_convs" in name) or ("location_convs" in name):
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


class MultiBoxLoss(nn.Module):
    """
    Multitask loss comprising localization part (MSE) and confidence part (CE)

    Parameters:
        priors_cxcy (Tensor): model's prior boxes in "cxcy" (cx, cy, w, h) format.
        img_size (int): spacial size of quadratic input image
        threshold (float): IoU threshold to consider prior box a positive match
        neg_pos_ratio (int): n_hard_negatives = neg_pos_ratio * n_positives
        alpha (float): coefficient before localization part in loss
    """

    def __init__(self, priors_cxcy, img_size=224, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.img_size = img_size
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred_locs, pred_scores, gt_boxes, gt_labels):
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = pred_scores.size(2)
        device = pred_locs.device

        print(f"{n_priors=}, {pred_locs.size(1)=}, {pred_scores.size(1)=}")
        assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, n_priors, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)   # (N, n_priors)

        # for each image
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)
            # if image contains no objects -> skip it
            if n_objects == 0: 
                continue

            # raw COCO boxes -> to xy-format + normalize
            img_boxes_xy = coco_to_xy(gt_boxes[i])
            img_boxes_xy = normalize_boxes(img_boxes_xy, img_width=self.img_size, img_height=self.img_size)

            overlap = calculate_iou(img_boxes_xy, self.priors_xy.to(device))     # (n_objects, n_priors)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)   # (n_priors)

            _, prior_for_each_object = overlap.max(dim=1)                        # (n_objects)


            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # labels for each prior
            label_for_each_prior = gt_labels[i][object_for_each_prior]          # (n_priors)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0   # (n_priors) -> background ID

            # store
            true_classes[i] = label_for_each_prior

            # encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(                          # (n_priors, 4)
                xy_to_cxcy(img_boxes_xy[object_for_each_prior]),
                self.priors_cxcy.to(device)
            )  

        positive_priors = true_classes != 0                           # (N, n_priors)
        n_positives = positive_priors.sum(dim=1)                      # (N)
        n_positives_clamped = n_positives.clamp(min=1.).float()
        n_hard_negatives = self.neg_pos_ratio * n_positives_clamped   # (N)

        # localization part
        if n_positives.sum() > 0:
            loc_loss = self.smooth_l1(pred_locs[positive_priors], true_locs[positive_priors])  # (), scalar
        else:
            loc_loss = torch.tensor(0., device=device)

        # confidence part
        conf_loss_all = self.cross_entropy(pred_scores.view(-1, n_classes), true_classes.view(-1))  # (N * n_priors)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)                                    # (N, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, n_priors)
        conf_loss_neg[positive_priors] = 0.    # (N, n_priors), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)    # (N, n_priors), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, n_priors)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, n_priors)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]               # (sum(n_hard_negatives))

        # as in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives_clamped.sum()  # (), scalar

        return conf_loss + self.alpha * loc_loss

