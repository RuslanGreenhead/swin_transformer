import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import cxcy_to_xy, xy_to_cxcy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils import coco_to_cxcy, coco_to_xy
from utils import calculate_iou, build_coco_label_index


class SSD(nn.Module):
    def __init__(self, backbone, n_classes, neck=None, input_size=224):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes

        self.location_convs = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(2048, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])

        self.score_convs = nn.ModuleList([
            nn.Conv2d(512, 4 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(2048, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 4 * n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)
        ])

        self.priors_cxcy = self.generate_priors(image_size=input_size)


    def forward(self, x):
        print(f"Input shape: {x.shape}")
        feature_maps = self.backbone(x)   # -> [C3 - C8]
        print(f"FMs shapes: {[x.shape for x in feature_maps]}")
        batch_size = feature_maps[0].size(0)
        bbox_offsets = []
        bbox_scores = []

        for fmap, loc_conv in zip(feature_maps, self.location_convs):
            # res = loc_conv(fmap)
            # print(res.shape)
            # input()
            bbox_offsets.append(loc_conv(fmap).view(batch_size, -1, 4))
        for fmap, score_conv in zip(feature_maps, self.score_convs):
            bbox_scores.append(score_conv(fmap).view(batch_size, -1, self.n_classes))

        # print(f"{[x.shape for x in bbox_offsets]=}")
        # print(f"{[x.shape for x in bbox_scores]=}")
        # input()
        bbox_offsets = torch.cat(bbox_offsets, axis=1)
        bbox_scores = torch.cat(bbox_scores, axis=1)

        return bbox_offsets, bbox_scores


    def generate_priors(self, image_size=224):
        if image_size == 224:
            feature_map_dims = [28, 14, 7, 4, 2, 1]  # Approximated for 224x224
        else:
            feature_map_dims = None 

        # Scales for each feature map (as fractions of image size)
        scales = np.empty(len(feature_map_dims))
        scales[0] = 0.1
        scales[1:] = np.linspace(0.2, 0.9, len(feature_map_dims) - 1)

        # Aspect ratios per feature map
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
                        # For aspect ratio 1, add an extra prior with scale = sqrt(s_k * s_{k+1})
                        if ar == 1.0:
                            if k < len(feature_map_dims):
                                extra_scale = np.sqrt(scale * scale_next)
                                priors.append([cx, cy, extra_scale, extra_scale])

        priors = torch.FloatTensor(np.clip(priors, 0.0, 1.0))    # --> to.(device)

        return priors  # shape: (4722, 4)


    def detect_objects(self, pred_locs, pred_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param pred_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        pred_scores = F.softmax(pred_scores, dim=2)  # (N, 4721, n_classes)
        device = pred_locs.device

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(pred_locs[i], self.priors_cxcy))  # (4721, 4)

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = pred_scores[i].max(dim=1)  # (4721)

            # for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = pred_scores[i][:, c]  # (4721)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
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

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.id_to_idx = build_coco_label_index()

    def forward(self, pred_locs, pred_scores, gt_boxes, gt_labels):
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = pred_scores.size(2)
        device = pred_locs.device

        # print(n_priors, pred_locs.shape, pred_scores.shape)
        # print(f"{gt_labels=}")
        gt_labels = [self.id_to_idx[img_labels] for img_labels in gt_labels]

        assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 4721, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 4721)

        # For each image
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)

            overlap = calculate_iou(gt_boxes[i], self.priors_xy)                 # (n_objects, 8732)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)   # (4721)

            _, prior_for_each_object = overlap.max(dim=1)                        # (n_objects)


            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = gt_labels[i][object_for_each_prior]          # (4721)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0   # (4721)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(coco_to_cxcy(gt_boxes[i][object_for_each_prior]), self.priors_cxcy)  # (4721, 4)

        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS
        loc_loss = self.smooth_l1(pred_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        conf_loss_all = self.cross_entropy(pred_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)                                    # (N, 8732)
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        return conf_loss + self.alpha * loc_loss

