import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
import time
import logging

import os
import numpy as np
import pickle
import yaml
from tqdm import tqdm

from ssd import SSD, MultiBoxLoss
from backbones import ResNet50Backbone
from data import build_loaders
from utils import AverageMeter, calculate_coco_mAP, calculate_coco_mAP_pcct
from utils import normalize_boxes, coco_to_xy

CONFIG_PATH = "../configs/detection_ssd_resnet50.yaml"


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def ddp_setup():
    """
    Prerequisites for distributed training.
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def build_model(cfg):
    if cfg['model']['backbone'] == "swin":
        pass
    elif cfg['model']['backbone'] == "resnet50":
        backbone = ResNet50Backbone(img_size=cfg['model']['image_size'], weights=cfg['model']['backbone_weights'])
    else:
        raise NotImplementedError("Unsupported backbone.")
    
    # here we increase the number of classes by one for the background
    model = SSD(backbone=backbone, n_classes=cfg['model']['n_classes'] + 1, input_size=cfg['model']['image_size'])

    return model


def build_optimizer(model, cfg):

    if cfg['model']['backbone'] == 'resnet50':
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'bn' in name or 'batchnorm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.SGD(
            [{'params': decay_params, 'weight_decay': cfg['train']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}], 
            lr=cfg['train']['initial_lr'],
            momentum=cfg['train']['momentum']
        )

    elif cfg['model']['backbone'] == 'swin':

        optimizer = torch.optim.AdamW(model.parameters(),
            lr=cfg['train']['initial_lr'],
            weight_decay=cfg['train']['weight_decay']
        )
    
    else:
        raise NotImplementedError("No optimizaer for such backbone name.")
    
    return optimizer


class TrainerDET:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, cfg):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs_run = 0
        self.batch_size = cfg['train']['batch_size']
        self.batch_interval = cfg['train']['batch_interval']
        self.num_epochs = cfg['train']['num_epochs']
        self.accumulation_steps = cfg['train']['accumulation_steps']
        self.save_every = cfg['train']['save_every']
        self.warmup_epochs = cfg['train']['warmup_epochs']
        self.snapshot_path = cfg['train']['snapshot_path']
        self.max_grad_norm = cfg['train']['max_grad_norm']
        self.image_size = cfg['model']['image_size']

        if self.max_grad_norm is None:
            self.max_grad_norm = 1e10

        self.model = model.to(self.gpu_id)
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot...")
            self._load_snapshot(self.snapshot_path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.num_batches = len(self.train_loader)
        self.scheduler = self._build_scheduler(cfg, self.num_batches // self.accumulation_steps)
        
        self.stats = {
            "train_losses": np.empty(self.num_epochs),
            "train_mAP": np.empty(self.num_epochs),
            "val_losses": np.empty(self.num_epochs),
            "val_mAP": np.empty(self.num_epochs),
            "val_mAP_pycoco": np.empty(self.num_epochs),
            "grad_norms": np.empty(self.num_epochs)
        }

        # logging setup
        log_path = f"training_gpu_{self.gpu_id}.log"
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')


    def _build_scheduler(self, cfg, n_iter_per_epoch):
        num_steps = cfg['train']['num_epochs'] * n_iter_per_epoch
        warmup_steps = cfg['train']['warmup_epochs'] * n_iter_per_epoch

        if cfg['model']['backbone'] == "resnet50":
            lr_scheduler = MultiStepLRScheduler(
                self.optimizer,
                decay_t=[33 * n_iter_per_epoch, 50 * n_iter_per_epoch],     # тут можно тоже запихнуть в конфиг       
                decay_rate=cfg['train']['lr_decay_rate'],         
                warmup_t=warmup_steps,             
                warmup_lr_init=cfg['train']['warmup_lr'],
                warmup_prefix=cfg['train']['warmup_prefix'],
                t_in_epochs=False 
            )
        elif cfg['model']['backbone'] == "swin":
            lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if cfg['warmup_prefix'] else num_steps,
                lr_min=cfg['min_lr'],
                warmup_lr_init=cfg['warmup_lr'],
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=cfg['train']['warmup_prefix']
            )

        return lr_scheduler
    

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

        logging.info(f"Resuming training from snapshot at Epoch {self.epochs_run}.")
    

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)

        logging.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}.")
    

    @staticmethod
    def _seconds_to_hhmmss(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = round(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()   # in case if some grads are left from prev epoch
        total_loss_meter = AverageMeter()
        interval_loss_meter = AverageMeter()

        logging.info(f"GPU[{self.gpu_id}]: Epoch {epoch} started.")

        self.train_loader.sampler.set_epoch(epoch)

        for i, (images, boxes, labels) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.gpu_id)
            boxes = [img_boxes.to(self.gpu_id) for img_boxes in boxes]
            labels = [img_labels.to(self.gpu_id) for img_labels in labels]
            
            pred_locs, pred_scores = self.model(images)
            loss = self.criterion(pred_locs, pred_scores, boxes, labels)
            
            if ((i + 1) % self.accumulation_steps == 0):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)   # grad clipping
                
                # grad_norm_1 = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1e10)
                # if self.gpu_id == 0:
                #     print(f'{self.max_grad_norm=}\nMy debug: {grad_norm=:.4f}', flush=True)
                # logging.info(f'Look at the norm: {grad_norm=:.4f}, {grad_norm_1=:.4f}')

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step_update((epoch * self.num_batches + i) // self.accumulation_steps)
            else:
                with self.model.no_sync():
                    loss.backward()
            
            interval_loss_meter.update(loss.item())
            total_loss_meter.update(loss.item())
            
            # stamping batch interval stats
            if ((i + 1) % self.batch_interval == 0) or ((i + 1) == self.num_batches):
                lr = self.optimizer.param_groups[0]['lr']
                
                # put no_grad just in case...
                with torch.no_grad():
                    # calculate COCO-style mAP
                    det_boxes, det_labels, det_scores = self.model.module.detect_objects(
                        pred_locs, pred_scores, 
                        min_score=0.5, max_overlap=0.5, top_k=5
                    )
                    # normalize GT boxes and cast them to "xyxy" format
                    boxes_n_xyxy = [normalize_boxes(coco_to_xy(boxes_n_img),
                                                    img_width=self.image_size,
                                                    img_height=self.image_size,
                                                    box_format="xy") for boxes_n_img in boxes]
                    # mAP = calculate_coco_mAP(det_boxes, det_labels, det_scores, boxes_n_xyxy, labels)
                    mAP_pycoco, _ = calculate_coco_mAP_pcct(det_boxes, det_labels, det_scores, boxes_n_xyxy, labels)
                
                # logging stats
                logging.info(f'GPU[{self.gpu_id}]:  ' +
                             f'batch [{i+1}/{self.num_batches}]:  ' +
                             f'loss={interval_loss_meter.avg:.4f}  ' +
                             f'COCO_mAP={mAP_pycoco:.6f}  ' +
                             f'grad_norm={grad_norm:.4f}  ' +
                             f'lr={lr:.6f}')
                
                interval_loss_meter.reset()   
 
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)

        # writing stats
        self.stats['train_losses'][epoch] = total_loss_meter.avg
        self.stats['grad_norms'][epoch] = grad_norm


    def validate(self, epoch):
        self.model.eval()
        val_loss_meter = AverageMeter()
        self.val_loader.sampler.set_epoch(epoch)

        all_det_boxes, all_det_labels, all_det_scores = [], [], []
        all_true_boxes, all_true_labels = [], []

        with torch.no_grad():
            for i, (images, boxes, labels) in enumerate(tqdm(self.val_loader)):
                images = images.to(self.gpu_id)
                boxes = [img_boxes.to(self.gpu_id) for img_boxes in boxes]
                labels = [img_labels.to(self.gpu_id) for img_labels in labels]
                
                pred_locs, pred_scores = self.model(images)
                loss = self.criterion(pred_locs, pred_scores, boxes, labels)
                val_loss_meter.update(loss.item())

                det_boxes, det_labels, det_scores = self.model.module.detect_objects(
                    pred_locs, pred_scores, 
                    min_score=0.5, max_overlap=0.5, top_k=5
                )

                # normalize GT boxes and cast them to "xyxy" format
                boxes_n_xyxy = [normalize_boxes(coco_to_xy(boxes_n_img),
                                                img_width=self.image_size,
                                                img_height=self.image_size,
                                                box_format="xy") for boxes_n_img in boxes]

                all_true_boxes.extend(boxes_n_xyxy)
                all_true_labels.extend(labels)
                all_det_boxes.extend(det_boxes)
                all_det_labels.extend(det_labels)
                all_det_scores.extend(det_scores)
            
        # calculating COCO-style mAP
        # mAP = calculate_coco_mAP(all_det_boxes, all_det_labels, all_det_scores, all_true_boxes, all_true_labels)
        mAP_pycoco, _ = calculate_coco_mAP_pcct(all_det_boxes, all_det_labels, all_det_scores, all_true_boxes, all_true_labels)

        # writing stats
        self.stats['val_losses'][epoch] = val_loss_meter.avg
        self.stats['val_mAP'][epoch] = mAP_pycoco

        # logging stats
        logging.info(f'Validation on GPU[{self.gpu_id}]:  ' +
                     f'val_loss={val_loss_meter.avg:.6f}  ' +
                     f'val_COCO_mAP={mAP_pycoco:.6f}')
    

    def train(self):
        start_time = time.time()
        for epoch in range(self.epochs_run, self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

            if (self.gpu_id == 0) and ((epoch + 1) % self.save_every == 0):
                self._save_snapshot(epoch)

            time_elapsed = self._seconds_to_hhmmss(time.time() - start_time)
            logging.info(f"(GPU[{self.gpu_id}]): Epoch {epoch} completed. Time elapsed: {time_elapsed}")
            print(f"(GPU[{self.gpu_id}]): Epoch {epoch} completed. Time elapsed: {time_elapsed}")
        
        # saving model weights (on one GPUs)
        if self.gpu_id == 0:
            logging.info(f"Saving model by GPU[{self.gpu_id}]...")
            torch.save(self.model.module.state_dict(), 'final_model.pth')

        # saving training stats (on both GPUs)
        with open(f'training_output_{self.gpu_id}.pkl', 'wb') as f:
            pickle.dump(self.stats, f)

        logging.info(f"Training for {self.num_epochs} epochs  is completely finished. Results saved.")
        print(f"Training for {self.num_epochs} epochs  is completely finished. Results saved.")


def main(cfg):                       
    ddp_setup()

    # learning rates scaling (according to total batch size of 32 -- for ResNet)
    linear_scaled_lr = cfg['train']['initial_lr'] * cfg['train']['batch_size'] * get_world_size() / 32.0
    linear_scaled_warmup_lr = cfg['train']['warmup_lr'] * cfg['train']['batch_size'] * get_world_size() / 32.0
    linear_scaled_min_lr = cfg['train']['min_lr'] * cfg['train']['batch_size'] * get_world_size() / 32.0
    if cfg['train']['accumulation_steps'] > 1:
        linear_scaled_lr = linear_scaled_lr * cfg['train']['accumulation_steps']
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg['train']['accumulation_steps']
        linear_scaled_min_lr = linear_scaled_min_lr * cfg['train']['accumulation_steps']
    cfg['train']['initial_lr'] = linear_scaled_lr
    cfg['train']['warmup_lr'] = linear_scaled_warmup_lr
    cfg['train']['min_lr'] = linear_scaled_min_lr
    
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg)
    criterion = MultiBoxLoss(model.priors_cxcy, img_size=cfg['model']['image_size'])
    optimizer = build_optimizer(model, cfg)
    
    trainer = TrainerDET(model, train_loader, val_loader, optimizer, criterion, cfg)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    main(cfg)