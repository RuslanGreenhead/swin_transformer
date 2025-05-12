import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy
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
from utils import AverageMeter

CONFIG_PATH = "configs/SwinT_det_30e_default.yaml"


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
    if cfg['backbone'] == "swin":
        backbone = ...
    elif cfg['backbone'] == "resnet50":
        backbone = ResNet50Backbone()
    
    model = SSD(backbone=backbone, n_classes=cfg['n_classes'])

    return model


class TrainerDET:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, cfg):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs_run = 0
        self.batch_size = cfg['batch_size']
        self.batch_interval = cfg['batch_interval']
        self.num_epochs = cfg['num_epochs']
        self.accumulation_steps = cfg['accumulation_steps']
        self.save_every = cfg['save_every']
        self.warmup_epochs = cfg['warmup_epochs']
        self.snapshot_path = cfg['snapshot_path']
        self.max_grad_norm = cfg['max_grad_norm']

        self.model = model.to(self.gpu_id)
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot...")
            self._load_snapshot(self.snapshot_path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.num_batches = len(self.train_loader)
        self.scheduler = self._build_scheduler(cfg, self.num_batches // self.accumulation_steps)
        
        self.stats = {
            "train_losses": np.empty(self.num_epochs),
            "train_accs": np.empty(self.num_epochs),
            "val_losses": np.empty(self.num_epochs),
            "val_top1_accs": np.empty(self.num_epochs),
            "val_top5_accs": np.empty(self.num_epochs),
            "grad_norms": np.empty(self.num_epochs)
        }

        # logging setup
        log_path = f"training_gpu_{self.gpu_id}.log"
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')


    def _build_scheduler(self, cfg, n_iter_per_epoch):
        """
        **old scheduler**: 
        warmup_scheduler = LinearLR(self.optimizer, total_iters=self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs - self.warmup_epochs, eta_min=1e-6)
        chained_scheduler = SequentialLR(self.optimizer,
                                         schedulers=[warmup_scheduler, cosine_scheduler],
                                         milestones=[self.warmup_epochs])
        
        return chained_scheduler 
        """

        num_steps = cfg['num_epochs'] * n_iter_per_epoch
        warmup_steps = cfg['warmup_epochs'] * n_iter_per_epoch

        # there was also "t_mul=1." - but this parameter is no more present at library source
        lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(num_steps - warmup_steps) if cfg['warmup_prefix'] else num_steps,
            lr_min=cfg['min_lr'],
            warmup_lr_init=cfg['warmup_lr'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=cfg['warmup_prefix']
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

        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            bboxes, labels = targets['bboxes'], targets['labels']
            images = images.to(self.gpu_id)
            labels = [img_labels.to(self.gpu_id) for img_labels in labels]
            bboxes = [img_bboxes.to(self.gpu_id) for img_bboxes in bboxes]
            
            pred_locs, pred_scores = self.model(images)
            loss = self.criterion(pred_locs, pred_scores, bboxes, labels)
            
            if ((i + 1) % self.accumulation_steps == 0):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)   # grad clipping
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step_update((epoch * self.num_batches + i) // self.accumulation_steps)
            else:
                with self.model.no_sync():
                    loss.backward()
            
            _, preds = torch.max(outputs, dim=1)
            interval_loss_meter.update(loss.item())
            total_loss_meter.update(loss.item())
            
            # stamping batch interval stats
            if ((i + 1) % self.batch_interval == 0) or ((i + 1) == self.num_batches):
                lr = self.optimizer.param_groups[0]['lr']
                
                # logging stats
                logging.info(f'GPU[{self.gpu_id}]:  ' +
                             f'batch [{i+1}/{self.num_batches}]:  ' +
                             f'loss={interval_loss_meter.avg:.4f}  ' +
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

        val_criterion = nn.CrossEntropyLoss()    # simple CE instead of SoftTarget

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
                
                outputs = self.model(images)
                loss = val_criterion(outputs, labels)
                _, preds = torch.max(outputs, dim=1)
                val_loss_meter.update(loss.item())
            
        # writing stats
        self.stats['val_losses'][epoch] = val_loss_meter.avg

        # logging stats
        logging.info(f'Validation on GPU[{self.gpu_id}]:  ' +
                     f'val_loss={val_loss_meter.avg:.6f}')
    

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

    # learning rates scaling
    linear_scaled_lr = cfg['training']['initial_lr'] * cfg['training']['batch_size'] * get_world_size() / 512.0
    linear_scaled_warmup_lr = cfg['training']['warmup_lr'] * cfg['training']['batch_size'] * get_world_size() / 512.0
    linear_scaled_min_lr = cfg['training']['min_lr'] * cfg['training']['batch_size'] * get_world_size() / 512.0
    if cfg['training']['accumulation_steps'] > 1:
        linear_scaled_lr = linear_scaled_lr * cfg['training']['accumulation_steps']
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg['training']['accumulation_steps']
        linear_scaled_min_lr = linear_scaled_min_lr * cfg['training']['accumulation_steps']
    cfg['training']['initial_lr'] = linear_scaled_lr
    cfg['training']['warmup_lr'] = linear_scaled_warmup_lr
    cfg['training']['min_lr'] = linear_scaled_min_lr
    
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg['model'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['training']['initial_lr'],
                                  weight_decay=cfg['training']['weight_decay'])
    
    if cfg['dataset']['aug']['mixup'] > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    
    trainer = TrainerDET(model, train_loader, val_loader, optimizer, criterion, mixup_fn, cfg['training'])
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    main(cfg)