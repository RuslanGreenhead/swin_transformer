import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from matplotlib import pyplot as plt
import time
import datetime
import logging

import os
import numpy as np
import pickle
import yaml
from tqdm import tqdm

# to see "detection.utils" module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data import build_loader
from diffusion_model import MetaConditionedModel
from schedulers import DDPMScheduler
from detection.utils import AverageMeter

CONFIG_PATH = "../configs/diffusion_unet_SwinT.yaml"


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def ddp_setup(nccl_timeout=30):
    """
    Prerequisites for distributed training.
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl", 
                       timeout=datetime.timedelta(minutes=nccl_timeout)
    )  


def build_model(cfg):
    swin_params = {
        'img_size': cfg['model']['image_size'],
        'emb_dim': cfg['model']['swin']['emb_dim'], 
        'depths': cfg['model']['swin']['depths'], 
        'n_heads': cfg['model']['swin']['n_heads'],
        'mlp_drop': cfg['model']['swin']['mlp_drop'],
        'attn_drop': cfg['model']['swin']['attn_drop'],
        'drop_path': cfg['model']['swin']['drop_path'],
        'patch_size': 4, 'input_dim': 1, 'window_size': 7,
        'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'pos_drop': 0.,
        'norm_layer': nn.LayerNorm, 'ape': False, 'patch_norm': True
    }

    # loading pretrained weights if needed
    state_dict = None
    if cfg['model']['pretrained_weights'] == "IMAGENET_CUSTOM":
        state_dict = torch.load("../saved_weights/SwinT_statedict.pth", weights_only=True)['MODEL_STATE']

    model = MetaConditionedModel(
        swin_params=swin_params,
        cfg=cfg,
        weights=state_dict,
        out_dim=cfg['model']['out_dim'],
        mid_depth=cfg['model']['mid_depth'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        condition=cfg['model']['condition'],
        condition_dim=cfg['model']['condition_dim']
    )

    return model


def build_optimizer(model, cfg):

    if cfg['train']['optimizer'] == 'sgd':
        decay_params = []
        no_decay_params = []

        # so-called "Tencent trick"
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': cfg['train']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.SGD(
            param_groups, 
            lr=cfg['train']['initial_lr'],
            momentum=cfg['train']['momentum']
        )

    elif cfg['train']['optimizer'] == 'adam':
        decay_params = []
        no_decay_params = []

        # similar to "Tencent trick", but also with rpb
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ("absolute_pos_emb" in name 
            or "rpb_table" in name
            or "norm" in name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': cfg['train']['weight_decay']}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg['train']['initial_lr'],
            weight_decay=cfg['train']['weight_decay']
        )
    
    else:
        raise NotImplementedError("No optimizer for such backbone name.")
    
    return optimizer


class TrainerDIFF:
    def __init__(self, model, train_loader, optimizer, noise_scheduler, cfg):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.epochs_run = 0
        self.batch_size = cfg['train']['batch_size']
        self.batch_interval = cfg['train']['batch_interval']
        self.num_epochs = cfg['train']['num_epochs']
        self.accumulation_steps = cfg['train']['accumulation_steps']
        self.save_every = cfg['train']['save_every']
        self.warmup_epochs = cfg['train']['warmup_epochs']
        self.snapshot_path = cfg['train']['snapshot_path']
        self.max_grad_norm = cfg['train']['max_grad_norm']
        self.sampling_interval = cfg['train']['sampling_interval']
        self.n_timesteps = cfg['train']['n_timesteps']
        self.image_size = cfg['model']['image_size']
        self.out_dim = cfg['model']['out_dim']

        # path to save generated samples
        self.samples_path = cfg['train']['samples_path'] + f"_gpu_{self.gpu_id}"
        if not os.path.exists(self.samples_path):
            os.mkdir(cfg['train']['samples_path'] + f"_gpu_{self.gpu_id}")

        if self.max_grad_norm is None:
            self.max_grad_norm = 1e10

        # whether to syncronise BN layers across all devices
        if cfg['model']['sync_batchnorm']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        self.model = model.to(self.gpu_id)
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot...")
            self._load_snapshot(self.snapshot_path)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.num_batches = len(self.train_loader)
        self.scheduler = self._build_scheduler(cfg, self.num_batches // self.accumulation_steps)
        
        self.stats = {
            "train_losses": np.empty(self.num_epochs),
            "grad_norms": np.empty(self.num_epochs)
        }

        # logging setup
        log_path = f"training_gpu_{self.gpu_id}.log"
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
    

    def _build_scheduler(self, cfg, n_iter_per_epoch):
        num_steps = cfg['train']['num_epochs'] * n_iter_per_epoch
        warmup_steps = cfg['train']['warmup_epochs'] * n_iter_per_epoch

        # lr_scheduler = MultiStepLRScheduler(
        #     self.optimizer, 
        #     decay_t=[ms * n_iter_per_epoch for ms in cfg['train']['lr_milestones']],     
        #     decay_rate=cfg['train']['lr_decay_rate'],         
        #     warmup_t=warmup_steps,             
        #     warmup_lr_init=cfg['train']['warmup_lr'],
        #     warmup_prefix=cfg['train']['warmup_prefix'],
        #     t_in_epochs=False 
        # )

        lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(num_steps - warmup_steps) if cfg['train']['warmup_prefix'] else num_steps,
            lr_min=cfg['train']['min_lr'],
            warmup_lr_init=cfg['train']['warmup_lr'],
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

        for i, (images, labels) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            
            noise = torch.randn_like(images)   # will be on the same device as images
            timesteps = torch.randint(0, self.n_timesteps, (images.shape[0],)).to(self.gpu_id)
            noisy_images = self.noise_scheduler.add_noise(images, timesteps, noise)
            pred_noise = self.model(noisy_images, timesteps, labels)  # note that we pass in the labels too
            loss = F.mse_loss(noise, pred_noise)                      # maybe swith to smooth_l1_loss
            
            if ((i + 1) % self.accumulation_steps == 0):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)   # grad clipping
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
    

    def train(self):
        start_time = time.time()
        for epoch in range(self.epochs_run, self.num_epochs):
            self.train_one_epoch(epoch)
            if ((epoch + 1) % self.sampling_interval == 0): 
                dist.barrier()
                samples = self.generate_samples()
                self.save_samples(samples, epoch)
                dist.barrier()
                logging.info(f"(GPU[{self.gpu_id}]): Samples generated & saved.")

            if (self.gpu_id == 0) and ((epoch + 1) % self.save_every == 0):
                self._save_snapshot(epoch)

            time_elapsed = self._seconds_to_hhmmss(time.time() - start_time)
            logging.info(f"(GPU[{self.gpu_id}]): Epoch {epoch} completed. Time elapsed: {time_elapsed}")
            print(f"(GPU[{self.gpu_id}]): Epoch {epoch} completed. Time elapsed: {time_elapsed}")
        
        # saving model weights (on one GPUs)
        if self.gpu_id == 0:
            logging.info(f"Saving model by GPU[{self.gpu_id}]...")
            torch.save(self.model.module.state_dict(), 'final_model.pth')

        # saving training stats (on all GPUs)
        with open(f'training_output_{self.gpu_id}.pkl', 'wb') as f:
            pickle.dump(self.stats, f)

        logging.info(f"Training for {self.num_epochs} epochs  is completely finished. Results saved.")
        print(f"Training for {self.num_epochs} epochs  is completely finished. Results saved.")
    

    @torch.no_grad()
    def generate_samples(self, n_samples=8, n_classes=10):
        x = torch.randn(n_samples * n_classes, self.out_dim, self.image_size, self.image_size).to(self.gpu_id)
        cond = torch.tensor([[i] * n_samples for i in range(n_classes)]).flatten().to(self.gpu_id)

        for t in reversed(range(0, self.noise_scheduler.timesteps)):
            t_tensor = torch.tensor([t]).to(self.gpu_id)
            with torch.no_grad():
                residual = self.model(x, t_tensor, cond)

            x = self.noise_scheduler.reduce_noise(
                residual, x,
                t_tensor, 
                t_index=t   # don't forget to pass t_index
            ) 
        
        x = (x + 1.0) / 2.0   # reverse to normalization in preprocessing
        
        return x
    

    # def save_samples(self, samples, epoch):
    #     fig, ax = plt.subplots(figsize=(12, 12))
    #     grid_img = make_grid(samples.detach().cpu().clip(-1, 1), nrow=8)[0]
    #     ax.imshow(grid_img, cmap="Greys")
    #     ax.axis('off')

    #     fig.savefig(self.samples_path + f"epoch_{epoch}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
    
    def save_samples(self, samples, epoch):
        fig, ax = plt.subplots(figsize=(12, 12))
        grid_img = make_grid(samples.detach().cpu().clip(0, 1), nrow=8, padding=5, pad_value=0.5)
        
        if grid_img.shape[0] == 1:
            img = grid_img.squeeze(0)
            ax.imshow(img, cmap='Greys')
        else:
            img = grid_img.permute(1, 2, 0)
            ax.imshow(img)

        ax.axis('off')
        fig.savefig(self.samples_path + f"/epoch_{epoch}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def main(cfg):                       
    ddp_setup()

    # sqrt lr scaling (better for diffusion) with reference to total effective bs of 32
    effective_bs = cfg['train']['batch_size'] * get_world_size() * cfg['train']['accumulation_steps']
    scale_factor = (effective_bs / 32.0) ** 0.5
    cfg['train']['initial_lr'] = cfg['train']['initial_lr'] * scale_factor
    cfg['train']['warmup_lr'] = cfg['train']['warmup_lr'] * scale_factor
    cfg['train']['min_lr'] = cfg['train']['min_lr'] * scale_factor
    
    train_loader = build_loader(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    noise_scheduler = DDPMScheduler(timesteps=cfg['train']['n_timesteps'])
    
    trainer = TrainerDIFF(model, train_loader, optimizer, noise_scheduler, cfg)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    main(cfg)
