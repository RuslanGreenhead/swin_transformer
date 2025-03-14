import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.multiprocessing as mp

import os
import numpy as np
import pickle
import yaml

from model import SwinTransformer
from data import build_loaders                  # WHOLE config to be put here

CONFIG_PATH = "configs/SwinT_300e_1024bs.yaml"

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


class TrainerCLF:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, cfg):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = cfg['batch_size']
        self.batch_interval = cfg['batch_interval']
        self.num_epochs = cfg['num_epochs']
        self.save_every = cfg['save_every']
        self.epochs_run = 0
        self.warmup_epochs = cfg['warmup_epochs']
        self.snapshot_path = cfg['snapshot_path']

        self.model = model.to(self.gpu_id)
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot...")
            self._load_snapshot(self.snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.scheduler = self._build_scheduler()
        self.num_batches = len(self.train_loader)
        
        self.stats = {
            "train_losses": np.empty(self.num_epochs),
            "train_accs": np.empty(self.num_epochs),
            "val_losses": np.empty(self.num_epochs),
            "val_accs": np.empty(self.num_epochs),
            "grad_norms": np.empty(self.num_epochs)
        }


    def _build_scheduler(self):
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs - self.warmup_epochs, eta_min=1e-6)
        chained_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[3])
        
        return chained_scheduler
    

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}.")
    

    def _save_snapshot(self, epoch):    # --> self.stats shall be added!
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}.")
    

    def train_one_epoch(self, epoch):
        self.model.train()
        interval_loss = 0.0
        interval_accuracy = 0
        total_loss = 0.0
        total_accuracy = 0
        batch_counter = 0

        print(f"(GPU[{self.gpu_id}]) Epoch {epoch} started.")

        self.train_loader.sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
            
            self.optimizer.zero_grad()
            outputs = self.model(images.permute(0, 2, 3, 1))
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(outputs, dim=1)
            interval_loss += loss.item()
            total_loss += interval_loss
            interval_accuracy += ((preds == labels).sum() / len(labels)).item()
            total_accuracy += interval_accuracy
            batch_counter += 1
            
            # stamping batch interval stats
            if ((i + 1) % self.batch_interval == 0) or ((i + 1) == self.num_batches):
                avg_interval_loss = interval_loss / batch_counter
                avg_interval_accuracy = interval_accuracy / batch_counter
                first_layer = list(self.model.module.parameters())[0]    # tracking grad norm on first layer
                grad_norm = torch.norm(first_layer.grad).item() 
                
                # printing stats
                print(f'GPU[{self.gpu_id}]: ' +
                      f'batch [{i+1}/{self.num_batches}]: ' +
                      f'loss={avg_interval_loss:.6f}' +
                      f'accuracy={avg_interval_accuracy:.6f}' +
                      f'grad_norm={grad_norm:.6f}', flush=True)
                
                interval_loss = 0.0
                interval_accuracy = 0
                batch_counter = 0    
 
        avg_loss = total_loss / self.num_batches
        avg_accuracy = total_accuracy / self.num_batches  
        first_layer = list(self.model.module.parameters())[0]    # tracking grad norm on first layer
        grad_norm = torch.norm(first_layer.grad).item() 

        # writing stats
        self.stats['train_losses'][epoch] = avg_loss
        self.stats['train_accs'][epoch] = avg_accuracy
        self.stats['grad_norms'][epoch] = grad_norm


    def validate(self, epoch):
        self.model.eval()
        total_correct = 0
        total_loss = 0

        self.val_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
                
                outputs = self.model(images.permute(0, 2, 3, 1))
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, dim=1)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
            
        val_accuracy = total_correct / len(self.val_loader.dataset)
        val_loss = total_loss / len(self.val_loader.dataset)
            
        # writing stats
        self.stats['val_losses'][epoch] = val_loss
        self.stats['val_accs'][epoch] = val_accuracy

        # printing stats
        print(f'Validation on GPU[{self.gpu_id}]:' +
              f'val_loss={val_loss:.6f}' +
              f'val_accuracy={val_accuracy:.6f}', flush=True)
    

    def train(self):
        for epoch in range(self.epochs_run, self.num_epochs):
            self.train_one_epoch(epoch)
            if self.gpu_id == 0:      # run validation (on one GPU)
                self.validate(epoch)
            self.scheduler.step()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        
        # saving stats + model weights (on one GPU)
        if self.gpu_id == 0:
            print(f"Saving model and stats by GPU[{self.gpu_id}]...")
            torch.save(self.model.module.state_dict(), 'final_model.pth')
            with open('training_output.pkl', 'wb') as f:
                pickle.dump(self.stats, f)

        print(f"Training for {self.num_epochs} is completely finished. Results saved.")


def main(cfg):                       
    ddp_setup()
    
    train_loader, val_loader = build_loaders(cfg)
    model = SwinTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    trainer = TrainerCLF(model, train_loader, val_loader, optimizer, criterion, cfg['training'])
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    main(cfg)