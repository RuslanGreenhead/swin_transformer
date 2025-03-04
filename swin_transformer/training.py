import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np


class SwinTrainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, criterion, batch_size, batch_interval,
                 num_epochs, warmup_epochs, num_gpus):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.num_gpus = num_gpus
        
        self._setup_ddp()
        self.scheduler = self._build_scheduler()
        
        self.num_batches = len(self.train_loader)
        self.train_stats = np.zeros((self.num_epochs * (self.num_batches // 2000 + 1), 4))
        self.val_stats = np.zeros((self.num_epochs, 2))
        self.grad_norm_stats = np.zeros((self.num_epochs * (self.num_batches // 2000 + 1), 2))
        

    def _setup_ddp(self):
        dist.init_process_group('nccl', init_method='env://')
        self.model = DDP(self.model, device_ids=[dist.get_rank()])


    def _build_scheduler(self):
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.001, end_factor=1.0, total_iters=self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs - self.warmup_epochs, eta_min=1e-6)
        chained_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[20])
        
        return chained_scheduler
    

    def train_one_epoch(self, epoch):
        self.model.train()
        interval_loss = 0.0
        interval_accuracy = 0
        total_loss = 0.0
        total_accuracy = 0
        batch_counter = 0

        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(dist.get_rank()), labels.to(dist.get_rank())
            
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
                
                print(f'batch [{i+1}/{self.num_batches}]: ' + 
                      f'train loss={avg_interval_loss:.6f}' +
                      f'train acc={avg_interval_accuracy:.6f}', flush=True)
                
                interval_loss = 0.0
                interval_accuracy = 0
                batch_counter = 0    
 
        avg_loss = total_loss / self.num_batches
        avg_accuracy = total_accuracy / self.num_batches  
        first_layer = list(self.model.module.parameters())[0]    # tracking grad norm on first layer
        grad_norm = torch.norm(first_layer.grad).item() 

        return avg_loss, avg_accuracy, grad_norm


    def validate(self, epoch):
        self.model.eval()
        total_correct = 0
        total_loss = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(dist.get_rank()), labels.to(dist.get_rank())
                
                outputs = self.model(images.permute(0, 2, 3, 1))
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, dim=1)
                total_loss += loss.item()
                total_correct += (predicted == labels).sum().item()
            
            val_accuracy = total_correct / len(self.val_loader.dataset)
            val_loss = total_loss / len(self.val_loader.dataset)
            
        return val_loss, val_accuracy
    

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            self.scheduler(epoch)
        
        # saving stats
        np.save('train_stats.npy', self.train_stats)
        np.save('val_stats.npy', self.val_stats)
        np.save('grad_norm_stats.npy', self.grad_norm_stats)
        
        dist.destroy_process_group()

def main(rank, num_gpus):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageNet(root='./data/imagenet', split='train', transform=transform)
    val_dataset = ImageNet(root='./data/imagenet', split='val', transform=transform)
    
    # Create distributed data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, sampler=val_sampler)
    
    # Initialize model
    model = SwinTransformer(weights=None)
    model.head = nn.Linear(model.head.in_features, 1000)  # Adjust head for ImageNet
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    trainer = SwinTrainer(model, train_loader, val_loader, optimizer,
                          batch_size=128, num_epochs=300, warmup_epochs=20,
                          num_gpus=num_gpus)
    trainer.train()


if __name__ == "__main__":
    num_gpus = 4
    mp.spawn(main, args=(num_gpus,), nprocs=num_gpus)