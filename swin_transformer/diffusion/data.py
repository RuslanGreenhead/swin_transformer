import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader


class DiffusionDataset(Dataset):
    def __init__(self, saved_data, transform=None):
        self.images = saved_data['images']
        self.labels = saved_data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)

        return img, label
    

def build_loader(cfg):
    if cfg['dataset']['name'] == "MNIST":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) 
        ])
    elif cfg['dataset']['name'] == "CIFAR-10":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),                         # converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])      # scales to [-1, 1]
        ])
    else:
        transform = None
    
    saved_data = torch.load(cfg['dataset']['root'])
    dataset = DiffusionDataset(saved_data, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    return loader
    
