import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import pickle

from model import SwinTransformer

DATA_ROOT = "/opt/software/datasets/LSVRC/imagenet" 


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Training model
        + train loss for each 2000 batches
        + train loss, train accuracy for each epoch
        + val loss, val accuracy for each epoch

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        dict: dictionary with losses and accuracies.
    """

    train_losses_per_2000b = []
    train_losses_per_epoch = []
    train_accs_per_epoch = []
    val_losses_per_epoch = []
    val_accs_per_epoch = []
    batch_interval = 1000
    model = model.to(device)

    # lookinto_dict = {}

    for epoch in range(num_epochs):

        # training stage
        model.train()
        train_loss_epoch = 0.0
        train_loss_2000b = 0.0
        correct = 0
        total_samples_2000b = 0
        total_samples_epoch = 0
        total_batches = len(train_loader)

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images.permute(0, 2, 3, 1))    # -> to (B, H, W, C)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            """ lookinto_dict[f"batch_{i}"] = {
                "imgs": images,
                "labels": labels,
                "outputs": outputs
            } """

            train_loss_2000b += loss.item()
            train_loss_epoch += loss.item()
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels).sum().item()
            total_samples_2000b += labels.size(0)
            total_samples_epoch += labels.size(0)

            # stamping frequent train stats
            if ((i + 1) % batch_interval == 0) or ((i + 1) == total_batches):
                train_losses_per_2000b.append(train_loss_2000b / total_samples_2000b)
                print(f'batch [{i+1}/{total_batches}]: ' + 
                      f'train loss={train_losses_per_2000b[-1]:.6f}', flush=True)
                train_loss_2000b = 0.0
                total_samples_2000b = 0
        
        train_losses_per_epoch.append(train_loss_epoch / total_samples_epoch)
        train_accs_per_epoch.append(correct / total_samples_epoch)

        # validation stage
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images.permute(0, 2, 3, 1))
                loss = criterion(val_outputs, val_labels)

                val_loss += loss.item()
                _, val_preds = torch.max(val_outputs.data, 1)
                correct_val += (val_preds == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss /= total_val_samples # or len(val_loader.dataset) devide by num of validation samples
        val_accuracy = correct_val / total_val_samples
        val_losses_per_epoch.append(val_loss)
        val_accs_per_epoch.append(val_accuracy)
            
        print(f'Epoch [{epoch+1}/{num_epochs}] completed:', flush=True)
        print(f'val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}', flush=True)

    """ with open('lookinto_dict.pkl', 'wb') as f:
        pickle.dump(lookinto_dict, f)
     """
    output = {
        "train_losses_per_2000b": train_losses_per_2000b,
        "train_losses_per_epoch": train_losses_per_epoch,
        "train_accs_per_epoch": train_accs_per_epoch,
        "val_losses_per_epoch": val_losses_per_epoch,
        "val_accs_per_epoch": val_accs_per_epoch
    }

    return output


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet standard transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=DATA_ROOT + "/train", transform=transform)
    val_dataset = ImageFolder(root=DATA_ROOT + "/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = SwinTransformer()    # default is (Swin-T)

    num_epochs = 50
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    

    # training
    output = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # saving trained model and statistics
    torch.save(model.state_dict(), 'basic_model.pth')
    with open('training_output.pkl', 'wb') as f:
        pickle.dump(output, f)

    print("Trainig finished, results saved.")
