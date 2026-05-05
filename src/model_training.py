"""
HAM10000 Model Training Script
------------------------------
This script performs fine-tuning on ResNet-50 and MobileNetV2 architectures
using the HAM10000 dataset. It adapts the final layers for 7-class 
dermatological classification.

Usage:
    python model_training.py --model resnet50
    python model_training.py --model mobilenet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import time
import copy
import argparse

# Import custom dataset loader
from dataset_loader import HAM10000Dataset, data_transforms

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=5):
    """
    Standard training loop with validation phase and best model saving.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save weights if validation accuracy improves[cite: 4, 7]
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def initialize_model(model_name, num_classes=7):
    """
    Initializes and adapts architectures for dermatological classification.
    """
    if model_name == 'resnet50':
        # Prepare ResNet-50[cite: 3, 4]
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet':
        # Prepare MobileNetV2[cite: 3, 7]
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError("Model not supported. Choose 'resnet50' or 'mobilenet'.")
    
    return model

if __name__ == '__main__':
    # Config[cite: 4, 7]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CSV_PATH = 'data/HAM10000_metadata.csv'
    IMAGES_DIR = 'data/images/'
    RANDOM_SEED = 42
    
    # 1. Load and Split Dataset (80% Train, 20% Val)[cite: 4, 7]
    full_dataset = HAM10000Dataset(csv_file=CSV_PATH, root_dir=IMAGES_DIR, transform=data_transforms)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': train_size, 'val': test_size}

    # 2. Train Models sequentially or via arguments
    for m_name in ['resnet50', 'mobilenet']:
        print(f"\n--- Training {m_name.upper()} ---")
        model = initialize_model(m_name).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low LR for fine-tuning[cite: 4, 7]

        best_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, DEVICE)
        
        # Save output weights[cite: 5, 7]
        torch.save(best_model.state_dict(), f'{m_name}_baseline.pth')
        print(f"Model saved as '{m_name}_baseline.pth'")
