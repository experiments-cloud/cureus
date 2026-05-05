"""
Robustness Evaluation Script: Visual Degradation Analysis
--------------------------------------------------------
This script evaluates the diagnostic performance of trained models under 
simulated real-world photographic conditions, including motion blur, 
low luminance, and sensor noise.

It ensures scientific rigor by using a strictly isolated test set (20%) 
with a fixed random seed to prevent data leakage.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter

# Import custom dataset loader
from dataset_loader import HAM10000Dataset, data_transforms

# --- 1. Custom Degradation Classes ---

class AddGaussianNoise(object):
    """Applies Gaussian noise directly to the image tensor to simulate sensor grain."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class DegradeBlur:
    """Applies Gaussian Blur to simulate kinetic hand tremor during capture."""
    def __init__(self, radius=2):
        self.radius = radius
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

class DegradeBrightness:
    """Simulates clinical photography under poor ambient lighting conditions."""
    def __init__(self, factor=0.5):
        self.factor = factor
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)

# --- 2. Evaluation Pipelines[cite: 9, 10] ---

degradation_pipelines = {
    'Baseline (Clean)': data_transforms,
    
    'Moderate_Blur': transforms.Compose([
        transforms.Resize((224, 224)),
        DegradeBlur(radius=2.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    
    'Low_Luminance': transforms.Compose([
        transforms.Resize((224, 224)),
        DegradeBrightness(factor=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    
    'Sensor_Noise': transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         AddGaussianNoise(0., 0.1),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# --- 3. Evaluation Logic ---

def evaluate_performance(model, dataloader, criterion, device):
    """Calculates loss and accuracy for a specific model and condition."""
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    return running_loss / total_samples, running_corrects.double() / total_samples

if __name__ == '__main__':
    # Config
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CSV_PATH = 'data/HAM10000_metadata.csv'
    IMAGES_DIR = 'data/images/'
    MODEL_PATH = 'resnet50_baseline.pth'
    RANDOM_SEED = 42
    
    print(f"Starting Vulnerability Experiment on: {DEVICE}")

    # Load Model Architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # Split Dataset (Strictly matching training split)[cite: 9]
    temp_dataset = HAM10000Dataset(csv_file=CSV_PATH, root_dir=IMAGES_DIR, transform=data_transforms)
    train_size = int(0.8 * len(temp_dataset))
    test_size = len(temp_dataset) - train_size
    
    criterion = nn.CrossEntropyLoss()

    for cond_name, transform in degradation_pipelines.items():
        # Re-create split with specific transform to avoid data contamination[cite: 9]
        eval_dataset = HAM10000Dataset(csv_file=CSV_PATH, root_dir=IMAGES_DIR, transform=transform)
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        _, test_dataset = random_split(eval_dataset, [train_size, test_size], generator=generator)
        
        eval_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        loss, acc = evaluate_performance(model, eval_loader, criterion, DEVICE)
        
        print(f"Condition: {cond_name:20} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")
