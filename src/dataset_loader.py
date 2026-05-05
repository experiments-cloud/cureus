"""
HAM10000 Dataset Loader for PyTorch
-----------------------------------
This module provides a robust data loading pipeline for the HAM10000 dataset,
standardizing the 7 skin lesion categories and applying necessary transforms
for deep learning architectures like ResNet-50 and MobileNetV2.

Author: [Tu Nombre/Laboratorio]
Date: 2026
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Mapeo oficial de las 7 patologías del dataset HAM10000
LESION_TYPE_DICT = {
    'nv': 0,      # Melanocytic nevi
    'mel': 1,     # Melanoma
    'bkl': 2,     # Benign keratosis-like lesions 
    'bcc': 3,     # Basal cell carcinoma
    'akiec': 4,   # Actinic keratoses
    'vasc': 5,    # Vascular lesions
    'df': 6       # Dermatofibroma
}

class HAM10000Dataset(Dataset):
    """
    Custom Dataset class for loading HAM10000 skin lesion images and metadata.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the metadata CSV file.
            root_dir (string): Directory containing all dataset images.
            transform (callable, optional): PyTorch transforms to apply to images[cite: 2, 8].
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Metadata file not found at: {csv_file}")
            
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping diagnostic labels to numeric targets[cite: 2, 8]
        self.metadata['target'] = self.metadata['dx'].map(LESION_TYPE_DICT)

    def __len__(self):
        """Returns the total number of samples in the dataset[cite: 2, 8]."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.
        
        Returns:
            image (Tensor): Transformed image data.
            label (int): Numerical label for the lesion type.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image file construction (e.g., ISIC_0027419.jpg)[cite: 2, 8]
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx]['image_id'] + '.jpg')
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise IOError(f"Could not open image {img_name}: {e}")
        
        label = int(self.metadata.iloc[idx]['target'])

        if self.transform:
            image = self.transform(image)

        return image, label

# Standard ImageNet transforms required for ResNet and MobileNet
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloader(csv_path, images_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Helper function to initialize the Dataset and DataLoader.[cite: 2, 8]
    """
    dataset = HAM10000Dataset(csv_file=csv_path, root_dir=images_dir, transform=data_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    # Example usage and sanity check
    CSV_PATH = 'data/HAM10000_metadata.csv'
    IMAGES_DIR = 'data/images/'
    
    print("--- HAM10000 Dataset Loader Test ---")
    try:
        loader = get_dataloader(CSV_PATH, IMAGES_DIR)
        images, labels = next(iter(loader))
        
        print(f"Status: Success!")
        print(f"Batch Image Shape: {images.shape}")  # Should be [32, 3, 224, 224][cite: 2, 8]
        print(f"Batch Label Shape: {labels.shape}")
        print(f"Sample Targets: {labels[:5].tolist()}")
        
    except Exception as error:
        print(f"Status: Failed")
        print(f"Error Details: {error}")
        print("\nPlease ensure your 'data/' folder structure follows the repository guidelines.")