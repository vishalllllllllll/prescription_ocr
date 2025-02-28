import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 512)),  # Resize images
    transforms.RandomRotation(degrees=10),  # Random rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Define dataset class
class PrescriptionDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.labels_path = os.path.join(dataset_dir, "labels.csv")
        self.images_dir = os.path.join(dataset_dir, "images")
        self.data = pd.read_csv(self.labels_path)
        self.transform = transform

      # Extract all characters to build a char_map
        all_texts = ''.join(self.data.iloc[:, 1].astype(str))  # Get all labels as text
        unique_chars = sorted(set(all_texts))  # Unique characters in dataset
        self.char_map = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        self.char_map['<BLANK>'] = 0  # Add blank token for CTC loss
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Image file name
        label = self.data.iloc[idx, 1]  # Corresponding text
        
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, label  # Return image tensor and text label
