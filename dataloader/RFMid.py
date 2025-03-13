import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class RFMiD(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform
        print(f"DEBUG: Dataset initialized with Image Dir -> {self.image_dir}")

        # Load CSV data
        data = pd.read_csv(label_dir)

        if data.shape[1] < 2:
            raise ValueError(f" Error: CSV file {label_dir} should have at least 2 columns (image ID, label).")

        self.image_all = data.iloc[:, 0].astype(str).values  
        self.label_all = data.iloc[:, 1].values  
        unique_labels = np.unique(self.label_all)
        print(f"DEBUG: Found {len(unique_labels)} unique labels -> {unique_labels}")

    def __getitem__(self, idx):

        image_name = f"{self.image_all[idx]}.png"
        label = self.label_all[idx]

        image_path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Using a black placeholder.")
            x = Image.new("RGB", (512, 512), (0, 0, 0))  
        else:
            x = Image.open(image_path).convert("RGB")

        print(f"DEBUG: Loaded Image -> {image_path}, Label -> {label}")

        if self.transform:
            x = self.transform(x)
        print(f"DEBUG------: Image shape after transform -> {x.shape}")
        num_classes = 2  
        label_onehot = np.zeros(num_classes, dtype=np.float32)

        if 0 <= label < num_classes:
            label_onehot[label] = 1
        else:
            print(f"Warning: Invalid label '{label}' at index {idx}. Skipping label one-hot encoding.")

        return x, torch.tensor(label_onehot, dtype=torch.float32), label

    def get_labels(self):
        return self.label_all

    def __len__(self):
        return len(self.label_all)

