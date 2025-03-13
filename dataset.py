import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class RFMiD(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        print(f"DEBUG: Dataset initialized with Image Dir -> {self.image_dir}")

        # Load CSV data
        data = pd.read_csv(label_dir)
        self.image_all = data.iloc[:, 0].values  # First column: image IDs
        self.label_all = data.iloc[:, 1].values  # Second column: labels

    def __getitem__(self, idx):
        image_name = str(self.image_all[idx]) + ".png"
        label = self.label_all[idx]

        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        print(f"DEBUG: Looking for Image -> {image_path}")

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found! Skipping index {idx}.")
            return None  

        x = Image.open(image_path)  
        print(f"DEBUG: Image shape before transform: {np.array(x).shape}")

        if self.transform:
            x = self.transform(x)

        print(f"DEBUG: Image shape after transform: {x.shape}")

        label_onehot = np.zeros(2)
        if 0 <= label < len(label_onehot):
            label_onehot[label] = 1
        else:
            print(f"Warning: Invalid label '{label}' at index {idx}")

        return x, label_onehot, label

    def get_labels(self):
        return self.label_all

    def __len__(self):
        return len(self.label_all)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    print(f" Building dataset: {args.data_set}")

    if args.data_set == "rfmid":
        image_dir = args.train_images if is_train else args.val_images
        label_file = args.train_labels if is_train else args.val_labels

        print(f"DEBUG: Dataset initialized. Looking for images in: {image_dir}")
        print(f"DEBUG: Using labels from: {label_file}")

        dataset = RFMiD(image_dir=image_dir, label_dir=label_file, transform=transform)
        nb_classes = 2  

    else:
        raise NotImplementedError(f"❌ Dataset {args.data_set} not implemented.")

    print(f"Number of classes: {nb_classes}")
    return dataset, nb_classes


def build_transform(is_train, args):

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

