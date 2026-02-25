import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from elpv_dataset.utils import load_dataset
import numpy as np
from PIL import Image

# Load dataset
images, proba, types = load_dataset()

transform = T.Compose([
    T.Resize((32, 32)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
])

class ELPVDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.label_map = {
            "poly": 0,
            "mono": 1
            # add more classes if needed
        }

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label_str = self.labels[idx]
        label = self.label_map[label_str]  # 🔥 FIXED

        return img, label

# Make the dataset
dataset = ELPVDataset(images, types, transform)

# Split
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=64)
