import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from elpv_dataset.utils import load_dataset
from PIL import Image
import numpy as np


# -------------------------------
# Load Dataset (Numpy arrays)
# -------------------------------
images, proba, types = load_dataset()


# -------------------------------
# Transforms for DARTS (32×32 RGB)
# -------------------------------
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),  # PIL → Tensor    
])


# -------------------------------
# Custom Dataset
# -------------------------------
class ELPVDataset(Dataset):
    def __init__(self, images, labels, transform=None):

        self.images = images            # numpy arrays (H,W or H,W,3)
        self.labels = labels            # string labels "mono"/"poly"
        self.transform = transform

        # label mapping for DARTS
        self.label_map = {
            "poly": 0,
            "mono": 1
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # numpy array → ensure 3 channels
        img_arr = self.images[idx]

        # Some images are grayscale (H×W), convert to (H×W×3)
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)

        img = Image.fromarray(img_arr.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        # Convert label string → int
        label_str = self.labels[idx]
        label = self.label_map[label_str]

        return img, label


# -------------------------------
# Build Dataloaders
# -------------------------------
dataset = ELPVDataset(images, types, transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
