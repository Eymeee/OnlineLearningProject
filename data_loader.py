import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None, target_type='smile', weights=None):
        self.img_dir = img_dir
        self.attrs = pd.read_csv(attr_file)
        self.transform = transform
        self.target_type = target_type
        self.weights = weights
        self.transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
        
        # Map target for classification (smiling: 1, not smiling: 0)
        if target_type == 'smile':
            self.attrs['target'] = (self.attrs['Smiling'] == 1).astype(float)
        elif target_type == 'regression':
            # Weighted sum
            self.attrs['target'] = 0
            for idx, col in enumerate(self.attrs.columns[1:-1]):  # Skip filename & target
                w = weights[idx] if weights else 1.0
                self.attrs['target'] += w * self.attrs[col]
        else:
            raise ValueError("target_type must be 'smile' or 'regression'")

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.attrs.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(self.attrs.iloc[idx]['target'], dtype=torch.float)
        return image, target
    






