import os
import torch
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class ToyVOCDataset(Dataset):
    def __init__(self, root, year='2012', image_set='train', download=True, num_samples=50, transform=None):
        self.voc = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.num_samples = min(num_samples, len(self.voc))
        self.transform = transform
        
        # PASCAL VOC colors: 21 classes (including background) + index 255 for ignoring
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img, mask = self.voc[idx]
        
        # Resize to fixed size
        img = img.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        if self.transform:
            img = self.transform(img)
            
        mask = torch.from_numpy(np.array(mask)).long()
        # Replace 255 with 0 (background) for simplicity in toy dataset, 
        # normally we'd use ignore_index in CrossEntropyLoss
        mask[mask == 255] = 0
            
        return img, mask

def get_dataloaders(root='./data', batch_size=4, num_train=40, num_val=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ToyVOCDataset(root, image_set='train', download=True, num_samples=num_train, transform=transform)
    val_dataset = ToyVOCDataset(root, image_set='val', download=True, num_samples=num_val, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == '__main__':
    print("Downloading and preparing toy dataset...")
    train_loader, val_loader = get_dataloaders()
    print(f"Num train batches: {len(train_loader)}, Num val batches: {len(val_loader)}")
    for imgs, masks in train_loader:
        print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
        break
