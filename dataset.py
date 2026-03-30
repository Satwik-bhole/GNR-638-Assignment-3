"""
Dataset utilities for training PSPNet on a toy subset of PASCAL VOC 2012.

We use a small subset of VOC to keep training fast while still demonstrating
that the model learns meaningful segmentation. The paper uses full datasets
like ADE20K and Cityscapes — here we scale down for practical assignment use.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from PIL import Image, ImageFilter
import random


# PASCAL VOC 2012 has 21 classes (including background)
VOC_NUM_CLASSES = 21
IGNORE_LABEL = 255  # label used by VOC for boundary / don't-care pixels


class VOCSegDataset(Dataset):
    """
    A thin wrapper around torchvision's VOCSegmentation that:
      - takes only a small subset (toy dataset) for quick experiments
      - applies training-time augmentations similar to the paper
        (random scale, random horizontal flip, random crop, color jitter)
      - properly preserves the ignore label (255) for boundary pixels
    """

    def __init__(self, root, image_set="train", download=True, num_samples=50,
                 crop_size=473, is_train=True):
        self.voc = VOCSegmentation(root=root, year="2012",
                                   image_set=image_set, download=download)
        self.num_samples = min(num_samples, len(self.voc))
        self.crop_size = crop_size
        self.is_train = is_train

        # standard ImageNet normalization (same as the paper's preprocessing)
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, mask = self.voc[idx]

        if self.is_train:
            img, mask = self._train_augment(img, mask)
        else:
            # during validation just resize to a fixed size
            img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        img = self.img_normalize(img)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

    # ------------------------------------------------------------------
    # Augmentation pipeline (simplified version of what the paper uses)
    # ------------------------------------------------------------------
    def _train_augment(self, img, mask):
        """
        Paper-inspired augmentations:
          1. Random scaling between 0.5x and 2.0x
          2. Random horizontal flip
          3. Random Gaussian blur
          4. Random crop to crop_size x crop_size
        """
        # random scale between 0.5 and 2.0
        scale = random.uniform(0.5, 2.0)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random brightness / contrast jitter (our own addition for extra variety)
        if random.random() > 0.5:
            from PIL import ImageEnhance
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # random Gaussian blur (paper applies this occasionally)
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # pad if smaller than crop_size, then random crop
        img, mask = self._pad_and_crop(img, mask)

        return img, mask

    def _pad_and_crop(self, img, mask):
        """Pad the image/mask with mean color / ignore label, then randomly crop."""
        w, h = img.size
        pad_w = max(self.crop_size - w, 0)
        pad_h = max(self.crop_size - h, 0)

        if pad_w > 0 or pad_h > 0:
            # pad image with ImageNet mean (roughly [124, 116, 104] in 0-255 scale)
            img = np.array(img, dtype=np.uint8)
            img = np.pad(img,
                         ((0, pad_h), (0, pad_w), (0, 0)),
                         mode="constant", constant_values=0)
            img = Image.fromarray(img)

            # pad mask with ignore label so those pixels don't affect the loss
            mask = np.array(mask, dtype=np.uint8)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                          mode="constant", constant_values=IGNORE_LABEL)
            mask = Image.fromarray(mask)

        w, h = img.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        mask = mask.crop((x, y, x + self.crop_size, y + self.crop_size))

        return img, mask


def get_dataloaders(root="./data", batch_size=4, num_train=200, num_val=50,
                    crop_size=473, num_workers=2):
    """
    Build train and validation DataLoaders for the toy VOC subset.

    Args:
        root:       where to download / find PASCAL VOC
        batch_size: images per batch
        num_train:  how many training images to use
        num_val:    how many validation images to use
        crop_size:  spatial size for training crops
        num_workers: dataloader workers
    Returns:
        (train_loader, val_loader)
    """
    train_ds = VOCSegDataset(root, image_set="train", download=True,
                             num_samples=num_train, crop_size=crop_size,
                             is_train=True)
    val_ds = VOCSegDataset(root, image_set="val", download=True,
                           num_samples=num_val, crop_size=crop_size,
                           is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    print("Downloading and preparing toy dataset...")
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    for imgs, masks in train_loader:
        print(f"Image batch: {imgs.shape}, Mask batch: {masks.shape}")
        print(f"Mask unique values: {torch.unique(masks).tolist()}")
        break
