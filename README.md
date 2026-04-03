# GNR 638 — Assignment 3: PSPNet from Scratch

**Paper:** Zhao et al., *"Pyramid Scene Parsing Network"*, CVPR 2017  
**Student:** Satwik Bhole

---

## Overview

This repository contains a **from-scratch PyTorch implementation** of PSPNet and a comparison against the [official implementation provided by hszhao](https://github.com/hszhao/semseg). Both models are trained on a **toy subset of PASCAL VOC 2012** and compared on loss, pixel accuracy, and mean IoU.

## Repository Structure

```
├── pspnet_scratch.py   # From-scratch PSPNet (Dilated ResNet-50 + PPM + Aux loss)
├── dataset.py          # PASCAL VOC 2012 toy dataset with augmentations
├── train.py            # Training loop, auto-clones official repo, evaluates and plots
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Key Architecture Details (from the paper)

| Component | Detail |
|---|---|
| **Backbone** | Dilated ResNet-50 with deep stem (three 3×3 convs) + dilated layer3 & layer4 (output stride = 8) |
| **Pyramid Pooling Module** | Adaptive average pooling at bin sizes (1, 2, 3, 6), each reduced to 512 channels, then upsampled and concatenated → 4096 channels |
| **Classifier** | Conv 3×3 (4096→512) → BN → ReLU → Dropout(0.1) → Conv 1×1 (512→C) |
| **Auxiliary Loss** | Attached to layer3 output; weight = 0.4; same head structure (1024→256→C) |
| **Optimizer** | SGD, momentum=0.9, weight_decay=1e-4 |
| **LR Schedule** | Poly decay: `lr = base_lr × (1 − iter/max_iter)^0.9` |
| **Differential LR** | Backbone at base_lr, new heads at 10× base_lr |
| **Crop Size** | 257×257 (Modified to fit 6GB VRAM and satisfy (x-1)%8==0 constraint) |
| **Augmentations** | Random scale [0.5, 2.0], horizontal flip, brightness jitter, Gaussian blur, random crop |
| **Multi-scale Test** | Scales [0.75, 1.0, 1.25] with horizontal flip |
| **Metrics** | Mean IoU (primary), pixel accuracy, loss |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train both models and generate comparison

```bash
python train.py
```

This will:
- Automatically clone the official `hszhao/semseg` repository if not present (`../semseg`)
- Download PASCAL VOC 2012 (on first run)
- Train the scratch PSPNet for 15 epochs on 257×257 crops
- Train the official hszhao PSPNet for 15 epochs on the same data
- Display interactive `tqdm` progress bars for training, validation, and multi-scale inference
- Run multi-scale inference (scales 0.75, 1.0, 1.25 + flip) on both models
- Save comparison plots to `comparison_plots.png`
- Save qualitative predictions to `qualitative_results.png`
- Print a final metrics comparison table (single-scale + multi-scale)
- Save model checkpoints to `checkpoints/`

### 3. Run the model smoke test

```bash
python pspnet_scratch.py
```

### 4. Test the dataset loading

```bash
python dataset.py
```

## Outputs

| File | Description |
|---|---|
| `comparison_plots.png` | 2×2 grid: train loss, val loss, pixel accuracy, mIoU over epochs |
| `qualitative_results.png` | Side-by-side: input, ground truth, scratch prediction, official prediction |
| `checkpoints/` | Saved model weights for both models |

## Scratch vs Official — What's Compared

| Aspect | Scratch | Official (hszhao) |
|---|---|---|
| Backbone | Dilated ResNet-50 with deep stem (ImageNet pretrained stages) | Same (Dilated ResNet-50) |
| Stem | Three 3×3 convs (paper's modified ResNet) | Same (Three 3x3 convs) |
| PPM | Custom MultiScalePooling module | Official PPM module |
| Auxiliary loss | Yes (0.4 weight) | Yes (0.4 weight) |
| Crop size | 257×257 | 257×257 (Matches model dimension assertions) |
| Multi-scale test | Yes (0.75, 1.0, 1.25 + flip) | Yes |
| Optimizer | SGD + Poly LR (10× for heads) | SGD + Poly LR |
| Dataset | VOC 2012 subset (500 train / 100 val) | Same |

## Notes

- We use a subset of VOC (500 train / 100 val) and 15 epochs to keep training practical on local hardware with 6GB VRAM.
- The deep stem in `pspnet_scratch.py` mimics `hszhao`'s codebase. Instead of a 1x1 adapter, it structurally adjusts the first bottleneck block of `layer1` to natively accept 128 input channels.
- Validation loops employ dynamic interpolation for inputs and target masks to guarantee the official implementation's strict `(x-1) % 8 == 0` dimension checks remain satisfied when arbitrary-sized streaming test batches are evaluated.
- All comparison is done on the exact same data splits with identical hyperparameters, and visual progress is tracked seamlessly.
