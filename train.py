"""
Compare our custom PSPNet against hszhao's official implementation.
Trains both on a subset of VOC and plots the results.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works on servers
import matplotlib.pyplot as plt
import sys
import subprocess

semseg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../semseg')
if not os.path.exists(semseg_dir):
    print(f"Cloning official hszhao/semseg repository to {semseg_dir}...")
    subprocess.run(["git", "clone", "https://github.com/hszhao/semseg.git", semseg_dir], check=True)

sys.path.append(semseg_dir)
from model.pspnet import PSPNet as HSZhaoPSPNet

from dataset import get_dataloaders, IGNORE_LABEL, VOC_NUM_CLASSES
from pspnet_scratch import PSPNetScratch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_miou(preds, targets, num_classes, ignore_index=255):
    """Computes mean intersection over union (mIoU) for valid pixels."""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    # mask out ignore pixels
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def compute_pixel_accuracy(preds, targets, ignore_index=255):
    """fraction of correctly classified valid pixels."""
    valid = targets != ignore_index
    correct = (preds[valid] == targets[valid]).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Poly learning rate schedule (from the paper)
# ---------------------------------------------------------------------------

def make_poly_scheduler(optimizer, total_iters, power=0.9):
    """Poly LR schedule: lr = base * (1 - iter/max)^0.9"""
    def _poly_lambda(current_step):
        return (1.0 - current_step / total_iters) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=10, base_lr=0.01,
                device="cpu", name="model", is_scratch=False):
    """Basic train/val loop handling both models' specific structures."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # separate the pretrained backbone parameters from the freshly-initialised
    # heads — the paper trains heads at 10x the backbone learning rate
    if is_scratch:
        new_layer_keywords = ["pyramid_pool", "seg_head", "aux_head",
                               "stem", "stem_adapter"]
        backbone_params, head_params = [], []
        for pname, p in model.named_parameters():
            if any(kw in pname for kw in new_layer_keywords):
                head_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params,     "lr": base_lr * 10},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    # the paper uses SGD with momentum and L2 regularisation
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)

    # smooth poly-decay schedule over the entire training run
    total_iters = epochs * len(train_loader)
    scheduler = make_poly_scheduler(optimizer, total_iters, power=0.9)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_miou": []}

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Epochs: {epochs}  |  LR: {base_lr}  |  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # ---- training phase ----
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs} Train")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            if hasattr(model, 'zoom_factor') and model.training:
                # Official hszhao PSPNet
                B, C, H, W = images.shape
                sh = ((H - 1) // 8) * 8 + 1
                sw = ((W - 1) // 8) * 8 + 1
                if sh != H or sw != W:
                    # Resize inputs and targets for strict compatibility
                    im_train = F.interpolate(images, size=(sh, sw), mode="bilinear", align_corners=True)
                    # Float then nearest interpolate for int targets
                    m_flt = masks.unsqueeze(1).float()
                    m_scaled = F.interpolate(m_flt, size=(sh, sw), mode="nearest").squeeze(1).long()
                else:
                    im_train, m_scaled = images, masks
                
                preds, main_loss, aux_loss = model(im_train, m_scaled)
                loss = main_loss + 0.4 * aux_loss
            else:
                outputs = model(images)

                # handle auxiliary output from our scratch model
                if isinstance(outputs, tuple):
                    main_out, aux_out = outputs
                    loss = criterion(main_out, masks) + 0.4 * criterion(aux_out, masks)
                else:
                    loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ---- validation phase ----
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                if hasattr(model, 'zoom_factor'):
                    # hszhao implementation requires shape matching (x-1) % 8 == 0
                    B, C, H, W = images.shape
                    sh = ((H - 1) // 8) * 8 + 1
                    sw = ((W - 1) // 8) * 8 + 1
                    if sh != H or sw != W:
                        images = F.interpolate(images, size=(sh, sw), mode="bilinear", align_corners=True)
                        outputs = model(images)
                        outputs = F.interpolate(outputs, size=(H, W), mode="bilinear", align_corners=True)
                    else:
                        outputs = model(images)
                else:
                    outputs = model(images)
                    
                val_loss_sum += criterion(outputs, masks).item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = val_loss_sum / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_acc = compute_pixel_accuracy(all_preds, all_targets, IGNORE_LABEL)
        val_miou = compute_miou(all_preds, all_targets, VOC_NUM_CLASSES, IGNORE_LABEL)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_miou"].append(val_miou)

        print(f"  Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  |  "
              f"Pixel Acc: {val_acc:.4f}  |  "
              f"mIoU: {val_miou:.4f}")

    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(hist_scratch, hist_official, epochs, save_path="comparison_plots.png"):
    """Generates the 2x2 metrics plot."""
    ep = range(1, epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = ["Training Loss", "Validation Loss", "Pixel Accuracy", "Mean IoU"]
    keys = ["train_loss", "val_loss", "val_acc", "val_miou"]
    ylabels = ["Loss", "Loss", "Accuracy", "mIoU"]

    for ax, title, key, ylabel in zip(axes.flat, titles, keys, ylabels):
        ax.plot(ep, hist_scratch[key], "o-", label="Scratch PSPNet", linewidth=2)
        ax.plot(ep, hist_official[key], "s--", label="Official (hszhao) PSPNet", linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("PSPNet: Scratch vs Official Implementation Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plots to '{save_path}'")
    plt.close()


def visualize_predictions(model_scratch, model_official, val_loader, device,
                          num_samples=4, save_path="qualitative_results.png"):
    """Plots side-by-side segmentation visuals."""
    model_scratch.eval()
    model_official.eval()

    # grab one batch
    images, masks = next(iter(val_loader))
    images_dev = images.to(device)
    num_samples = min(num_samples, images.size(0))

    with torch.no_grad():
        pred_scratch = torch.argmax(model_scratch(images_dev), dim=1).cpu()
        
        # Format images for official implementation
        B, C, H, W = images_dev.shape
        sh = ((H - 1) // 8) * 8 + 1
        sw = ((W - 1) // 8) * 8 + 1
        if sh != H or sw != W:
            off_images = F.interpolate(images_dev, size=(sh, sw), mode="bilinear", align_corners=True)
            off_out = model_official(off_images)
            off_out = F.interpolate(off_out, size=(H, W), mode="bilinear", align_corners=True)
        else:
            off_out = model_official(images_dev)
        pred_official = torch.argmax(off_out, dim=1).cpu()

    # un-normalize the images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    col_titles = ["Input Image", "Ground Truth", "Scratch PSPNet", "Official (hszhao)"]

    for i in range(num_samples):
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()

        gt = masks[i].numpy()
        s_pred = pred_scratch[i].numpy()
        o_pred = pred_official[i].numpy()

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="tab20", vmin=0, vmax=20)
        axes[i, 2].imshow(s_pred, cmap="tab20", vmin=0, vmax=20)
        axes[i, 3].imshow(o_pred, cmap="tab20", vmin=0, vmax=20)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12)

    fig.suptitle("Qualitative Comparison: Scratch vs Official PSPNet",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved qualitative results to '{save_path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Multi-scale inference (paper uses scales [0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
# ---------------------------------------------------------------------------

def multiscale_predict(model, image_batch, num_classes, device,
                       scales=(0.75, 1.0, 1.25), flip=True):
    """Evaluates images at different scales, averaged."""
    model.eval()
    B, C, H, W = image_batch.shape
    total_logits = torch.zeros(B, num_classes, H, W, device=device)

    with torch.no_grad():
        for s in scales:
            sh, sw = int(H * s), int(W * s)
            if hasattr(model, 'zoom_factor'):
                # Ensure dimensions satisfy (X-1) % 8 == 0 for hszhao's implementation
                sh = ((sh - 1) // 8) * 8 + 1
                sw = ((sw - 1) // 8) * 8 + 1
            scaled = F.interpolate(image_batch, size=(sh, sw),
                                   mode="bilinear", align_corners=True)
            out = model(scaled)
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=True)
            total_logits += out

            if flip:
                # horizontal flip
                flipped = torch.flip(scaled, dims=[3])
                out_f = model(flipped)
                out_f = torch.flip(out_f, dims=[3])
                out_f = F.interpolate(out_f, size=(H, W),
                                      mode="bilinear", align_corners=True)
                total_logits += out_f

    return torch.argmax(total_logits, dim=1)


def evaluate_with_multiscale(model, val_loader, device, num_classes):
    """
    Run multi-scale inference on the whole val set and return pixel acc + mIoU.
    """
    model.eval()
    all_preds, all_targets = [], []
    for images, masks in val_loader:
        images = images.to(device)
        preds = multiscale_predict(model, images, num_classes, device).cpu()
        all_preds.append(preds)
        all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = compute_pixel_accuracy(all_preds, all_targets, IGNORE_LABEL)
    miou = compute_miou(all_preds, all_targets, num_classes, IGNORE_LABEL)
    return acc, miou


def print_summary_table(hist_scratch, hist_official,
                        ms_scratch=None, ms_official=None):
    """Print a clean comparison table of final-epoch metrics."""
    print("\n" + "=" * 65)
    print("  FINAL METRICS COMPARISON (last epoch)")
    print("=" * 65)
    header = f"  {'Metric':<22} {'Scratch PSPNet':>18} {'Official (hszhao)':>18}"
    print(header)
    print("-" * 65)

    metrics = [
        ("Train Loss",  "train_loss"),
        ("Val Loss",    "val_loss"),
        ("Pixel Accuracy", "val_acc"),
        ("Mean IoU",    "val_miou"),
    ]
    for label, key in metrics:
        s = hist_scratch[key][-1]
        o = hist_official[key][-1]
        print(f"  {label:<22} {s:>18.4f} {o:>18.4f}")

    if ms_scratch and ms_official:
        print("-" * 65)
        print(f"  {'MS Pixel Accuracy':<22} {ms_scratch[0]:>18.4f} {ms_official[0]:>18.4f}")
        print(f"  {'MS Mean IoU':<22} {ms_scratch[1]:>18.4f} {ms_official[1]:>18.4f}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- hyperparameters (scaled for RTX 3050 6GB VRAM) ---
    epochs = 15
    batch_size = 2        # 6GB GPU can't handle larger batches at this resolution
    base_lr = 0.01        # paper uses 0.01 as the base learning rate
    num_train = 500       # subset of VOC
    num_val = 100
    crop_size = 257       # 473 OOMs on 6GB; 257 fits comfortably and works with hszhao's (x-1)%8 == 0 assert
    num_classes = VOC_NUM_CLASSES  # 21 for PASCAL VOC

    # --- prepare data ---
    print("\nPreparing datasets (will download VOC 2012 on first run)...")
    train_loader, val_loader = get_dataloaders(
        root="./data", batch_size=batch_size,
        num_train=num_train, num_val=num_val, crop_size=crop_size,
    )
    print(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"Val  : {len(val_loader.dataset)} images ({len(val_loader)} batches)")

    # ------------------------------------------------------------------
    # 1. Train scratch PSPNet (our implementation with auxiliary loss)
    # ------------------------------------------------------------------
    print("\n>>> Initializing Scratch PSPNet with auxiliary branch...")
    model_scratch = PSPNetScratch(num_classes=num_classes, use_aux=True)
    model_scratch, hist_scratch = train_model(
        model_scratch, train_loader, val_loader,
        epochs=epochs, base_lr=base_lr, device=device,
        name="Scratch PSPNet", is_scratch=True,
    )

    # ------------------------------------------------------------------
    # 2. Train official PSPNet by hszhao
    # ------------------------------------------------------------------
    print("\n>>> Initializing Official (hszhao) PSPNet...")
    model_official = HSZhaoPSPNet(
        layers=50,
        classes=num_classes,
        zoom_factor=8,
        pretrained=False  # To simplify model weights loading handling or set True if supported
    )
    
    # We define parameter groups simply if we wanted diff lr, but train_model does it broadly for !is_scratch
    
    model_official, hist_official = train_model(
        model_official, train_loader, val_loader,
        epochs=epochs, base_lr=base_lr, device=device,
        name="Official (hszhao) PSPNet", is_scratch=False,
    )

    # ------------------------------------------------------------------
    # 3. Multi-scale inference (paper evaluates at multiple scales)
    # ------------------------------------------------------------------
    print("\n>>> Running multi-scale inference (scales: 0.5–1.75 + flip)...")
    ms_scratch = evaluate_with_multiscale(
        model_scratch, val_loader, device, num_classes)
    print(f"  Scratch  MS — Pixel Acc: {ms_scratch[0]:.4f}, mIoU: {ms_scratch[1]:.4f}")

    ms_official = evaluate_with_multiscale(
        model_official, val_loader, device, num_classes)
    print(f"  Official MS — Pixel Acc: {ms_official[0]:.4f}, mIoU: {ms_official[1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Comparison outputs
    # ------------------------------------------------------------------
    print("\n>>> Generating comparison plots...")
    plot_comparison(hist_scratch, hist_official, epochs)

    print("\n>>> Generating qualitative predictions...")
    visualize_predictions(model_scratch, model_official, val_loader, device)

    print_summary_table(hist_scratch, hist_official, ms_scratch, ms_official)

    # save trained weights so results are reproducible
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_scratch.state_dict(), "checkpoints/pspnet_scratch.pth")
    torch.save(model_official.state_dict(), "checkpoints/pspnet_official.pth")
    print("\nSaved model checkpoints to 'checkpoints/'")


if __name__ == "__main__":
    main()
