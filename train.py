import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from dataset import get_dataloaders
from pspnet_scratch import PSPNetScratch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def plot_comparison(history_scratch, history_official, epochs):
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Train Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history_scratch['train_loss'], marker='o', label='Scratch PSPNet')
    plt.plot(epochs_range, history_official['train_loss'], marker='o', label='Official PSPNet')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Val Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history_scratch['val_loss'], marker='o', label='Scratch PSPNet')
    plt.plot(epochs_range, history_official['val_loss'], marker='o', label='Official PSPNet')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Val Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history_scratch['val_acc'], marker='o', label='Scratch PSPNet')
    plt.plot(epochs_range, history_official['val_acc'], marker='o', label='Official PSPNet')
    plt.title('Validation Pixel Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png')
    print("\nSaved comparison plots to 'comparison_plots.png'")

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu', name='model'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\\n--- Training {name} ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle auxiliary outputs for scratch implementation
            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss1 = criterion(main_out, masks)
                loss2 = criterion(aux_out, masks)
                loss = loss1 + 0.4 * loss2  # Paper sets aux weight to 0.4
            else:
                loss = criterion(outputs, masks)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Accuracy calculation
                preds = torch.argmax(outputs, dim=1)
                mask_valid = (masks != 255)
                correct += (preds[mask_valid] == masks[mask_valid]).sum().item()
                total += mask_valid.sum().item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_acc:.4f}")

    return model, history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    epochs = 5
    batch_size = 4
    num_train = 600  # Toy dataset size
    num_val = 200
    num_classes = 21

    # Dataloaders - Downloads VOC if needed
    print("\\nPreparing datasets and dataloaders...")
    train_loader, val_loader = get_dataloaders('./data', batch_size=batch_size, num_train=num_train, num_val=num_val)
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")

    # 1. Scratch Implementation
    print("\\n\\nInitializing Scratch PSPNet with Auxiliary Branch (0.4 loss weight)...")
    model_scratch = PSPNetScratch(num_classes=num_classes, use_aux=True)
    _, history_scratch = train_model(model_scratch, train_loader, val_loader, epochs=epochs, device=device, name="Scratch_PSPNet")

    # 2. Official/Standard Library Implementation (using SMP)
    print("\\n\\nInitializing Official/SMP PSPNet...")
    model_official = smp.PSPNet(
        encoder_name="resnet50",        # backbone architecture
        encoder_weights="imagenet",     # pre-trained weights
        in_channels=3,
        classes=num_classes,            # output classes
    )
    _, history_official = train_model(model_official, train_loader, val_loader, epochs=epochs, device=device, name="Official_PSPNet")

    # 3. Generate Comparison Plots
    print("\\nGenerating comparison plot...")
    plot_comparison(history_scratch, history_official, epochs)

if __name__ == '__main__':
    main()
