import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===================================================")
    print(f"[System] Domain Adaptation Fine-Tuning on: {device}")
    print(f"===================================================")

    # 1. Initialize Model and Load Golden Weights
    model = FTCABlock(embed_dim=512, num_heads=8)
    if os.path.exists('best_ftca_pad_model.pth'):
        model.load_state_dict(torch.load('best_ftca_pad_model.pth', map_location=device))
        print("-> Successfully loaded previous weights for fine-tuning.")
    else:
        print("-> WARNING: Previous weights not found. Training from scratch.")

    model = model.to(device)

    # 2. FREEZE THE SPATIAL BACKBONE
    for name, param in model.named_parameters():
        if 'temporal' in name or 'classifier' in name or 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False  # Freeze 2D CNN

    train_dir = './processed_tensors/train'
    train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)

    # 3. STANDARD DATALOADER (Since data is physically balanced)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # 4. REDUCED LEARNING RATE FOR FINE TUNING
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 15
    best_train_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            predictions = torch.sigmoid(logits) > 0.5
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        epoch_acc = correct / max(1, total)

        scheduler.step(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # Save model if training loss improves (since we don't have a separate Val set for this fast sprint)
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            torch.save(model.state_dict(), 'patent_ftca_v2.pth')
            print("   >>> Saved Fine-Tuned Domain-Adapted Checkpoint")

    print(f"\n[System] Domain Adaptation Fine-Tuning Complete.")


if __name__ == "__main__":
    train_model()