import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
import os

from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===================================================")
    print(f"[System] Domain Adaptation Fine-Tuning on: {device}")
    print(f"===================================================")

    model = FTCABlock(embed_dim=512, num_heads=8)
    if os.path.exists('best_ftca_pad_model.pth'):
        # Security Fix
        model.load_state_dict(torch.load('best_ftca_pad_model.pth', map_location=device, weights_only=True),
                              strict=False)
        print("-> Successfully loaded previous weights for fine-tuning.")

    model = model.to(device)

    # EXACT FREEZING LOGIC: Freeze R3D backbone, unfreeze spatial-temporal heads
    for name, param in model.named_parameters():
        if name.startswith('rgb_'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Trainable params: {trainable:,}")

    train_dir = './processed_tensors/train'

    # ISOLATED TRAIN/VAL SPLIT (Fixes the data leakage bug)
    base_train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)
    base_val_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=False)

    dataset_size = len(base_train_dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    val_size = max(1, int(0.15 * dataset_size))

    train_dataset = Subset(base_train_dataset, indices[val_size:])
    val_dataset = Subset(base_val_dataset, indices[:val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 15
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
            correct += (predictions == labels.bool()).sum().item()

        train_loss = running_loss / max(1, len(train_dataset))
        train_acc = correct / max(1, total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(logits) > 0.5
                val_total += labels.size(0)
                val_correct += (predictions == labels.bool()).sum().item()

        val_loss /= max(1, len(val_dataset))
        val_acc = val_correct / max(1, val_total)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.1f}s | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'patent_ftca_v2.pth')
            print("   >>> Saved checkpoint (best val loss)")

    print(f"\n[System] Fine-Tuning Complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model()