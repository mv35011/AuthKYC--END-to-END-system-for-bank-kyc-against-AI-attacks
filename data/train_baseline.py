import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm  # Required for Xception model
import time
import os
from dataset import DeepfakeVideoDataset


def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Xception Baseline on: {device}")

    # 1. Initialize Xception Model as per Exp 1 requirements
    # pretrained=True uses ImageNet weights, num_classes=1 sets up binary classification
    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model = model.to(device)

    # 2. Data Loaders (Reusing the exact same dataset folder as FTCA)
    train_dir = './processed_tensors/train'
    val_dir = './processed_tensors/val'

    train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)
    val_dataset = DeepfakeVideoDataset(data_dir=val_dir, is_training=False)

    # V100 can easily handle batch_size 32 for a 2D model
    # Reverted to Linux multiprocessing and A6000 batch sizes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # 3. Optimization Strategy (As defined in Experiment 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 20  # As specified in Exp 1 setup
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # THE HACK: Convert 3D sequence [B, 3, 16, 224, 224] to 2D frame [B, 3, 224, 224]
            # We just take the first frame of the sequence for the static 2D CNN
            inputs_2d = inputs[:, :, 0, :, :]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs_2d)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                inputs_2d = inputs[:, :, 0, :, :]  # Slice to 2D

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs_2d)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_xception_baseline.pth')
            print(">>> Saved New Best Baseline Checkpoint")


if __name__ == "__main__":
    train_baseline()