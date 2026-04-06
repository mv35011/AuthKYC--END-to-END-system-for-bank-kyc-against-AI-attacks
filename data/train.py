import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# Import our custom novel architecture
from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset


def train_model():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training FTCA Architecture on: {device}")

    # 2. Initialize our Custom FTCA Model
    # The FTCABlock already has the correct binary classification head built-in
    model = FTCABlock(embed_dim=512, num_heads=8)
    model = model.to(device)

    # 3. Data Loaders (A6000 can easily handle batch_size 32 with AMP)
    train_dir = './processed_tensors/train'
    val_dir = './processed_tensors/val'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dataset = DeepfakeVideoDataset(data_dir=train_dir, is_training=True)
    val_dataset = DeepfakeVideoDataset(data_dir=val_dir, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # 4. Optimization Strategy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # AMP Scaler for Half-Precision Training
    scaler = torch.amp.GradScaler('cuda')

    epochs = 15
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(logits) > 0.5
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ftca_pad_model.pth')
            print(">>> Saved New Best FTCA Model Checkpoint")


if __name__ == "__main__":
    train_model()