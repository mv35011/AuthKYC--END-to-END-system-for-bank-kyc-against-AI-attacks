import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from dataset import DeepfakeVideoDataset
import time


def train_model():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Load Pre-trained 3D ResNet and Modify Classification Head
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)

    # Replace the final fully connected layer for binary classification (Fake vs Real)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # 3. Data Loaders (Tweak batch_size based on your A6000 VRAM, start with 16 or 32)
    train_dataset = DeepfakeVideoDataset(data_dir='./processed_tensors/train', is_training=True)
    val_dataset = DeepfakeVideoDataset(data_dir='./processed_tensors/val', is_training=False)

    # num_workers=8 is ideal for a beefy RunPod instance
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # 4. Optimization Strategy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # AMP Scaler for Half-Precision Training (Saves VRAM and RunPod $)
    scaler = torch.amp.GradScaler('cuda')

    # 5. The Training Loop
    epochs = 10
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Cast operations to mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 6. Quick Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predictions = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save Checkpoint to avoid losing progress if RunPod preempts
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_temporal_deepfake_model.pth')
            print(">>> Saved New Best Model Checkpoint")


if __name__ == "__main__":
    train_model()