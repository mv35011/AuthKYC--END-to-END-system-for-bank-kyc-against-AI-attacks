import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from modules.ftca_module import FTCABlock
from dataset import DeepfakeVideoDataset


def evaluate_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Loading Best FTCA Checkpoint on {device}...")

    # 1. Initialize the architecture
    model = FTCABlock(embed_dim=512, num_heads=8)
    model = model.to(device)

    # 2. Load the saved weights
    checkpoint_path = 'best_ftca_pad_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    # 3. Load ONLY the validation data
    val_dataset = DeepfakeVideoDataset(data_dir='./processed_tensors/val', is_training=False)
    # Batch size 16 to safely handle the 8-sequence tensors
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0
    correct = 0
    total = 0

    print("[System] Running evaluation pass. This will take about 60 seconds...")
    start_time = time.time()

    # 4. Run the forward pass without tracking gradients
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

    final_loss = val_loss / max(1, len(val_loader.dataset))
    final_acc = correct / max(1, total)

    print(f"\n===================================================")
    print(f"FTCA BEST CHECKPOINT RECOVERY (PHASE 2)")
    print(f"Time Taken: {time.time() - start_time:.2f}s")
    print(f"Validation Loss: {final_loss:.4f}")
    print(f"Validation Accuracy: {final_acc * 100:.2f}%")
    print(f"===================================================\n")


if __name__ == "__main__":
    evaluate_best_model()