import torch
from dataset import DeepfakeVideoDataset
from torch.utils.data import DataLoader
from collections import Counter

val_dataset = DeepfakeVideoDataset('./processed_tensors/val', is_training=False)
print(f"Val dataset size: {len(val_dataset)}")

# Check label distribution
labels = [val_dataset.labels[i] for i in range(len(val_dataset))]
print(f"Label distribution: {Counter(labels)}")

# Check what Xception actually receives
loader = DataLoader(val_dataset, batch_size=4)
inputs, labels = next(iter(loader))
print(f"Full input shape: {inputs.shape}")       # Should be (4, 3, 16, 224, 224)
inputs_2d = inputs[:, :, 0, :, :]
print(f"Xception input shape: {inputs_2d.shape}") # Should be (4, 3, 224, 224)
print(f"Labels: {labels}")
print(f"Input min/max: {inputs_2d.min():.3f}, {inputs_2d.max():.3f}")