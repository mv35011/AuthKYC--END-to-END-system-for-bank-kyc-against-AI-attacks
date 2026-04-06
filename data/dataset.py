import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        """
        Expects a base directory containing 'real' and 'fake' subfolders with .pt files.
        """
        self.is_training = is_training
        self.file_paths = []
        self.labels = []

        # Load from 'fake' directory (Label 1.0)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(fake_dir, f))
                    self.labels.append(1.0)

        # Load from 'real' directory (Label 0.0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(real_dir, f))
                    self.labels.append(0.0)

        # GPU-friendly augmentations
        self.train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        tensor_data = torch.load(file_path)

        num_sequences = tensor_data.shape[0]
        seq_idx = torch.randint(0, num_sequences, (1,)).item() if self.is_training else 0
        sequence = tensor_data[seq_idx]

        if self.is_training:
            sequence = self.train_transforms(sequence)

        # Output shape matches FTCA requirement: [3, 16, 224, 224]
        sequence = sequence.permute(1, 0, 2, 3)

        return sequence, torch.tensor([label], dtype=torch.float32)