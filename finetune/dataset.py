import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.is_training = is_training
        self.file_paths = []
        self.labels = []

        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(fake_dir, f))
                    self.labels.append(1.0)

        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(real_dir, f))
                    self.labels.append(0.0)

        # THE DIRTY WEBCAM AUGMENTATIONS
        self.train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.3),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        tensor_data = torch.load(file_path, weights_only=True)
        num_sequences = tensor_data.shape[0]
        seq_idx = torch.randint(0, num_sequences, (1,)).item() if self.is_training else 0
        sequence = tensor_data[seq_idx]

        if self.is_training:
            sequence = self.train_transforms(sequence)
            # Add raw digital sensor noise manually
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(sequence) * 0.05
                sequence = sequence + noise

        sequence = sequence.permute(1, 0, 2, 3) # -> (C, T, H, W)
        return sequence, torch.tensor([label], dtype=torch.float32)