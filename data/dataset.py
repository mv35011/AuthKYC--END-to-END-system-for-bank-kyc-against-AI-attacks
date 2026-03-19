import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        """
        Expects a directory containing .pt files of shape [Num_Sequences, 16, 3, 224, 224]
        Labeling convention: Files starting with 'real_' are 0, 'fake_' are 1.
        """
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.is_training = is_training

        # GPU-friendly augmentations that apply consistently across the time dimension
        self.train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Determine label from filename (0 for real, 1 for fake)
        filename = os.path.basename(file_path).lower()
        label = 1.0 if 'fake' in filename else 0.0

        # Load the pre-extracted tensor: [Seq, 16, 3, 224, 224]
        tensor_data = torch.load(file_path)

        # Randomly select one 16-frame sequence from the video file to add variance
        num_sequences = tensor_data.shape[0]
        seq_idx = torch.randint(0, num_sequences, (1,)).item() if self.is_training else 0
        sequence = tensor_data[seq_idx]

        if self.is_training:
            # Apply augmentations across the [16, 3, 224, 224] block
            sequence = self.train_transforms(sequence)

        # PyTorch 3D CNNs expect shape: [Channels, Depth(Time), Height, Width]
        # Current shape is [16, 3, 224, 224]. We must permute to [3, 16, 224, 224]
        sequence = sequence.permute(1, 0, 2, 3)

        return sequence, torch.tensor([label], dtype=torch.float32)