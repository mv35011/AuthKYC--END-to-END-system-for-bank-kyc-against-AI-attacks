import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import torchvision.transforms.v2 as v2
import numpy as np
import os


class TemporalDeepfakeDetector:
    def __init__(self, weights_path='best_temporal_deepfake_model.pth'):
        # 1. Hardware Auto-Detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')  # For the RunPod A6000
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # For the local M2 Apple Silicon
        else:
            self.device = torch.device('cpu')

        print(f"[Deepfake Module] Initialized on device: {self.device}")

        # 2. Build the Model Architecture
        self.model = r3d_18(weights=None)  # We don't need ImageNet video weights
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification head

        # 3. Load Weights or Use Graceful Fallback
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("[Deepfake Module] Successfully loaded trained weights.")
        else:
            print(f"[Deepfake Module] WARNING: {weights_path} not found.")
            print("[Deepfake Module] Running with untrained weights for API testing.")

        self.model = self.model.to(self.device)
        self.model.eval()  # Strictly set to evaluation mode

        # 4. Standardized Input Transformations
        self.transform = v2.Compose([
            v2.ToImage(),  # Converts numpy arrays to tensors
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer(self, frames):
        """
        Accepts a list of exactly 16 RGB numpy frames (from OpenCV).
        Returns a float score between 0.0 (Real) and 1.0 (Fake).
        """
        if len(frames) != 16:
            print(f"[Deepfake Module] Error: Expected 16 frames, got {len(frames)}. Returning 0.0")
            return 0.0

        try:
            # Transform each frame and stack them: [16, 3, 224, 224]
            transformed_frames = [self.transform(frame) for frame in frames]
            tensor_seq = torch.stack(transformed_frames)

            # PyTorch 3D CNNs require shape: [Batch, Channels, Depth, Height, Width]
            # Current shape: [16, 3, 224, 224] -> Permute to: [3, 16, 224, 224]
            tensor_seq = tensor_seq.permute(1, 0, 2, 3)

            # Add Batch dimension: [1, 3, 16, 224, 224] and move to device
            tensor_seq = tensor_seq.unsqueeze(0).to(self.device)

            # Run Inference without tracking gradients (saves memory)
            with torch.no_grad():
                logits = self.model(tensor_seq)

                # Apply sigmoid to convert raw logits to a 0-1 probability
                probability = torch.sigmoid(logits).item()

            return probability

        except Exception as e:
            print(f"[Deepfake Module] Inference failed: {str(e)}")
            return 0.0