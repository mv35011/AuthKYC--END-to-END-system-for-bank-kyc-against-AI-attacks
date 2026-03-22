import torch
import torch.nn as nn
import torch.fft
from torchvision.models.video import r3d_18


class FrequencyEncoder(nn.Module):
    """
    A lightweight 3D CNN designed specifically to process the
    Fourier Magnitude Spectrums over time.
    """

    def __init__(self):
        super().__init__()
        # Input is 1 channel (Grayscale Magnitude Spectrum), [B, 1, T, H, W]
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.layer1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FTCABlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()

        # --- 1. The RGB Branch (Spatial-Temporal) ---
        # We use a standard R3D-18 but strip off the final pooling and classification layers
        # so we can access the raw feature maps before they are flattened.
        r3d = r3d_18(weights=None)
        self.rgb_stem = r3d.stem
        self.rgb_layer1 = r3d.layer1
        self.rgb_layer2 = r3d.layer2
        self.rgb_layer3 = r3d.layer3
        self.rgb_layer4 = r3d.layer4

        # --- 2. The Frequency Branch ---
        self.freq_encoder = FrequencyEncoder()

        # --- 3. The Cross-Attention Fusion Layer ---
        # embed_dim matches the 512 output channels of the R3D and Frequency branches
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # --- 4. The Classification Head ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary classification (Real vs Fake)
        )

    def compute_frequency_spectrum(self, rgb_tensor):
        """
        Differentiable 2D Fast Fourier Transform on the GPU.
        Converts RGB to Grayscale, applies FFT2, and extracts the log magnitude.
        rgb_tensor shape: [B, 3, T, H, W]
        """
        # Convert to roughly grayscale using standard luminescence weights
        gray = 0.2989 * rgb_tensor[:, 0:1, :, :, :] + \
               0.5870 * rgb_tensor[:, 1:2, :, :, :] + \
               0.1140 * rgb_tensor[:, 2:3, :, :, :]

        # Apply 2D FFT across the spatial dimensions (dim=-2 and dim=-1)
        fft_complex = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")

        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        # Extract magnitude and apply log scale: log(|F| + 1)
        magnitude = torch.abs(fft_shifted)
        log_magnitude = torch.log(magnitude + 1e-8)

        return log_magnitude

    def forward(self, rgb_video):
        """
        Expects rgb_video of shape: [Batch, Channels(3), Time(16), Height(224), Width(224)]
        """
        B, C, T, H, W = rgb_video.shape

        # === BRANCH 1: RGB Spatial-Temporal Processing ===
        x_rgb = self.rgb_stem(rgb_video)
        x_rgb = self.rgb_layer1(x_rgb)
        x_rgb = self.rgb_layer2(x_rgb)
        x_rgb = self.rgb_layer3(x_rgb)
        x_rgb = self.rgb_layer4(x_rgb)
        # x_rgb shape is now roughly: [B, 512, T/8, H/32, W/32]

        # Flatten spatial and temporal dimensions to create a sequence for Attention
        # Shape becomes: [B, 512, Seq_Len] -> transpose to [B, Seq_Len, 512]
        f_rgb = x_rgb.flatten(2).transpose(1, 2)

        # === BRANCH 2: Frequency Processing ===
        freq_video = self.compute_frequency_spectrum(rgb_video)
        x_freq = self.freq_encoder(freq_video)
        # Flatten to match RGB sequence
        f_freq = x_freq.flatten(2).transpose(1, 2)

        # === CROSS-ATTENTION FUSION ===
        # Query comes from RGB (Where should I look in the image/time?)
        # Key and Value come from Frequency (What frequency artifacts exist there?)
        attn_output, _ = self.cross_attention(query=f_rgb, key=f_freq, value=f_freq)

        # Add residual connection and normalize
        fused_features = self.layer_norm(f_rgb + attn_output)

        # === CLASSIFICATION ===
        # Pool across the sequence length to get a single vector per batch
        # Transpose back to [B, 512, Seq_Len] for pooling
        pooled = self.pool(fused_features.transpose(1, 2)).squeeze(-1)

        # Pass through final linear layers
        logits = self.classifier(pooled)

        return logits