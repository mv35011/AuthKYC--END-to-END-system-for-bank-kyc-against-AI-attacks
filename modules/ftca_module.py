import torch
import torch.nn as nn
import torch.fft
from torchvision.models.video import r3d_18


class FrequencyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches R3D-18's exact 8x temporal downsampling
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
        # Added to reach the final T/8 sequence length
        self.layer4 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FTCABlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        r3d = r3d_18(weights=None)
        self.rgb_stem = r3d.stem
        self.rgb_layer1 = r3d.layer1
        self.rgb_layer2 = r3d.layer2
        self.rgb_layer3 = r3d.layer3
        self.rgb_layer4 = r3d.layer4

        self.freq_encoder = FrequencyEncoder()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def compute_frequency_spectrum(self, rgb_tensor):
        gray = 0.2989 * rgb_tensor[:, 0:1, :, :, :] + \
               0.5870 * rgb_tensor[:, 1:2, :, :, :] + \
               0.1140 * rgb_tensor[:, 2:3, :, :, :]
        fft_complex = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        return torch.log(torch.abs(fft_shifted) + 1e-8)

    def forward(self, rgb_video):
        # RGB Branch
        x_rgb = self.rgb_stem(rgb_video)
        x_rgb = self.rgb_layer1(x_rgb)
        x_rgb = self.rgb_layer2(x_rgb)
        x_rgb = self.rgb_layer3(x_rgb)
        x_rgb = self.rgb_layer4(x_rgb)
        f_rgb = x_rgb.flatten(2).transpose(1, 2)

        # Frequency Branch
        freq_video = self.compute_frequency_spectrum(rgb_video)
        x_freq = self.freq_encoder(freq_video)
        f_freq = x_freq.flatten(2).transpose(1, 2)

        # Fusion & Classification
        attn_output, _ = self.cross_attention(query=f_rgb, key=f_freq, value=f_freq)
        fused_features = self.layer_norm(f_rgb + attn_output)
        pooled = self.pool(fused_features.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled)