import cv2
import numpy as np


class PRNUDetector:
    def __init__(self, energy_threshold=0.5):
        # Threshold for the minimum amount of sensor noise expected from a real camera
        self.energy_threshold = energy_threshold
        self.noise_residuals = []
        self.frame_shape = None

    def extract_noise_residual(self, frame):
        """
        Extracts the high-frequency noise from a frame by subtracting a denoised
        version of the frame from the original.
        """
        # Convert to grayscale to reduce computational load
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downsample to a fixed resolution so the score is resolution-independent.
        # This normalizes phone 1080p, webcam 720p, and replay 480p to the same scale.
        target_width = 480
        h, w = gray.shape
        if w > target_width:
            scale = target_width / w
            gray = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Store shape for diagnostics
        if self.frame_shape is None:
            self.frame_shape = gray.shape

        # FIX: Use kernel=3 instead of 5 to preserve more of the actual sub-pixel PRNU.
        # Kernel=5 was too aggressive and destroyed the real sensor fingerprint,
        # leaving only compression artifacts.
        denoised = cv2.medianBlur(gray, 3)

        # The residual is the raw noise isolated from the image
        # We use int16 to prevent underflow when subtracting
        residual = np.int16(gray) - np.int16(denoised)
        return residual

    def process_frame(self, frame):
        """Buffers noise residuals across the video stream."""
        residual = self.extract_noise_residual(frame)
        self.noise_residuals.append(residual)

    def analyze_fingerprint(self):
        """
        Aggregates the residuals over time to find the camera's persistent fingerprint.

        Physical cameras produce unique, persistent sensor defects (PRNU) with
        broadband high-frequency noise. Virtual cameras (OBS) and screen replays
        produce near-zero persistent noise OR periodic compression artifacts.

        Returns:
            (noise_energy, is_physical_camera)
        """
        if len(self.noise_residuals) < 10:
            return 0.0, False  # Not enough frames

        # Stack residuals across time: [Time, Height, Width]
        stacked_residuals = np.stack(self.noise_residuals)

        # Calculate the temporal mean to find the persistent PRNU fingerprint
        # Random noise cancels out; persistent sensor defects remain
        prnu_fingerprint = np.mean(stacked_residuals, axis=0)

        # --- Primary metric: normalized energy ---
        # Variance of the fingerprint, normalized by spatial area
        # so it's comparable across different input resolutions
        noise_energy = float(np.var(prnu_fingerprint))

        # --- Secondary metric: spectral flatness ---
        # Real PRNU has broadband noise (flat spectrum).
        # Compression artifacts have sharp periodic peaks at 8x8 block boundaries.
        # Spectral flatness = geometric_mean(PSD) / arithmetic_mean(PSD)
        # Values close to 1.0 = broadband (real PRNU), close to 0.0 = periodic (artifacts)
        fft_2d = np.fft.fft2(prnu_fingerprint)
        magnitude = np.abs(fft_2d)

        # Exclude the DC component (index [0,0])
        magnitude_flat = magnitude.flatten()
        magnitude_flat = magnitude_flat[1:]  # drop DC
        magnitude_flat = magnitude_flat[magnitude_flat > 0]  # avoid log(0)

        if len(magnitude_flat) > 0:
            log_mean = np.mean(np.log(magnitude_flat + 1e-10))
            geometric_mean = np.exp(log_mean)
            arithmetic_mean = np.mean(magnitude_flat)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        else:
            spectral_flatness = 0.0

        # --- Combined decision ---
        # A physical camera needs BOTH:
        #   1. Sufficient noise energy (sensor defects exist)
        #   2. Broadband spectrum (not just compression block artifacts)
        #
        # Spectral flatness thresholds (empirical):
        #   Real camera PRNU:    ~0.4 - 0.8 (broadband)
        #   Compression artifacts: ~0.1 - 0.3 (periodic peaks)
        #   Virtual camera (OBS):  very low energy regardless

        is_physical_camera = bool(
            noise_energy > self.energy_threshold and spectral_flatness > 0.3
        )

        # Clean up buffer for the next stream
        self.noise_residuals.clear()
        self.frame_shape = None

        return noise_energy, is_physical_camera