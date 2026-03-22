import cv2
import numpy as np


class PRNUDetector:
    def __init__(self, energy_threshold=0.5):
        # Threshold for the minimum amount of sensor noise expected from a real camera
        self.energy_threshold = energy_threshold
        self.noise_residuals = []

    def extract_noise_residual(self, frame):
        """
        Extracts the high-frequency noise from a frame by subtracting a denoised
        version of the frame from the original.
        """
        # Convert to grayscale to reduce computational load
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a median blur to approximate the "clean" image content
        # Median blur is excellent for preserving edges while removing noise
        denoised = cv2.medianBlur(gray, 5)

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
        Software cameras (OBS) will have near-zero persistent noise energy.
        """
        if len(self.noise_residuals) < 10:
            return 0.0, False  # Not enough frames

        # Stack residuals across time: [Time, Height, Width]
        stacked_residuals = np.stack(self.noise_residuals)

        # Calculate the temporal mean to find the persistent PRNU fingerprint
        # Random noise cancels out; persistent sensor defects remain
        prnu_fingerprint = np.mean(stacked_residuals, axis=0)

        # Calculate the energy (variance) of the fingerprint
        noise_energy = np.var(prnu_fingerprint)

        # Clean up buffer for the next stream
        self.noise_residuals.clear()

        is_physical_camera = bool(noise_energy > self.energy_threshold)
        return noise_energy, is_physical_camera