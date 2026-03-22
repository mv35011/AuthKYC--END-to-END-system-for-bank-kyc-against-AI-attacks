import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, welch


class AdvancedrPPGDetector:
    def __init__(self, fps=30, window_seconds=5, snr_threshold=3.5):
        self.fps = fps
        self.buffer_size = fps * window_seconds
        self.snr_threshold = snr_threshold

        # Buffers for the raw spatial means of R, G, B
        self.r_buffer = []
        self.g_buffer = []
        self.b_buffer = []

        # MediaPipe Face Mesh optimized for CPU/Apple Silicon
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Expanded ROI: Forehead and upper cheeks for better blood perfusion tracking
        self.roi_indices = [
            10, 151, 337, 336, 109, 108,  # Forehead
            118, 119, 100, 126,  # Right Cheek
            347, 348, 329, 355  # Left Cheek
        ]

    def _design_bandpass_filter(self, lowcut=0.7, highcut=4.0, order=4):
        """Bandpass for physiological range: 0.7 Hz to 4.0 Hz (42 to 240 BPM)"""
        nyquist = 0.5 * self.fps
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
        return b, a

    def extract_spatial_means(self, frame):
        """Extracts the spatial average of R, G, B channels from the facial ROI."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, None, frame

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Map landmarks to pixel coordinates
        roi_points = []
        for idx in self.roi_indices:
            lm = face_landmarks.landmark[idx]
            roi_points.append((int(lm.x * w), int(lm.y * h)))

        # Create a convex hull mask for the ROI points
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(np.array(roi_points))
        cv2.fillConvexPoly(mask, hull, 255)

        # Calculate mean for each channel (R, G, B) strictly within the mask
        # Note: OpenCV image format is BGR, so indices are 2=R, 1=G, 0=B
        mean_b = cv2.mean(frame[:, :, 0], mask=mask)[0]
        mean_g = cv2.mean(frame[:, :, 1], mask=mask)[0]
        mean_r = cv2.mean(frame[:, :, 2], mask=mask)[0]

        # Draw the ROI for visual debugging
        cv2.polylines(frame, [hull], True, (0, 255, 255), 1)

        return mean_r, mean_g, mean_b, frame

    def apply_chrom(self):
        """Executes the CHROM algorithm on the buffered RGB signals."""
        r = np.array(self.r_buffer)
        g = np.array(self.g_buffer)
        b = np.array(self.b_buffer)

        # 1. Normalize signals by their rolling mean
        rn = r / (np.mean(r) + 1e-8)
        gn = g / (np.mean(g) + 1e-8)
        bn = b / (np.mean(b) + 1e-8)

        # 2. Project into orthogonal chrominance space
        x = 3 * rn - 2 * gn - bn
        y = 1.5 * rn + gn - 1.5 * bn

        # 3. Calculate ratio of standard deviations to find the scaling factor (alpha)
        # We add a tiny epsilon to prevent division by zero
        alpha = np.std(x) / (np.std(y) + 1e-8)

        # 4. Extract final pulse signal
        pulse_signal = x - alpha * y
        return pulse_signal

    def calculate_snr_and_bpm(self, pulse_signal):
        """Filters the pulse signal and calculates the frequency-domain SNR."""
        # Filter the raw pulse signal
        b, a = self._design_bandpass_filter()
        filtered_pulse = filtfilt(b, a, pulse_signal)

        # Use Welch's method to compute the Power Spectral Density (PSD)
        freqs, psd = welch(filtered_pulse, fs=self.fps, nperseg=len(filtered_pulse))

        # Restrict to physiological heart rate range (0.7 Hz - 4.0 Hz)
        valid_indices = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
        valid_freqs = freqs[valid_indices]
        valid_psd = psd[valid_indices]

        if len(valid_psd) == 0:
            return 0.0, 0.0

        # Find the dominant frequency (Peak of the pulse)
        max_idx = np.argmax(valid_psd)
        dominant_freq = valid_freqs[max_idx]
        bpm = dominant_freq * 60.0

        # Calculate SNR: Power of the peak vs Power of the rest of the spectrum
        peak_power = valid_psd[max_idx]
        total_power = np.sum(valid_psd)
        noise_power = total_power - peak_power

        snr = peak_power / (noise_power + 1e-8)
        # Convert to Decibels
        snr_db = 10 * np.log10(snr) if snr > 0 else 0

        return bpm, snr_db

    def process_frame(self, frame):
        """
        Main entry point for the orchestrator.
        Returns the current BPM, SNR, Liveness Decision, and annotated frame.
        """
        mean_r, mean_g, mean_b, processed_frame = self.extract_spatial_means(frame)

        bpm, snr = 0.0, 0.0
        passed = False

        if mean_r is not None:
            self.r_buffer.append(mean_r)
            self.g_buffer.append(mean_g)
            self.b_buffer.append(mean_b)

            # Maintain sliding window
            if len(self.r_buffer) > self.buffer_size:
                self.r_buffer.pop(0)
                self.g_buffer.pop(0)
                self.b_buffer.pop(0)

            # Process once buffer has enough data (e.g., at least 3 seconds)
            if len(self.r_buffer) >= (self.fps * 3):
                pulse_signal = self.apply_chrom()
                bpm, snr = self.calculate_snr_and_bpm(pulse_signal)

                # The Liveness Gate Logic
                if snr >= self.snr_threshold and 45 <= bpm <= 150:
                    passed = True

        return {
            "bpm": float(bpm),
            "snr_db": float(snr),
            "passed": passed,
            "buffer_fill_ratio": len(self.r_buffer) / self.buffer_size
        }, processed_frame


# Quick local test block for Apple Silicon
if __name__ == "__main__":
    detector = AdvancedrPPGDetector(fps=30)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        result, display_frame = detector.process_frame(frame)

        if result['buffer_fill_ratio'] < 1.0:
            status = f"BUFFERING: {int(result['buffer_fill_ratio'] * 100)}%"
            color = (0, 165, 255)
        elif result['passed']:
            status = f"LIVENESS: VERIFIED | SNR: {result['snr_db']:.1f}dB"
            color = (0, 255, 0)
        else:
            status = f"LIVENESS: FAILED | SNR: {result['snr_db']:.1f}dB"
            color = (0, 0, 255)

        cv2.putText(display_frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Pulse: {result['bpm']:.1f} BPM", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow('Advanced rPPG (CHROM)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()