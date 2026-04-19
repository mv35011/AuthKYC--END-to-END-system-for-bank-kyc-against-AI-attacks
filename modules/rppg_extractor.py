import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt, welch


class AdvancedrPPGDetector:
    def __init__(self, fps=30, window_seconds=10, snr_threshold=1.5):
        self.fps = fps
        self.buffer_size = fps * window_seconds
        self.snr_threshold = snr_threshold

        self.r_buffer = []
        self.g_buffer = []
        self.b_buffer = []

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def reset(self):
        """Clears all buffers so a new video starts with a clean slate."""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()

    def _design_bandpass_filter(self, lowcut=0.7, highcut=4.0, order=4):
        nyquist = 0.5 * self.fps
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
        return b, a

    def extract_spatial_means(self, frame):
        # Work in RGB space for MediaPipe, but extract means from RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, None, frame

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        regions = {
            "forehead": [10, 109, 67, 103, 54, 151, 337, 336, 338, 297, 332, 284],
            "right_cheek": [117, 118, 119, 100, 126, 205, 206, 207, 142],
            "left_cheek": [346, 347, 348, 329, 355, 425, 426, 427, 371]
        }

        for name, indices in regions.items():
            pts = np.array([
                (int(face_landmarks.landmark[idx].x * w),
                 int(face_landmarks.landmark[idx].y * h))
                for idx in indices
            ])
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            cv2.polylines(frame, [hull], True, (0, 255, 255), 1)

        # FIX 1: Extract from RGB frame, not BGR frame
        # rgb_frame channels: 0=R, 1=G, 2=B
        mean_r = cv2.mean(rgb_frame[:, :, 0], mask=mask)[0]
        mean_g = cv2.mean(rgb_frame[:, :, 1], mask=mask)[0]
        mean_b = cv2.mean(rgb_frame[:, :, 2], mask=mask)[0]

        return mean_r, mean_g, mean_b, frame

    def apply_chrom(self):
        from scipy.signal import detrend

        r = np.array(self.r_buffer)
        g = np.array(self.g_buffer)
        b = np.array(self.b_buffer)

        # Step 1: Normalize by temporal mean FIRST (standard CHROM preprocessing)
        # This gives us the relative change: (x - mean) / mean ≈ AC/DC component
        rn = r / (np.mean(r) + 1e-8)
        gn = g / (np.mean(g) + 1e-8)
        bn = b / (np.mean(b) + 1e-8)

        # Step 2: Detrend AFTER normalizing to remove slow luminance drift
        # (e.g. from M2 webcam auto-exposure wobble)
        rn = detrend(rn)
        gn = detrend(gn)
        bn = detrend(bn)

        # Step 3: Full CHROM projection — both terms must include blue
        x = 3 * rn - 2 * gn  # ← WRONG in your current code
        x = 3 * rn - 2 * gn - bn  # ← correct
        y = 1.5 * rn + gn - 1.5 * bn

        # Step 4: Alpha scaling and pulse extraction
        alpha = np.std(x) / (np.std(y) + 1e-8)
        pulse_signal = x - alpha * y

        return pulse_signal

    def calculate_snr_and_bpm(self, pulse_signal):
        b, a = self._design_bandpass_filter()
        filtered_pulse = filtfilt(b, a, pulse_signal)

        # FIX 3: Use nperseg as a fraction of signal length for better freq resolution
        nperseg = min(len(filtered_pulse), self.fps * 4)
        freqs, psd = welch(filtered_pulse, fs=self.fps, nperseg=nperseg,
                           noverlap=nperseg // 2)

        valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
        valid_freqs = freqs[valid_idx]
        valid_psd = psd[valid_idx]

        if len(valid_psd) == 0:
            return 0.0, 0.0

        max_idx = np.argmax(valid_psd)
        dominant_freq = valid_freqs[max_idx]
        bpm = dominant_freq * 60.0

        # SNR: peak bin power vs average noise floor (excluding ±2 bins around peak)
        peak_power = valid_psd[max_idx]
        mask_bins = np.ones(len(valid_psd), dtype=bool)
        lo = max(0, max_idx - 2)
        hi = min(len(valid_psd), max_idx + 3)
        mask_bins[lo:hi] = False
        noise_power = np.mean(valid_psd[mask_bins]) if mask_bins.any() else 1e-8

        snr = peak_power / (noise_power + 1e-8)
        snr_db = 10 * np.log10(snr) if snr > 0 else 0

        return bpm, snr_db

    def process_frame(self, frame):
        mean_r, mean_g, mean_b, processed_frame = self.extract_spatial_means(frame)

        bpm, snr = 0.0, 0.0
        passed = False

        if mean_r is not None:
            self.r_buffer.append(mean_r)
            self.g_buffer.append(mean_g)
            self.b_buffer.append(mean_b)

            if len(self.r_buffer) > self.buffer_size:
                self.r_buffer.pop(0)
                self.g_buffer.pop(0)
                self.b_buffer.pop(0)

            if len(self.r_buffer) >= (self.fps * 3):  # need at least 5s
                pulse_signal = self.apply_chrom()
                bpm, snr = self.calculate_snr_and_bpm(pulse_signal)

                if snr >= self.snr_threshold and 45 <= bpm <= 150:
                    passed = True

        return {
            "bpm": float(bpm),
            "snr_db": float(snr),
            "passed": passed,
            "buffer_fill_ratio": len(self.r_buffer) / self.buffer_size
        }, processed_frame


if __name__ == "__main__":
    detector = AdvancedrPPGDetector(fps=30)
    cap = cv2.VideoCapture(0)

    # FIX 4: Lock exposure on MacBook to prevent auto-exposure drift
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual on macOS
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # try -5 to -7

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

        cv2.putText(display_frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Pulse: {result['bpm']:.1f} BPM", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Advanced rPPG (CHROM)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()