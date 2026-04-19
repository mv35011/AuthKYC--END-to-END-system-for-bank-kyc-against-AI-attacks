import cv2
import numpy as np


class ReplayAttackDetector:
    def __init__(self, threshold=75000):
        # Threshold for high-frequency noise spikes
        self.threshold = threshold

    def analyze_frame(self, frame):
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize to a fixed resolution so scores are comparable
        # across different camera/video resolutions
        target_width = 640
        h, w = gray.shape
        if w != target_width:
            scale = target_width / w
            gray = cv2.resize(gray, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = gray.shape

        # 2. Compute the 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 3. Calculate Magnitude Spectrum (Log scale for visualization)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # 4. Resolution-proportional mask to block low frequencies (center of spectrum)
        # We only care about unnatural high-frequency spikes (screen grids/Moiré)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        # Mask the center 15% of the spectrum (natural image content)
        mask_h = int(rows * 0.15)
        mask_w = int(cols * 0.15)

        # Work on a copy so we don't mutate the original for the magnitude display
        fshift_masked = fshift.copy()
        fshift_masked[crow - mask_h: crow + mask_h, ccol - mask_w: ccol + mask_w] = 0

        # 5. High-frequency energy — use sum, not mean
        # Mean dilutes the score over all frequency bins (most of which are near-zero).
        # Sum captures the total high-frequency energy which is what we want to threshold.
        high_freq_magnitude = np.abs(fshift_masked)
        anomaly_score = np.sum(high_freq_magnitude) / 1e6  # scale to manageable range

        # 6. Peak detection — screen pixel grids create sharp periodic peaks
        # that stand out above the natural noise floor.
        # Find how spiky the high-freq spectrum is vs the noise floor.
        flat_mag = high_freq_magnitude.flatten()
        flat_mag = flat_mag[flat_mag > 0]  # exclude masked zeros
        if len(flat_mag) > 100:
            # Ratio of the top-1% energy to the median energy
            top_1_pct = np.percentile(flat_mag, 99)
            median_val = np.median(flat_mag)
            peak_ratio = top_1_pct / (median_val + 1e-8)
        else:
            peak_ratio = 1.0

        # A screen grid creates strong periodic peaks (peak_ratio >> 5)
        # Natural scenes have more uniform high-frequency distribution (peak_ratio ~ 2-4)
        # Boost the anomaly score if there are sharp spectral peaks
        if peak_ratio > 5.0:
            anomaly_score *= (peak_ratio / 3.0)

        # 7. Normalize magnitude spectrum for visualization (0-255)
        mag_display = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return anomaly_score, mag_display

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            score, mag_display = self.analyze_frame(frame)

            # Determine if it's a physical camera or a screen replay
            if score > self.threshold:
                status = "WARNING: SCREEN / REPLAY DETECTED"
                color = (0, 0, 255)  # Red
            else:
                status = "LIVE: PHYSICAL CAMERA"
                color = (0, 255, 0)  # Green

            # Display the score and status on the original frame
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"High-Freq Score: {score:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Show original feed and the frequency domain
            cv2.imshow('KYC Feed', frame)
            cv2.imshow('Frequency Domain (FFT)', mag_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ReplayAttackDetector(threshold=75000)
    detector.run()