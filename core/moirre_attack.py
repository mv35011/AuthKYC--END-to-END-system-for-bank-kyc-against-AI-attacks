import cv2
import numpy as np


class ReplayAttackDetector:
    def __init__(self, threshold=150):
        # Threshold for high-frequency noise spikes
        self.threshold = threshold

    def analyze_frame(self, frame):
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Compute the 2D Fast Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 3. Calculate Magnitude Spectrum (Log scale for visibility)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # 4. Mask the low frequencies (the center of the spectrum)
        # We only care about unnatural high-frequency spikes (screen grids/Moiré)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask_size = 50

        # Create a mask to block out the center (natural image frequencies)
        fshift[crow - mask_size: crow + mask_size, ccol - mask_size: ccol + mask_size] = 0

        # 5. Calculate the remaining high-frequency energy
        high_freq_magnitude = np.abs(fshift)
        anomaly_score = np.mean(high_freq_magnitude)

        # 6. Normalize magnitude spectrum for visualization (0-255)
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
    detector = ReplayAttackDetector(threshold=120000)  # Tune this threshold based on your webcam
    detector.run()