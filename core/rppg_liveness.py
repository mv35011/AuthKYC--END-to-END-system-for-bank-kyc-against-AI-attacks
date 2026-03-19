import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq


class rPPGDetector:
    def __init__(self, fps=30, buffer_size=150):
        # 150 frames at 30 FPS = 5 seconds of data
        self.fps = fps
        self.buffer_size = buffer_size
        self.signal_buffer = []

        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Forehead landmarks in MediaPipe
        self.forehead_indices = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                                 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297,
                                 338]
        # simplified forehead block for stable extraction
        self.roi_indices = [10, 109, 67, 103, 332, 338, 297, 332]

    def design_bandpass_filter(self, lowcut=0.75, highcut=2.5, order=5):
        """Designs a Butterworth bandpass filter for human heart rate (45-150 BPM)."""
        nyquist = 0.5 * self.fps
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def extract_signal(self, frame):
        """Extracts the mean green channel value from the forehead ROI."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Get coordinates for the forehead
                roi_points = []
                for idx in [10, 151, 337, 336]:  # Central forehead block
                    lm = face_landmarks.landmark[idx]
                    roi_points.append((int(lm.x * w), int(lm.y * h)))

                # Create a mask for the ROI
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(roi_points)], 255)

                # Extract the green channel
                green_channel = frame[:, :, 1]

                # Calculate the mean of the green channel within the ROI
                mean_g = cv2.mean(green_channel, mask=mask)[0]

                # Draw the ROI for visual feedback
                cv2.polylines(frame, [np.array(roi_points)], True, (0, 255, 0), 2)

                return mean_g, frame
        return None, frame

    def process_buffer(self):
        """Applies filtering and FFT to determine the heart rate."""
        if len(self.signal_buffer) < self.buffer_size:
            return None

        # Detrend the signal (remove the mean)
        signal = np.array(self.signal_buffer)
        signal = signal - np.mean(signal)

        # Apply Bandpass Filter
        b, a = self.design_bandpass_filter()
        filtered_signal = filtfilt(b, a, signal)

        # Apply FFT to find the dominant frequency
        N = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(N, 1 / self.fps)

        # Only look at the positive frequencies
        idx = np.argmax(np.abs(yf[1:N // 2])) + 1
        dominant_freq = xf[idx]

        bpm = dominant_freq * 60
        return bpm

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mean_g, processed_frame = self.extract_signal(frame)

            if mean_g is not None:
                self.signal_buffer.append(mean_g)

                # Keep buffer at fixed size
                if len(self.signal_buffer) > self.buffer_size:
                    self.signal_buffer.pop(0)

                    bpm = self.process_buffer()

                    if bpm is not None and 45 <= bpm <= 150:
                        cv2.putText(processed_frame, f"Liveness: VERIFIED (Pulse: {bpm:.1f} BPM)",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(processed_frame, "Liveness: ANALYZING...",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.putText(processed_frame, "No Face Detected",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('rPPG Biological Liveness', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = rPPGDetector(fps=30, buffer_size=150)
    detector.run()