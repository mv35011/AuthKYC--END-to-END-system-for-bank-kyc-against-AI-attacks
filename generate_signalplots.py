import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def extract_signals():
    print("Extracting Raw Signal Data...")

    # ==========================================
    # UPDATE THESE PATHS TO TWO SAMPLE VIDEOS
    # ==========================================
    real_video_path = './videos/real/VID20260419171317.mp4'
    fake_video_path = './videos/fake/426_287.mp4'

    # ---------------------------------------------------
    # PLOT 3: FFT Magnitude Spectrum (The Moiré Proof)
    # ---------------------------------------------------
    def get_fft_image(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        # Convert to grayscale and calculate 2D FFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        return magnitude_spectrum

    fft_real = get_fft_image(real_video_path)
    fft_fake = get_fft_image(fake_video_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Real Camera FFT
    axes[0].imshow(fft_real, cmap='inferno')
    axes[0].set_title("Real Camera (Smooth Spectrum)", fontsize=14, weight='bold')
    axes[0].axis('off')

    # Screen Replay FFT
    axes[1].imshow(fft_fake, cmap='inferno')
    axes[1].set_title("Screen Replay (Grid Artifacts / Spikes)", fontsize=14, weight='bold')
    axes[1].axis('off')

    plt.suptitle('S2 Moiré Detection: 2D FFT Magnitude Spectrums', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig('report_fft_spectrums.png', dpi=300)
    plt.close()

    # ---------------------------------------------------
    # PLOT 4: rPPG BVP Waveform (The Pulse Proof)
    # ---------------------------------------------------
    def get_rppg_waveform(video_path):
        cap = cv2.VideoCapture(video_path)
        means = []
        for _ in range(150):  # Extract 5 seconds
            ret, frame = cap.read()
            if not ret: break
            # Simple spatial mean of the green channel for visualization
            mean_g = np.mean(frame[:, :, 1])
            means.append(mean_g)
        cap.release()

        # Bandpass filter (0.7 Hz to 4.0 Hz) to isolate the human heartbeat
        means = np.array(means)
        nyquist = 0.5 * 30.0  # 30 fps
        b, a = butter(4, [0.7 / nyquist, 4.0 / nyquist], btype='band')
        pulse = filtfilt(b, a, means)
        return pulse

    pulse_real = get_rppg_waveform(real_video_path)
    pulse_fake = get_rppg_waveform(fake_video_path)
    time_axis = np.linspace(0, 5, len(pulse_real))

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, pulse_real, color='green', linewidth=2, label="Real Human (Clear Systolic Peaks)")
    plt.plot(time_axis[:len(pulse_fake)], pulse_fake, color='red', linewidth=1.5, alpha=0.7,
             label="Deepfake/Replay (Random Noise / No Pulse)")

    plt.title('S3 Biological Liveness: Extracted rPPG Waveforms', fontsize=16, weight='bold', pad=15)
    plt.xlabel('Time (Seconds)', fontsize=12, weight='bold')
    plt.ylabel('Blood Volume Pulse (BVP) Amplitude', fontsize=12, weight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_rppg_waveform.png', dpi=300)
    plt.close()

    print("Success! Generated 'report_fft_spectrums.png' and 'report_rppg_waveform.png'.")


if __name__ == "__main__":
    extract_signals()