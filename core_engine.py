import cv2
import numpy as np

# Import your isolated modules
from core.moirre_attack import ReplayAttackDetector
from core.rppg_liveness import rPPGDetector
from core.temporal_deepfake import TemporalDeepfakeDetector


class KYCOrchestrator:
    def __init__(self):
        # Initialize the independent modules
        self.moire_module = ReplayAttackDetector(threshold=120000)
        self.rppg_module = rPPGDetector(fps=30)
        self.deepfake_module = TemporalDeepfakeDetector()

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        moire_scores = []
        rppg_buffer = []
        frames_for_cnn = []

        frame_count = 0
        while cap.isOpened() and frame_count < 150:  # Analyze first 150 frames
            ret, frame = cap.read()
            if not ret: break

            # 1. Route to Moiré Module
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            moire_score = self.moire_module.analyze_frame(gray_frame)
            moire_scores.append(moire_score)

            # 2. Route to rPPG Module
            mean_g = self.rppg_module.extract_signal(frame)
            if mean_g is not None:
                rppg_buffer.append(mean_g)

            # 3. Route to Deepfake Module
            if frame_count % int(max(1, fps // 3)) == 0 and len(frames_for_cnn) < 16:
                frames_for_cnn.append(frame)

            frame_count += 1

        cap.release()

        # Final Calculations using the modules' internal logic
        avg_moire = np.mean(moire_scores) if moire_scores else 0
        bpm = self.rppg_module.calculate_bpm(rppg_buffer)
        deepfake_score = self.deepfake_module.infer(frames_for_cnn)

        return {
            "replay_attack_score": float(avg_moire),
            "is_replay_attack": bool(avg_moire > self.moire_module.threshold),
            "biological_bpm": float(bpm),
            "is_lively": bool(45 <= bpm <= 150),
            "ai_manipulation_score": float(deepfake_score),
            "is_deepfake": bool(deepfake_score > 0.5)
        }