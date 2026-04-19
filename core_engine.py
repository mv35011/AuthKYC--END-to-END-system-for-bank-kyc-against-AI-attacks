import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image

# Import all 4 independent security modules
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector
from modules.ftca_module import FTCABlock


class KYCOrchestrator:
    def __init__(self):
        print("[Engine] Initializing 4-Layer PAD System...")
        self.replay_module = ReplayAttackDetector(threshold=120000)
        self.rppg_module = AdvancedrPPGDetector(fps=30)
        self.prnu_module = PRNUDetector(energy_threshold=0.5)

        # Initialize FTCA Spatio-Temporal Model
        self.ftca_module = FTCABlock()

        # Hardware Acceleration: Use MPS (Apple Silicon), CUDA, or CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.ftca_module.to(self.device)

        # Load Golden Weights [cite: 338, 340]
        if os.path.exists("best_ftca_pad_model.pth"):
            self.ftca_module.load_state_dict(torch.load("best_ftca_pad_model.pth", map_location=self.device))
            print(f"[System] Golden FTCA Weights Loaded Successfully on {self.device}!")
        else:
            print("[WARNING] Weights not found. Using untrained model.")

        self.ftca_module.eval()

        # Domain Matching: Face Cropper to ensure FTCA only sees faces [cite: 71]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        moire_scores = []
        frames_for_ftca = []
        rppg_results = {"bpm": 0.0, "snr_db": 0.0, "passed": False, "buffer_fill_ratio": 0.0}

        frame_count = 0
        # Analyze 180 frames (6 seconds) to ensure buffer and tensor stability [cite: 117, 307]
        while cap.isOpened() and frame_count < 180:
            ret, frame = cap.read()
            if not ret: break

            # Stage 1: PRNU Sensor Fingerprinting [cite: 137, 141]
            self.prnu_module.process_frame(frame)

            # Stage 2: Moiré/Replay Grid Detection [cite: 172, 178]
            moire_output = self.replay_module.analyze_frame(frame)
            score_only = moire_output[0] if isinstance(moire_output, tuple) else moire_output
            moire_scores.append(score_only)

            # Stage 3: Biological Liveness (rPPG) [cite: 217, 221]
            rppg_state, _ = self.rppg_module.process_frame(frame)
            if rppg_state["buffer_fill_ratio"] >= 1.0:
                rppg_results = rppg_state

                # Stage 4: AI Manipulation Prep (Face Cropping) [cite: 71]
            if frame_count % int(max(1, fps // 3)) == 0 and len(frames_for_ftca) < 16:
                gray_for_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_for_crop, 1.1, 5)

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    pad = 40
                    y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
                    x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
                    frames_for_ftca.append(frame[y1:y2, x1:x2])
                else:
                    frames_for_ftca.append(frame)

            frame_count += 1
        cap.release()

        # --- HEURISTIC FUSION LAYER (PATENT INTENT LOGIC) [cite: 383, 385] ---

        # 1. Camera Authenticity (PRNU) [cite: 145]
        prnu_energy, is_physical = self.prnu_module.analyze_fingerprint()

        # 2. Presentation Attack (Moiré PMR Score) [cite: 180]
        avg_moire = np.mean(moire_scores) if moire_scores else 0
        is_replay = bool(avg_moire > self.replay_module.threshold)

        # 3. Biological Context (S3) [cite: 222]
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        # S3 "Vouching": A high SNR pulse is a strong physical proof of life [cite: 319, 320]
        is_biological_confirmed = (snr > 2.5 and 45 <= bpm <= 120)

        # 4. AI Manipulation Inference (S4) [cite: 257]
        ai_score = 0.0
        if len(frames_for_ftca) >= 16:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            processed_tensors = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in
                                 frames_for_ftca[:16]]
            video_tensor = torch.stack(processed_tensors, dim=1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.ftca_module(video_tensor)
                ai_score = torch.sigmoid(logits).item()

        # --- FINAL WATERFALL DECISION ---
        # If biological life is confirmed, we increase AI tolerance to 0.75 [cite: 317, 385]
        dynamic_threshold = 0.75 if is_biological_confirmed else 0.5
        is_deepfake = bool(ai_score > dynamic_threshold)

        return {
            "prnu_energy": float(prnu_energy),
            "is_virtual_camera": not is_physical,
            "moire_score": float(avg_moire),
            "is_replay_attack": is_replay,
            "biological_bpm": float(bpm),
            "rppg_snr": float(snr),
            "is_lively": rppg_results.get("passed", False) or is_biological_confirmed,
            "ai_manipulation_score": float(ai_score),
            "is_deepfake": is_deepfake,
            "dynamic_threshold_used": dynamic_threshold
        }