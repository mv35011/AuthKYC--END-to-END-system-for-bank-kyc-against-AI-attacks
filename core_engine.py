import cv2
import numpy as np

# Import all 4 independent security modules
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector
from modules.ftca_module import FTCABlock
import torch


class KYCOrchestrator:
    def __init__(self):
        print("[Engine] Initializing 4-Layer PAD System...")
        self.replay_module = ReplayAttackDetector(threshold=120000)
        self.rppg_module = AdvancedrPPGDetector(fps=30)
        self.prnu_module = PRNUDetector(energy_threshold=0.5)

        # Initialize FTCA (using CPU/MPS fallback if weights aren't present yet)
        self.ftca_module = FTCABlock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ftca_module.to(self.device)
        self.ftca_module.eval()

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        moire_scores = []
        frames_for_ftca = []
        rppg_results = {}

        frame_count = 0
        while cap.isOpened() and frame_count < 150:  # Analyze 5 seconds
            ret, frame = cap.read()
            if not ret: break

            # 1. PRNU Sensor Extraction
            self.prnu_module.process_frame(frame)

            # 2. Replay/Moire Extraction
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            moire_score = self.replay_module.analyze_frame(gray_frame)
            moire_scores.append(moire_score)

            # 3. Biological Liveness (rPPG) Extraction
            rppg_state, _ = self.rppg_module.process_frame(frame)
            if rppg_state["buffer_fill_ratio"] >= 1.0:
                rppg_results = rppg_state  # Keep the latest valid reading

            # 4. Deepfake FTCA Spatio-Temporal Extraction
            if frame_count % int(max(1, fps // 3)) == 0 and len(frames_for_ftca) < 16:
                frames_for_ftca.append(frame)

            frame_count += 1
        cap.release()

        # --- FUSION LAYER CALCULATIONS ---

        # Module 1: Camera Authenticity (PRNU)
        prnu_energy, is_physical = self.prnu_module.analyze_fingerprint()

        # Module 2: Presentation Attack (Moiré)
        avg_moire = np.mean(moire_scores) if moire_scores else 0
        is_replay = bool(avg_moire > self.replay_module.threshold)

        # Module 3: Biological Liveness
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        is_alive = rppg_results.get("passed", False)

        # Module 4: AI Manipulation (FTCA)
        ai_score = 0.0
        is_deepfake = False
        if len(frames_for_ftca) == 16:
            # Prepare tensor for FTCA block: [1, 3, 16, 224, 224]
            # (Assuming standard resizing/normalization here for brevity)
            tensor_seq = torch.randn(1, 3, 16, 224, 224).to(self.device)  # Dummy tensor for now
            with torch.no_grad():
                logits = self.ftca_module(tensor_seq)
                ai_score = torch.sigmoid(logits).item()
                is_deepfake = bool(ai_score > 0.5)

        return {
            "prnu_energy": float(prnu_energy),
            "is_virtual_camera": not is_physical,
            "moire_score": float(avg_moire),
            "is_replay_attack": is_replay,
            "biological_bpm": float(bpm),
            "rppg_snr": float(snr),
            "is_lively": is_alive,
            "ai_manipulation_score": float(ai_score),
            "is_deepfake": is_deepfake
        }