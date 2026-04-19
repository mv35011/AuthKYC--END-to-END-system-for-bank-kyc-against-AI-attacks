import cv2
import numpy as np
import os
import torch
from torchvision.transforms import v2 as T
from facenet_pytorch import MTCNN

# Import all 4 independent security modules
from modules.moire_detector import ReplayAttackDetector
from modules.rppg_extractor import AdvancedrPPGDetector
from modules.prnu_forensics import PRNUDetector
from modules.ftca_module import FTCABlock


class KYCOrchestrator:
    def __init__(self):
        print("[Engine] Initializing 4-Layer PAD System...")
        self.replay_module = ReplayAttackDetector(threshold=1500)
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

        # Security Update: Load Domain-Adapted Golden Weights securely
        if os.path.exists("best_ftca_pad_model.pth"):
            self.ftca_module.load_state_dict(torch.load("best_ftca_pad_model.pth", map_location=self.device, weights_only=True), strict=False)
            print(f"[System] Domain-Adapted FTCA Weights Loaded Successfully on {self.device}!")
        else:
            print("[WARNING] Weights not found. Using untrained model.")

        self.ftca_module.eval()

        # FIX: MTCNN crashes on Apple Silicon (MPS) due to a known PyTorch pooling bug.
        # We force MTCNN to run on the CPU, while keeping the heavy FTCA model on the M2 GPU.
        mtcnn_device = 'cpu' if self.device.type == 'mps' else self.device

        self.face_detector = MTCNN(
            image_size=224, margin=40, keep_all=False,
            post_process=False, device=mtcnn_device
        )

        # Normalization matching the training dataset.py
        self.ftca_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        moire_scores = []
        frames_for_ftca = []
        rppg_results = {"bpm": 0.0, "snr_db": 0.0, "passed": False, "buffer_fill_ratio": 0.0}

        # FIX: Reset rPPG buffers so previous video's pulse doesn't leak
        self.rppg_module.reset()

        frame_count = 0
        # Analyze 180 frames (6 seconds) to ensure buffer and tensor stability
        while cap.isOpened() and frame_count < 180:
            ret, frame = cap.read()
            if not ret: break

            # Stage 1: PRNU Sensor Fingerprinting
            self.prnu_module.process_frame(frame)

            # Stage 2: Moiré/Replay Grid Detection
            moire_output = self.replay_module.analyze_frame(frame)
            score_only = moire_output[0] if isinstance(moire_output, tuple) else moire_output
            moire_scores.append(score_only)

            # Stage 3: Biological Liveness (rPPG)
            rppg_state, _ = self.rppg_module.process_frame(frame)
            # FIX: Buffer Starvation Bug. Lock in the results the moment we get a valid pulse reading!
            if rppg_state["bpm"] > 0:
                rppg_results = rppg_state

            # Stage 4: AI Manipulation Prep — MTCNN Face Detection
            # FIX: Removed the "fps // 3" skip. We MUST extract 16 contiguous frames
            # so the 3D Temporal CNN and Frequency Encoder can see actual motion.
            if len(frames_for_ftca) < 16:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.face_detector(rgb_frame)

                # Only append valid MTCNN-cropped faces, never full frames
                if face is not None:
                    # MTCNN returns uint8 [0,255] tensor -> convert to float [0,1]
                    face_float = face / 255.0
                    frames_for_ftca.append(face_float)

            frame_count += 1
        cap.release()

        # --- INDEPENDENT WATERFALL LOGIC ---

        # 1. Camera Authenticity (PRNU)
        prnu_energy, is_physical = self.prnu_module.analyze_fingerprint()

        # 2. Presentation Attack (Moiré Score)
        # Screen replays LOSE high-frequency detail due to double compression
        # (original → screen render → camera → encode), so they score LOW.
        # Real cameras preserve high-freq texture. Score below threshold = replay.
        avg_moire = np.mean(moire_scores) if moire_scores else 0
        is_replay = bool(avg_moire < self.replay_module.threshold)

        # 3. Biological Context (S3)
        bpm = rppg_results.get("bpm", 0.0)
        snr = rppg_results.get("snr_db", 0.0)
        is_lively = rppg_results.get("passed", False) or (snr > 2.5 and 45 <= bpm <= 120)

        # 4. AI Manipulation Inference (S4) — tensor assembly matches training pipeline
        ai_score = 0.0
        if len(frames_for_ftca) >= 16:
            # Normalize each face tensor exactly like finetune/dataset.py
            processed = [self.ftca_normalize(f) for f in frames_for_ftca[:16]]

            # Stack as [16, 3, 224, 224] → permute to [3, 16, 224, 224] → add batch dim
            video_tensor = torch.stack(processed).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.ftca_module(video_tensor)
                ai_score = torch.sigmoid(logits).item()

        # --- FINAL DECISION WITH BIOLOGICAL VOUCHING ---
        # Base threshold for AI manipulation detection
        base_threshold = 0.65

        # If S1 (PRNU) confirms a real physical sensor AND S3 (rPPG) confirms a
        # real heartbeat, we have strong independent physical evidence of a live human.
        # In this case, we raise the FTCA threshold to 0.85 — the model needs to be
        # very confident it's a deepfake to override two independent physical proofs.
        # This compensates for the partially-trained frequency encoder (layer4 random init).
        is_biologically_vouched = is_physical and is_lively and not is_replay
        dynamic_threshold = 0.85 if is_biologically_vouched else base_threshold

        is_deepfake = bool(ai_score > dynamic_threshold)

        return {
            "prnu_energy": float(prnu_energy),
            "is_virtual_camera": not is_physical,
            "moire_score": float(avg_moire),
            "is_replay_attack": is_replay,
            "biological_bpm": float(bpm),
            "rppg_snr": float(snr),
            "is_lively": is_lively,
            "ai_manipulation_score": float(ai_score),
            "is_deepfake": is_deepfake
        }