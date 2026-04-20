import cv2
import numpy as np
import torch
import time
from torchvision.transforms import v2 as T
from core_engine import KYCOrchestrator


def run_live_demo():
    print("=============================================")
    print(" EXPERIMENT 7: LIVE WEBCAM HUD DEMO")
    print("=============================================")

    engine = KYCOrchestrator()

    # Open the M2 Webcam
    cap = cv2.VideoCapture(0)

    # Lock exposure on macOS to prevent auto-exposure drift from ruining rPPG/PRNU
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    print("[System] Webcam active. Press 'q' to quit.")

    frames_for_ftca = []
    ai_score = 0.500
    is_deepfake = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # 1. Run Physical Sensors (PRNU & Moire)
        engine.prnu_module.process_frame(frame)
        moire_score = engine.replay_module.analyze_frame(frame)
        if isinstance(moire_score, tuple): moire_score = moire_score[0]

        # 2. Run Biological Sensor (rPPG)
        rppg_state, _ = engine.rppg_module.process_frame(frame)

        # 3. Buffer 16 frames for the AI Module
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = engine.face_detector(rgb_frame)

        if face is not None:
            # Draw Face Bounding Box
            h, w, _ = frame.shape
            cv2.rectangle(display_frame, (w // 2 - 112, h // 2 - 112), (w // 2 + 112, h // 2 + 112), (255, 255, 0), 2)

            face_float = face / 255.0
            frames_for_ftca.append(face_float)

            # 4. Trigger FTCA Inference every 16 frames
            if len(frames_for_ftca) == 16:
                processed = [engine.ftca_normalize(f) for f in frames_for_ftca]
                video_tensor = torch.stack(processed).permute(1, 0, 2, 3).unsqueeze(0).to(engine.device)

                with torch.no_grad():
                    logits = engine.ftca_module(video_tensor)
                    ai_score = torch.sigmoid(logits).item()
                    is_deepfake = bool(ai_score > 0.50)

                # Clear buffer for the next chunk
                frames_for_ftca.clear()

        # --- DRAW THE FUTURISTIC HUD ---
        # S1: PRNU Status
        cv2.putText(display_frame, f"S1 PRNU: Accumulating...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # S2: Moire Status
        # Low moire score = replay attack (screen destroys high-freq detail)
        s2_color = (0, 0, 255) if moire_score < engine.replay_module.threshold else (0, 255, 0)
        cv2.putText(display_frame, f"S2 Moire: {moire_score:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s2_color, 2)

        # S3: rPPG Status
        bpm = rppg_state.get("bpm", 0.0)
        s3_color = (0, 255, 0) if bpm > 45 else (0, 165, 255)
        pulse_text = f"S3 Pulse: {bpm:.1f} BPM" if bpm > 0 else f"S3 Pulse: Buffering {int(rppg_state.get('buffer_fill_ratio', 0) * 100)}%"
        cv2.putText(display_frame, pulse_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s3_color, 2)

        # S4: FTCA Status
        s4_color = (0, 0, 255) if is_deepfake else (0, 255, 0)
        cv2.putText(display_frame, f"S4 AI Score: {ai_score:.3f} {'[FAKE]' if is_deepfake else '[REAL]'}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, s4_color, 2)

        cv2.imshow('AuthKYC Live Security Feed', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()