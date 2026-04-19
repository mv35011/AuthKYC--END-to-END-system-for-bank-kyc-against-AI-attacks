import time
import os
from core_engine import KYCOrchestrator


def run_ablation_study():
    print("=============================================")
    print(" EXPERIMENT 6: WATERFALL ABLATION PIPELINE")
    print("=============================================")

    engine = KYCOrchestrator()

    # --- SETUP YOUR TEST FOLDERS HERE ---
    # Put 5-10 short video clips in each of these folders
    test_categories = {
        "Real Humans": "./videos/real",
        "Screen Replays (Phone)": "./videos/Replay attack",
        "Deepfakes (FF++)": "./videos/fake"
    }

    for category, folder in test_categories.items():
        if not os.path.exists(folder):
            print(f"Skipping {category}: Folder '{folder}' not found.")
            continue

        print(f"\n--- Testing Category: {category} ---")

        for filename in os.listdir(folder):
            if not filename.endswith(('.mp4', '.avi')): continue

            video_path = os.path.join(folder, filename)
            start_time = time.perf_counter()

            # Run the waterfall
            results = engine.analyze_video(video_path)

            latency = (time.perf_counter() - start_time) * 1000  # in ms

            # Determine which stage rejected it
            rejection_stage = "PASSED"
            if results["is_virtual_camera"]:
                rejection_stage = "S1 (PRNU/Telemetry)"
            elif results["is_replay_attack"]:
                rejection_stage = "S2 (Moiré)"
            elif not results["is_lively"]:
                rejection_stage = "S3 (rPPG)"
            elif results["is_deepfake"]:
                rejection_stage = "S4 (FTCA)"

            print(f"File: {filename[:15]:<15} | Latency: {latency:.1f}ms | Result: {rejection_stage}")


if __name__ == "__main__":
    run_ablation_study()