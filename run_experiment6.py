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
        "Real Humans": ("./videos/real", "real"),
        "Screen Replays (Phone)": ("./videos/Replay attack", "attack"),
        "Deepfakes (FF++)": ("./videos/fake", "fake"),
    }

    # Aggregate metrics across all categories
    all_results = []

    for category, (folder, ground_truth) in test_categories.items():
        if not os.path.exists(folder):
            print(f"Skipping {category}: Folder '{folder}' not found.")
            continue

        print(f"\n--- Testing Category: {category} ---")
        print(f"{'File':<20} | {'Latency':>8} | {'S1-PRNU':>8} | {'S2-Moire':>9} | {'S3-rPPG':>8} | {'S4-FTCA':>8} | {'Result':<20}")
        print("-" * 110)

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(('.mp4', '.avi', '.mov', '.webm')): continue

            video_path = os.path.join(folder, filename)
            start_time = time.perf_counter()

            # Run the full waterfall
            results = engine.analyze_video(video_path)

            latency = (time.perf_counter() - start_time) * 1000  # in ms

            # --- THE DYNAMIC FALLBACK LOGIC ---
            s1_prnu_pass = not results["is_virtual_camera"]
            s2_moire_pass = not results["is_replay_attack"]
            s3_rppg_pass = results["is_lively"]
            s4_ftca_pass = not results["is_deepfake"]

            # If PRNU fails (due to phone stabilization) but we detect a real human pulse
            # and no AI manipulation, the biological evidence overrides the hardware failure.
            if not s1_prnu_pass and s3_rppg_pass and s4_ftca_pass:
                s1_pass = True  # Dynamically Overridden
                prnu_display = "OVRIDE" # Show this clearly in the terminal
            else:
                s1_pass = s1_prnu_pass
                prnu_display = "PASS" if s1_prnu_pass else "FAIL"

            # Determine waterfall rejection point
            if not s1_pass:
                rejection_stage = "REJECTED: S1 (Virtual Cam)"
            elif not s2_moire_pass:
                rejection_stage = "REJECTED: S2 (Moiré)"
            elif not s3_rppg_pass:
                rejection_stage = "REJECTED: S3 (No Pulse)"
            elif not s4_ftca_pass:
                rejection_stage = "REJECTED: S4 (Deepfake)"
            else:
                rejection_stage = "PASSED (All Clear)"

            print(
                f"{filename[:20]:<20} | {latency:>6.0f}ms | "
                f"{prnu_display:>8} | "
                f"{'PASS' if s2_moire_pass else 'FAIL':>9} | "
                f"{'PASS' if s3_rppg_pass else 'FAIL':>8} | "
                f"{results['ai_manipulation_score']:>7.3f}{'✓' if s4_ftca_pass else '✗'} | "
                f"{rejection_stage:<20}"
            )

            all_results.append({
                "file": filename,
                "category": category,
                "ground_truth": ground_truth,
                "s1_pass": s1_pass,
                "s2_pass": s2_moire_pass,
                "s3_pass": s3_rppg_pass,
                "s4_pass": s4_ftca_pass,
                "ai_score": results["ai_manipulation_score"],
                "waterfall_passed": s1_pass and s2_moire_pass and s3_rppg_pass and s4_ftca_pass,
                "latency_ms": latency,
            })

    # --- SUMMARY METRICS ---
    if not all_results:
        print("\n[WARNING] No videos were processed.")
        return

    print("\n\n" + "=" * 60)
    print(" ABLATION SUMMARY")
    print("=" * 60)

    # Per-category breakdown
    for category, (_, gt_label) in test_categories.items():
        category_results = [r for r in all_results if r["ground_truth"] == gt_label]
        if not category_results:
            continue

        total = len(category_results)
        expected_pass = (gt_label == "real")  # Real should pass, attacks/fakes should be rejected

        if expected_pass:
            correct = sum(1 for r in category_results if r["waterfall_passed"])
        else:
            correct = sum(1 for r in category_results if not r["waterfall_passed"])

        accuracy = correct / total * 100

        # Per-stage false rejection/acceptance
        s1_issues = sum(1 for r in category_results if r["s1_pass"] != expected_pass)
        s2_issues = sum(1 for r in category_results if r["s2_pass"] != expected_pass)
        s3_issues = sum(1 for r in category_results if r["s3_pass"] != expected_pass)
        s4_issues = sum(1 for r in category_results if r["s4_pass"] != expected_pass)

        print(f"\n  {category} ({total} videos)")
        print(f"    Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"    Stage Errors:  S1={s1_issues}  S2={s2_issues}  S3={s3_issues}  S4={s4_issues}")
        if not expected_pass:
            # Show which stage actually caught the attacks
            caught_s1 = sum(1 for r in category_results if not r["s1_pass"])
            caught_s2 = sum(1 for r in category_results if r["s1_pass"] and not r["s2_pass"])
            caught_s3 = sum(1 for r in category_results if r["s1_pass"] and r["s2_pass"] and not r["s3_pass"])
            caught_s4 = sum(1 for r in category_results if r["s1_pass"] and r["s2_pass"] and r["s3_pass"] and not r["s4_pass"])
            escaped = sum(1 for r in category_results if r["waterfall_passed"])
            print(f"    Caught by:  S1={caught_s1}  S2={caught_s2}  S3={caught_s3}  S4={caught_s4}  Escaped={escaped}")

    # Overall confusion matrix
    total_all = len(all_results)
    tp = sum(1 for r in all_results if r["ground_truth"] == "real" and r["waterfall_passed"])
    fn = sum(1 for r in all_results if r["ground_truth"] == "real" and not r["waterfall_passed"])
    fp = sum(1 for r in all_results if r["ground_truth"] != "real" and r["waterfall_passed"])
    tn = sum(1 for r in all_results if r["ground_truth"] != "real" and not r["waterfall_passed"])

    overall_acc = (tp + tn) / total_all * 100 if total_all > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in all_results) / total_all

    print(f"\n  {'─' * 40}")
    print(f"  CONFUSION MATRIX")
    print(f"                    Predicted PASS  Predicted REJECT")
    print(f"    Actual Real:      TP={tp:<6}       FN={fn:<6}")
    print(f"    Actual Attack:    FP={fp:<6}       TN={tn:<6}")
    print(f"\n  OVERALL ACCURACY: {overall_acc:.1f}% ({tp + tn}/{total_all})")
    print(f"  AVG LATENCY: {avg_latency:.0f}ms per video")
    print("=" * 60)


if __name__ == "__main__":
    run_ablation_study()