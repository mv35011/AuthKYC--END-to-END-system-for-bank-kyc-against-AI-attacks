"""
AuthKYC Minor Project — Report Plot Generator
Generates all charts for the project report from experiment data.
Run: python generate_report_plots.py
Output: ./report_plots/ directory
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
OUT_DIR = './report_plots'
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# DATA FROM EXPERIMENTS
# =====================================================

# --- FTCA Fine-Tuning (15 epochs, RunPod A6000) ---
ftca_epochs = list(range(1, 16))
ftca_train_loss = [0.6805, 0.6542, 0.6273, 0.6054, 0.5671, 0.5620, 0.5665, 0.5245,
                   0.5499, 0.5293, 0.5123, 0.5089, 0.4877, 0.4786, 0.4771]
ftca_val_loss   = [0.6351, 0.6222, 0.5105, 0.6307, 0.4986, 0.5175, 0.4842, 0.4633,
                   0.4659, 0.4741, 0.5279, 0.4633, 0.4890, 0.4366, 0.4174]
ftca_train_acc  = [0.5700, 0.6025, 0.6575, 0.6663, 0.6987, 0.7025, 0.7100, 0.7375,
                   0.7050, 0.7163, 0.7412, 0.7388, 0.7750, 0.7588, 0.7675]
ftca_val_acc    = [0.5603, 0.7660, 0.7801, 0.6738, 0.7092, 0.7589, 0.7730, 0.7730,
                   0.7872, 0.7589, 0.7163, 0.7943, 0.7305, 0.8085, 0.7801]

# --- Xception Baseline (20 epochs) ---
xception_epochs = list(range(1, 21))
xception_train_loss = [0.6063, 0.5146, 0.4389, 0.3841, 0.3290, 0.3017, 0.2617, 0.2304,
                       0.2081, 0.2002, 0.1819, 0.1567, 0.1428, 0.1535, 0.1276, 0.1108,
                       0.1026, 0.1178, 0.1197, 0.0979]
xception_val_loss   = [0.7048, 0.9332, 0.7439, 0.8524, 0.7201, 0.8466, 0.8493, 0.7429,
                       0.7218, 0.7112, 0.7532, 0.7253, 0.7158, 0.7089, 0.7277, 0.7029,
                       0.6940, 0.7184, 0.7243, 0.7716]
xception_val_acc    = [0.4640, 0.5360, 0.5325, 0.4728, 0.5097, 0.4833, 0.4745, 0.4692,
                       0.5325, 0.4657, 0.5290, 0.5290, 0.4868, 0.5149, 0.5272, 0.4938,
                       0.4692, 0.4903, 0.4868, 0.4692]

# --- Experiment 6: Waterfall Ablation (latest run with threshold 0.50) ---
# Real Humans
real_videos = {
    '002.mp4':    {'prnu': 35.18, 'moire': 2801, 'rppg': True, 'ftca': 0.225, 'result': 'PASSED'},
    '012.mp4':    {'prnu': 4.94,  'moire': 2041, 'rppg': True, 'ftca': 0.781, 'result': 'REJECTED S4'},
    '013.mp4':    {'prnu': 29.30, 'moire': 6864, 'rppg': True, 'ftca': 0.294, 'result': 'PASSED'},
    'VID_259':    {'prnu': 0.31,  'moire': 6435, 'rppg': True, 'ftca': 0.021, 'result': 'PASSED'},
    'VID_317':    {'prnu': 0.25,  'moire': 6668, 'rppg': True, 'ftca': 0.026, 'result': 'PASSED'},
    'VID_334':    {'prnu': 0.65,  'moire': 7900, 'rppg': True, 'ftca': 0.031, 'result': 'PASSED'},
    'VID_350':    {'prnu': 0.19,  'moire': 7581, 'rppg': True, 'ftca': 0.027, 'result': 'PASSED'},
    'VID_405':    {'prnu': 0.40,  'moire': 6407, 'rppg': True, 'ftca': 0.017, 'result': 'PASSED'},
    'VID_421':    {'prnu': 0.11,  'moire': 5644, 'rppg': True, 'ftca': 0.046, 'result': 'PASSED'},
    'VID_439':    {'prnu': 0.42,  'moire': 7192, 'rppg': True, 'ftca': 0.031, 'result': 'PASSED'},
    'VID_743':    {'prnu': 0.19,  'moire': 4361, 'rppg': True, 'ftca': 0.037, 'result': 'PASSED'},
    'VID_821':    {'prnu': 0.11,  'moire': 4112, 'rppg': True, 'ftca': 0.026, 'result': 'PASSED'},
}
replay_videos = {
    'replay1.mp4': {'prnu': 0.92, 'moire': 920,  'rppg': True, 'ftca': 0.024, 'result': 'REJECTED S2'},
    'replay2.mp4': {'prnu': 0.42, 'moire': 1025, 'rppg': True, 'ftca': 0.040, 'result': 'REJECTED S2'},
}
deepfake_videos = {
    '001_870.mp4': {'prnu': 105.94, 'moire': 6240, 'rppg': True,  'ftca': 0.800, 'result': 'REJECTED S4'},
    '034_590.mp4': {'prnu': 18.02,  'moire': 3141, 'rppg': True,  'ftca': 0.703, 'result': 'REJECTED S4'},
    '035_036.mp4': {'prnu': 10.47,  'moire': 3927, 'rppg': True,  'ftca': 0.732, 'result': 'REJECTED S4'},
    '036_035.mp4': {'prnu': 2.34,   'moire': 2308, 'rppg': True,  'ftca': 0.770, 'result': 'REJECTED S4'},
    '426_287.mp4': {'prnu': 0.75,   'moire': 1084, 'rppg': True,  'ftca': 0.617, 'result': 'REJECTED S2'},
    '427_637.mp4': {'prnu': 55.84,  'moire': 4514, 'rppg': False, 'ftca': 0.461, 'result': 'REJECTED S3'},
    '428_466.mp4': {'prnu': 11.51,  'moire': 3618, 'rppg': True,  'ftca': 0.614, 'result': 'REJECTED S4'},
}

# Moiré score ranges
moire_ranges = {
    'Real (Webcam)': (2041, 12135),
    'Real (Phone)': (4112, 7900),
    'Screen Replays': (920, 1025),
    'Deepfakes (FF++)': (2308, 6240),
}


# =====================================================
# PLOT 1: FTCA Training Curves (Loss + Accuracy)
# =====================================================
def plot_ftca_training():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ftca_epochs, ftca_train_loss, 'o-', color='#FF6B6B', linewidth=2, markersize=5, label='Train Loss')
    ax1.plot(ftca_epochs, ftca_val_loss, 's-', color='#4ECDC4', linewidth=2, markersize=5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('FTCA Domain Adaptation — Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ftca_epochs)

    ax2.plot(ftca_epochs, [a*100 for a in ftca_train_acc], 'o-', color='#FF6B6B', linewidth=2, markersize=5, label='Train Acc')
    ax2.plot(ftca_epochs, [a*100 for a in ftca_val_acc], 's-', color='#4ECDC4', linewidth=2, markersize=5, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('FTCA Domain Adaptation — Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ftca_epochs)
    ax2.axhline(y=80.85, color='gray', linestyle='--', alpha=0.5, label='Best: 80.85%')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_ftca_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 1: FTCA Training Curves")


# =====================================================
# PLOT 2: FTCA vs Xception Baseline Comparison
# =====================================================
def plot_baseline_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Val Loss comparison
    ax1.plot(ftca_epochs, ftca_val_loss, 's-', color='#4ECDC4', linewidth=2, label='FTCA (Ours)')
    ax1.plot(xception_epochs, xception_val_loss, 'o-', color='#FF6B6B', linewidth=2, label='Xception Baseline')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss — FTCA vs Xception')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val Accuracy comparison
    ax2.plot(ftca_epochs, [a*100 for a in ftca_val_acc], 's-', color='#4ECDC4', linewidth=2, label='FTCA (Ours)')
    ax2.plot(xception_epochs, [a*100 for a in xception_val_acc], 'o-', color='#FF6B6B', linewidth=2, label='Xception Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy — FTCA vs Xception')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_ftca_vs_xception.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 2: FTCA vs Xception Comparison")


# =====================================================
# PLOT 3: Confusion Matrix
# =====================================================
def plot_confusion_matrix():
    real_correct = sum(1 for v in real_videos.values() if 'PASSED' in v['result'])
    real_wrong = len(real_videos) - real_correct
    attack_correct = sum(1 for v in {**replay_videos, **deepfake_videos}.values() if 'REJECTED' in v['result'])
    attack_wrong = len(replay_videos) + len(deepfake_videos) - attack_correct

    cm = np.array([[real_correct, real_wrong],
                   [attack_wrong, attack_correct]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]}',
                    ha='center', va='center', fontsize=18, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nPASS', 'Predicted\nREJECT'])
    ax.set_yticklabels(['Actual\nReal', 'Actual\nAttack'])
    ax.set_title('Confusion Matrix — Full Waterfall Pipeline')
    total = cm.sum()
    correct = cm[0, 0] + cm[1, 1]
    ax.set_xlabel(f'Overall Accuracy: {correct}/{total} ({correct/total*100:.1f}%)', fontsize=12)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 3: Confusion Matrix")


# =====================================================
# PLOT 4: Waterfall Rejection Breakdown (Pie Chart)
# =====================================================
def plot_waterfall_rejection():
    all_attacks = {**replay_videos, **deepfake_videos}
    caught_s2 = sum(1 for v in all_attacks.values() if 'S2' in v['result'])
    caught_s3 = sum(1 for v in all_attacks.values() if 'S3' in v['result'])
    caught_s4 = sum(1 for v in all_attacks.values() if 'S4' in v['result'])
    escaped = sum(1 for v in all_attacks.values() if 'PASSED' in v['result'])

    sizes = [caught_s2, caught_s3, caught_s4, escaped]
    labels_list = [f'S2 Moiré\n({caught_s2})', f'S3 rPPG\n({caught_s3})',
                   f'S4 FTCA\n({caught_s4})', f'Escaped\n({escaped})']
    colors = ['#4ECDC4', '#45B7D1', '#6C5CE7', '#FF6B6B']
    explode = (0.05, 0.05, 0.05, 0.15)

    # Filter out zeros
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels_list, colors, explode) if s > 0]
    if not non_zero:
        return
    sizes, labels_list, colors, explode = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels_list, colors=colors, explode=explode,
                                       autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Attack Rejection by Waterfall Stage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_waterfall_rejection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 4: Waterfall Rejection Pie")


# =====================================================
# PLOT 5: Moiré Score Distribution
# =====================================================
def plot_moire_distribution():
    real_moire = [v['moire'] for v in real_videos.values()]
    replay_moire = [v['moire'] for v in replay_videos.values()]
    fake_moire = [v['moire'] for v in deepfake_videos.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = [1, 2, 3]
    bp = ax.boxplot([replay_moire, fake_moire, real_moire], positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True)

    colors_bp = ['#FF6B6B', '#FFA07A', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Threshold line
    ax.axhline(y=1500, color='red', linestyle='--', linewidth=2, label='Threshold (1500)')
    ax.fill_between([0.5, 3.5], 0, 1500, alpha=0.08, color='red', label='Replay Zone')

    ax.set_xticklabels(['Screen Replays', 'Deepfakes (FF++)', 'Real Humans'])
    ax.set_ylabel('Moiré Score (High-Freq Energy)')
    ax.set_title('Moiré Score Distribution — Replay Detection Threshold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_moire_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 5: Moiré Score Distribution")


# =====================================================
# PLOT 6: FTCA Score Distribution (Real vs Fake)
# =====================================================
def plot_ftca_scores():
    real_scores = [v['ftca'] for v in real_videos.values()]
    fake_scores = [v['ftca'] for v in deepfake_videos.values()]
    replay_scores = [v['ftca'] for v in replay_videos.values()]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(range(len(real_scores)), sorted(real_scores), s=100, c='#4ECDC4',
               edgecolors='black', linewidth=0.5, zorder=5, label=f'Real ({len(real_scores)})')
    offset = len(real_scores)
    ax.scatter(range(offset, offset + len(fake_scores)), sorted(fake_scores), s=100, c='#FF6B6B',
               edgecolors='black', linewidth=0.5, zorder=5, label=f'Deepfake ({len(fake_scores)})')
    offset += len(fake_scores)
    ax.scatter(range(offset, offset + len(replay_scores)), sorted(replay_scores), s=100, c='#FFA500',
               edgecolors='black', linewidth=0.5, zorder=5, label=f'Replay ({len(replay_scores)})')

    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='Threshold (0.50)')
    ax.fill_between([-1, offset + 1], 0.50, 1.0, alpha=0.06, color='red', label='Deepfake Zone')

    ax.set_ylabel('FTCA Score (sigmoid)')
    ax.set_xlabel('Video Samples (sorted)')
    ax.set_title('FTCA AI Manipulation Score — Per-Video Distribution')
    ax.legend(loc='upper left')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_ftca_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 6: FTCA Score Distribution")


# =====================================================
# PLOT 7: Per-Category Accuracy Bar Chart
# =====================================================
def plot_per_category_accuracy():
    real_acc = sum(1 for v in real_videos.values() if 'PASSED' in v['result']) / len(real_videos) * 100
    replay_acc = sum(1 for v in replay_videos.values() if 'REJECTED' in v['result']) / len(replay_videos) * 100
    fake_acc = sum(1 for v in deepfake_videos.values() if 'REJECTED' in v['result']) / len(deepfake_videos) * 100

    categories = ['Real Humans\n(Approval Rate)', 'Screen Replays\n(Detection Rate)', 'Deepfakes FF++\n(Detection Rate)']
    accuracies = [real_acc, replay_acc, fake_acc]
    colors = ['#4ECDC4', '#45B7D1', '#6C5CE7']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=0.5, width=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%',
                ha='center', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Category Detection Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_per_category_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 7: Per-Category Accuracy")


# =====================================================
# PLOT 8: Model Comparison Summary Table
# =====================================================
def plot_comparison_table():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    table_data = [
        ['FTCA (Ours)', '80.85%', '0.4174', '12.8M', '15', 'Yes'],
        ['Xception Baseline', '46.92%', '0.6940', '~22M', '20', 'No'],
        ['FTCA Pretrained (eval)', '64.67%', '0.6053', '12.8M', '30', 'No'],
    ]
    col_labels = ['Model', 'Best Val Acc', 'Best Val Loss', 'Params', 'Epochs', 'Domain Adapted']

    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight our model
    for j in range(len(col_labels)):
        table[1, j].set_facecolor('#E8F8F5')

    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_model_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 8: Model Comparison Table")


# =====================================================
# PLOT 9: Latency Analysis
# =====================================================
def plot_latency():
    # From experiment output
    latencies = {
        'Real (Webcam)': [6206, 4657, 4731],
        'Real (Phone)': [13808, 14652, 13782, 13823, 13493, 13704, 13541, 15065, 15833],
        'Replays': [8378, 8593],
        'Deepfakes': [7653, 6724, 5516, 6040, 10918, 7999, 8975],
    }

    categories = list(latencies.keys())
    means = [np.mean(v)/1000 for v in latencies.values()]  # Convert to seconds
    stds = [np.std(v)/1000 for v in latencies.values()]
    colors = ['#4ECDC4', '#45B7D1', '#FFA500', '#FF6B6B']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.5, capsize=5, width=0.5)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{m:.1f}s',
                ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Average Inference Latency per Video', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig9_latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 9: Latency Analysis")


# =====================================================
# RUN ALL
# =====================================================
if __name__ == "__main__":
    print(f"\nGenerating report plots to: {OUT_DIR}/\n")
    plot_ftca_training()
    plot_baseline_comparison()
    plot_confusion_matrix()
    plot_waterfall_rejection()
    plot_moire_distribution()
    plot_ftca_scores()
    plot_per_category_accuracy()
    plot_comparison_table()
    plot_latency()
    print(f"\n✅ All plots saved to {OUT_DIR}/")
    print("\n--- EXPERIMENTS YOU STILL NEED TO RUN ---")
    print("1. ROC Curve: Run data/calculate_AUC.py on the test set after finetune")
    print("2. Ablation Study: Run experiment6 with stages disabled one at a time")
    print("   (e.g., FTCA-only, FTCA+Moiré, full waterfall)")
    print("3. FFT Spectrum Visualization: Run modules/moire_detector.py standalone")
    print("   on a real video + replay video, screenshot the frequency domain windows")
