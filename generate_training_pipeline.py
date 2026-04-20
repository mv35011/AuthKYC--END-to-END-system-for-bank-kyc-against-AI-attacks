"""
AuthKYC — Training Pipeline Architecture Diagram (Publication Quality)
Generates a clean, academic-style figure suitable for a conference/journal paper.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = './report_plots'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Academic Color Palette (muted, print-friendly) ──
PAL = {
    'data':      '#D6EAF8',  # light blue
    'process':   '#FADBD8',  # light coral
    'train':     '#D5F5E3',  # light green
    'output':    '#FCF3CF',  # light yellow
    'finetune':  '#E8DAEF',  # light purple
    'border':    '#2C3E50',  # dark slate
    'text':      '#1B2631',  # near-black
    'arrow':     '#566573',  # gray
    'bg':        '#FFFFFF',  # white
    'phase_bg':  '#F8F9F9',  # near-white
    'accent':    '#E74C3C',  # red accent for key outputs
}


def box(ax, cx, cy, w, h, label, fill, fontsize=8.5, lw=1.0, fontstyle='normal', fontweight='normal'):
    """Draw a clean rounded box with centered multi-line text."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.12', facecolor=fill,
        edgecolor=PAL['border'], linewidth=lw, zorder=3
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize,
            color=PAL['text'], fontweight=fontweight, fontstyle=fontstyle,
            zorder=4, linespacing=1.35, family='serif')


def arrow(ax, x1, y1, x2, y2, label='', lw=1.2, color=None, rad=0.0):
    """Draw a clean arrow with optional label."""
    c = color or PAL['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=2, shrinkB=2),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                fontsize=7, color=c, fontstyle='italic', family='serif')


def phase_label(ax, x, y, text):
    """Draw a phase label in small caps style."""
    ax.text(x, y, text, ha='left', va='center', fontsize=9,
            fontweight='bold', color=PAL['border'], family='serif',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=PAL['phase_bg'],
                      edgecolor=PAL['border'], linewidth=0.8))


def generate():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 11.5)
    ax.axis('off')
    fig.patch.set_facecolor(PAL['bg'])

    # ════════════════════════════════════════════
    # TITLE
    # ════════════════════════════════════════════
    ax.text(8.25, 11.0, 'Training and Evaluation Pipeline',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color=PAL['text'], family='serif')
    ax.text(8.25, 10.6, 'Data extraction, model training, domain adaptation, and waterfall ablation',
            ha='center', va='center', fontsize=9, color='#7F8C8D', family='serif',
            fontstyle='italic')

    # ════════════════════════════════════════════
    # ROW 1: DATA SOURCES (y ≈ 9.5)
    # ════════════════════════════════════════════
    phase_label(ax, 0, 9.5, 'Phase 1')

    box(ax, 3.0, 9.5, 2.6, 0.85,
        'FaceForensics++ (C23)\n5 manipulation methods\n+ original sequences', PAL['data'], fontsize=8)
    box(ax, 6.0, 9.5, 2.2, 0.85,
        'Celeb-DF v2\nCeleb-real\nYouTube-real', PAL['data'], fontsize=8)

    # Merge arrow into balancer
    arrow(ax, 4.3, 9.5, 7.8, 8.5)
    arrow(ax, 6.0, 9.08, 7.8, 8.5)

    # ════════════════════════════════════════════
    # ROW 2: BALANCING + EXTRACTION (y ≈ 8.2)
    # ════════════════════════════════════════════
    box(ax, 9.5, 8.5, 3.0, 0.7,
        'Class Balancing\n1500 real  :  1500 fake\n80/20 train/val split', PAL['process'], fontsize=8)

    arrow(ax, 11.0, 8.5, 12.5, 8.5)

    box(ax, 14.2, 8.5, 2.8, 0.7,
        'MTCNN Face Extraction\n224×224, margin=40\nbatch size 32', PAL['process'], fontsize=8)

    # ════════════════════════════════════════════
    # ROW 3: TENSOR OUTPUT (y ≈ 7.2)
    # ════════════════════════════════════════════
    arrow(ax, 14.2, 8.15, 14.2, 7.55)

    box(ax, 14.2, 7.2, 3.2, 0.55,
        '.pt tensors  [N, 16, 3, 224, 224]\n8 sequences per video', PAL['output'], fontsize=8, fontweight='bold')

    # Connect tensors down to both training branches
    arrow(ax, 13.0, 7.2, 5.5, 6.0, label='train split')
    arrow(ax, 14.2, 6.93, 11.5, 6.0, label='val split')

    # ════════════════════════════════════════════
    # ROW 4: PARALLEL TRAINING (y ≈ 5.5)
    # ════════════════════════════════════════════
    phase_label(ax, 0, 5.5, 'Phase 2')

    # FTCA Branch
    box(ax, 4.5, 5.5, 3.8, 1.2,
        'FTCA Training\n\n'
        'R3D-18 (RGB) + FreqEncoder (DFT)\n'
        'Cross-Attention fusion (8 heads)\n'
        'AdamW, lr = 1e-4, wd = 1e-3\n'
        '30 epochs, AMP, BCE loss',
        PAL['train'], fontsize=7.5)

    # Xception Branch
    box(ax, 11.5, 5.5, 3.8, 1.2,
        'Xception Baseline\n\n'
        'timm legacy_xception\n'
        'Random temporal frame sampling\n'
        'Adam, lr = 1e-4\n'
        '20 epochs, AMP, BCE loss',
        PAL['train'], fontsize=7.5)

    # ════════════════════════════════════════════
    # ROW 5: PHASE 2 OUTPUTS (y ≈ 4.0)
    # ════════════════════════════════════════════
    arrow(ax, 4.5, 4.9, 4.5, 4.3)
    arrow(ax, 11.5, 4.9, 11.5, 4.3)

    box(ax, 4.5, 3.95, 3.2, 0.55,
        'best_ftca_pad_model.pth\nVal Acc = 64.67%', PAL['output'], fontsize=8, fontweight='bold')
    box(ax, 11.5, 3.95, 3.2, 0.55,
        'best_xception_baseline.pth\nVal Acc = 46.92%', PAL['output'], fontsize=8, fontweight='bold')

    # Xception gets an X mark (worse)
    ax.text(13.5, 3.95, '✗', fontsize=14, color='#C0392B', ha='center', va='center',
            fontweight='bold', zorder=5)

    # ════════════════════════════════════════════
    # ROW 6: FINE-TUNING (y ≈ 2.7)
    # ════════════════════════════════════════════
    phase_label(ax, 0, 2.7, 'Phase 3')

    # Arrow from FTCA checkpoint to fine-tuning
    arrow(ax, 4.5, 3.68, 4.5, 3.1, label='load weights')

    box(ax, 4.5, 2.5, 4.5, 0.95,
        'Domain Adaptation Fine-Tuning\n\n'
        'Freeze: R3D backbone (rgb_* layers)\n'
        'Train:  FreqEncoder + CrossAttn + Classifier\n'
        'lr = 1e-5, 15 epochs, +250 custom anchors',
        PAL['finetune'], fontsize=7.5)

    # Fine-tune output
    arrow(ax, 6.75, 2.5, 9.0, 2.5)

    box(ax, 11.0, 2.5, 3.2, 0.6,
        'patent_ftca_v2.pth\nVal Acc = 80.85%,  Loss = 0.4174',
        PAL['output'], fontsize=8, fontweight='bold', lw=1.8)

    # Red border highlight on final model
    highlight = FancyBboxPatch(
        (11.0 - 1.6 - 0.08, 2.5 - 0.3 - 0.08), 3.36, 0.76,
        boxstyle='round,pad=0.12', facecolor='none',
        edgecolor=PAL['accent'], linewidth=2.0, linestyle='--', zorder=3
    )
    ax.add_patch(highlight)

    # ════════════════════════════════════════════
    # ROW 7: EVALUATION (y ≈ 1.0)
    # ════════════════════════════════════════════
    phase_label(ax, 0, 1.0, 'Phase 4')

    arrow(ax, 11.0, 2.2, 8.5, 1.35)
    arrow(ax, 11.0, 2.2, 13.5, 1.35)

    box(ax, 8.0, 1.0, 3.5, 0.6,
        'eval_ftca.py\nHeld-out validation set evaluation', PAL['train'], fontsize=8)
    box(ax, 13.5, 1.0, 3.5, 0.6,
        'run_experiment6.py\nWaterfall ablation  (21 test videos)', PAL['train'], fontsize=8)

    # Final metrics
    arrow(ax, 8.0, 0.7, 8.0, 0.25)
    arrow(ax, 13.5, 0.7, 13.5, 0.25)

    ax.text(8.0, 0.05, 'Acc = 80.85%', ha='center', fontsize=9,
            fontweight='bold', color=PAL['border'], family='serif')
    ax.text(13.5, 0.05, 'Overall = 95.2%  (20/21)', ha='center', fontsize=9,
            fontweight='bold', color=PAL['border'], family='serif')

    # ════════════════════════════════════════════
    # INFRASTRUCTURE NOTE (bottom)
    # ════════════════════════════════════════════
    ax.text(8.25, -0.35, 'Hardware: NVIDIA RTX A6000 (48 GB)  ·  PyTorch 2.x  ·  Mixed Precision (AMP)',
            ha='center', va='center', fontsize=8, color='#95A5A6', family='serif')

    # ════════════════════════════════════════════
    # SAVE
    # ════════════════════════════════════════════
    plt.tight_layout(pad=0.5)
    out = f'{OUT_DIR}/fig10_training_pipeline.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    generate()
