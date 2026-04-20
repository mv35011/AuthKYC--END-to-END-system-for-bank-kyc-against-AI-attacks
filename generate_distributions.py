import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_distributions():
    sns.set_theme(style="whitegrid")

    # --- S1 PRNU DATA TO DATAFRAME ---
    # Grouping into a dataframe makes seaborn categorical plots much cleaner
    data = [
        {"Score": 106.38, "Category": "Physical Webcams (Pass)"},
        {"Score": 35.18, "Category": "Physical Webcams (Pass)"},
        {"Score": 4.94, "Category": "Physical Webcams (Pass)"},
        {"Score": 29.30, "Category": "Physical Webcams (Pass)"},
        {"Score": 131.16, "Category": "Physical Webcams (Pass)"},
        {"Score": 0.65, "Category": "Physical Webcams (Pass)"},

        {"Score": 0.31, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.25, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.19, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.40, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.11, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.42, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.19, "Category": "Stabilized Phones (Override)"},
        {"Score": 0.11, "Category": "Stabilized Phones (Override)"},

        {"Score": 0.92, "Category": "Virtual Cams (Attacks)"},
        {"Score": 0.42, "Category": "Virtual Cams (Attacks)"}
    ]
    df = pd.DataFrame(data)

    # ---------------------------------------------------
    # PLOT 1: PRNU Energy Strip Plot (Log Scale)
    # ---------------------------------------------------
    plt.figure(figsize=(10, 4))

    # Strip plot puts every video as a discrete dot on the line
    sns.stripplot(
        data=df, x="Score", y="Category", hue="Category",
        jitter=0.1, size=12, alpha=0.8,
        palette=["green", "orange", "red"], legend=False
    )

    # Draw the strict S1 cutoff line
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='S1 Cutoff Threshold (0.5)')

    # A log scale naturally handles data spanning from 0.1 to 100+
    plt.xscale('log')

    plt.title('S1 PRNU Energy Forensics', fontsize=16, weight='bold', pad=15)
    plt.xlabel('PRNU Energy Score (Logarithmic Scale)', fontsize=12, weight='bold')
    plt.ylabel('')
    plt.legend(loc='lower right', framealpha=1.0)
    plt.tight_layout()
    plt.savefig('report_prnu_distribution.png', dpi=300)
    plt.close()

    # ---------------------------------------------------
    # PLOT 2: Moire Grid Distribution (Kept as KDE)
    # ---------------------------------------------------
    # S2: Moire Score Data
    moire_real = [6215, 2801, 2041, 6864, 12135, 6435, 6668, 7900, 7581, 6407, 5644, 7192, 4361, 4112]
    moire_replays = [920, 1025]

    plt.figure(figsize=(10, 6))

    sns.kdeplot(moire_real, fill=True, color="#4A90E2", label="Direct Camera Recordings", linewidth=2)
    sns.scatterplot(x=moire_replays, y=[0.0001, 0.0001], color="red", s=200, marker="X",
                    label="Screen Replays (Attacks)")

    # Changed from 80000 to 1500 to show the true mathematical separation boundary
    plt.axvline(x=1500, color='red', linestyle='--', label='S2 Rejection Threshold')

    plt.title('S2 Frequency Analysis: Moiré Interference Scores', fontsize=16, weight='bold', pad=15)
    plt.xlabel('High-Frequency Grid Score', fontsize=12, weight='bold')
    plt.ylabel('Density', fontsize=12, weight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('report_moire_distribution.png', dpi=300)
    plt.close()

    print("Success! Generated corrected 'report_prnu_distribution.png' and 'report_moire_distribution.png'.")


if __name__ == "__main__":
    plot_distributions()