# Defensive KYC: Biologically-Anchored Detection of Synthetic Camera Streams and Replay Attacks

> **NIT Patna — Department of Electronics and Communication Engineering**
> Manmohan Vishwakarma · Ankur Raj

---

## Abstract

Remote Know Your Customer (KYC) verification is now the standard onboarding layer for banking, fintech, and digital identity services. Yet traditional video-based verification is fundamentally broken against modern threats: real-time deepfake pipelines, virtual-camera injection (OBS, ManyCam), and screen-replay attacks can impersonate a legitimate user without triggering any existing liveness check.

This project builds an **end-to-end, biologically-grounded fraud detection system** that operates on live video streams. The core insight is that deepfake generators optimise against human visual perception — they have no incentive to synthesise a coherent cardiovascular pulse, a consistent sensor-noise fingerprint, or plausible GAN-free frequency spectra. We exploit all three of these blind spots simultaneously.

---

## The Four-Layer Attack Threat Model

| Attack Type | Vector | Why Naive Detectors Fail |
|---|---|---|
| **Screen Replay** | Prerecorded video played to a physical camera | Moire/flicker-only detectors miss digital injection |
| **Virtual Camera Injection** | OBS / ManyCam injects directly into OS camera driver | No screen artefacts; bypasses replay detectors entirely |
| **Real-time Deepfake** | Live GPU face-swap (DeepFaceLive, Roop) | Single-frame detectors miss temporal inconsistencies |
| **Fully Synthetic AI Video** | Diffusion/GAN talking-head generation | Requires frequency-domain + biological signal analysis |

---

## Novel Contributions

This system introduces three novel technical contributions beyond the baseline replay and deepfake detection modules:

### 1. rPPG Physiological Liveness Gate *(Patent Anchor)*

Remote photoplethysmography (rPPG) recovers the sub-visible colour fluctuations in facial skin caused by the cardiac cycle. A genuine live human produces a coherent blood-volume pulse (BVP) in the 0.7–4.0 Hz band. Deepfake generators produce no such signal.

**Mechanism:**
- Extract facial ROI via MediaPipe Face Mesh across the full verification window (~10 s)
- Apply CHROM chrominance projection to recover the raw pulse signal `p(t)`
- Bandpass filter to `[0.7, 4.0] Hz` (42–240 bpm physiological range)
- Compute pulse SNR in the frequency domain via DFT

**Gate logic:** If `SNR_pulse < threshold`, the session is **hard-rejected** before any deep-learning module runs. This is a fast, CPU-only pre-filter that no current synthetic video generator can defeat without explicit physiological modelling.

**Why it is patentable:** Fusing rPPG as a *liveness gate* — rather than a heart-rate estimator — within a KYC fraud pipeline is an unexplored and defensible claim. No commercial KYC product currently deploys this mechanism.

---

### 2. Frequency-Temporal Cross-Attention (FTCA) Block *(Publication Anchor)*

GAN and diffusion model artefacts manifest as anomalous high-frequency components in the Fourier domain. The *temporal evolution* of these frequency artefacts is a stronger discriminator than any single-frame FFT.

**Architecture:**

```
Input Clip (T × H × W × 3)
       │
       ├─────────────────────────────────────────┐
       │                                         │
 [RGB Branch]                            [Frequency Branch]
 3D-CNN Backbone                     Per-frame 2D DFT → log|F|
 (ResNet3D-18)                       → Frequency CNN Encoder
       │                                         │
 F_rgb ∈ R^(T' × d)                   F_freq ∈ R^(T' × d)
       │                                         │
       └──────────── Cross-Attention ────────────┘
                Q = F_rgb · W_Q
                K = F_freq · W_K
                V = F_freq · W_V
                F_fused = softmax(QK^T / √d) · V
                          │
                    Classification Head
                    (Real / Fake score)
```

**Key property:** The cross-attention forces the model to learn *which frequency artefacts correlate with which spatial-temporal patterns* — a training signal unavailable to either branch in isolation.

**Training loss:**

```
L = L_BCE(y, ŷ) + λ₁ · L_freq(F_freq, F_freq_real) + λ₂ · L_temporal(F_fused, Δt)
```

where `L_freq` is a contrastive loss on frequency embeddings and `L_temporal` penalises inconsistent frequency trajectories over time.

---

### 3. PRNU Sensor-Noise Stream Forensics *(Deployment Anchor)*

Every physical camera sensor has a unique Photo-Response Non-Uniformity (PRNU) fingerprint arising from manufacturing imperfections. Virtual-camera software renders frames entirely in software — they carry **no PRNU signature**.

**Mechanism:**
- Estimate residual noise field by subtracting a BM3D/DnCNN-denoised version of each frame
- Correlate estimated fingerprint against a pre-enrolled reference or use absence-of-fingerprint as the anomaly signal
- Streams from OBS/ManyCam produce near-zero correlation → flagged as virtual injection

---

## Unified System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LIVE VIDEO STREAM                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Pre-processing   │
                    │  Face Det. (MTCNN) │
                    │  ROI Extraction    │
                    └──┬──┬──┬──┬───────┘
                       │  │  │  │
          ┌────────────┘  │  │  └────────────┐
          │               │  │               │
  ┌───────▼──────┐ ┌──────▼──▼──────┐ ┌─────▼────────┐
  │   MODULE 1   │ │   MODULE 2     │ │  MODULE 3    │
  │ rPPG         │ │ FTCA Deepfake  │ │ PRNU Sensor  │
  │ Liveness     │ │ Detector       │ │ Forensics    │
  │ Gate         │ │ (Novel Arch.)  │ │              │
  └───────┬──────┘ └──────┬─────────┘ └─────┬────────┘
          │               │                 │
          │         ┌─────▼──────┐          │
          │         │  MODULE 4  │          │
          │         │  Replay    │          │
          │         │  Attack    │          │
          │         │  Detector  │          │
          │         └─────┬──────┘          │
          │               │                 │
          └───────┬────────┴─────────────────┘
                  │
         ┌────────▼────────┐
         │  Weighted Score │
         │  Fusion Layer   │
         │  (learned w_i)  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Fraud Risk     │    r < 0.3  → PASS
         │  Score r∈[0,1]  │    0.3–0.7 → CHALLENGE
         └─────────────────┘    r ≥ 0.7  → REJECT + LOG
```

**Hard override:** If the rPPG gate fires with confidence > 0.95, the session is hard-rejected immediately, bypassing all downstream modules.

---

## Technology Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.x |
| Video / Image | OpenCV 4.x, ffmpeg-python |
| Face Detection | MediaPipe Face Mesh, MTCNN (`facenet-pytorch`) |
| rPPG | Custom CHROM/POS + TS-CAN (PyTorch) |
| Frequency Analysis | NumPy FFT, SciPy signal |
| 3D-CNN Backbone | torchvision ResNet3D-18 (Kinetics pretrained) |
| PRNU Denoising | BM3D / DnCNN |
| Data Augmentation | Albumentations (H.264/JPEG compression, blur, colour jitter) |
| Inference Service | FastAPI + Uvicorn (WebSocket endpoint) |
| Experiment Tracking | Weights & Biases (wandb) |
| Deployment | Docker, ONNX Runtime |

---

## Hardware Setup

### Local Prototyping (Apple Silicon / M2)
Used for data extraction scripts, FastAPI endpoint validation, and CPU/MPS debugging.

```bash
pip install torch torchvision torchaudio
pip install opencv-python facenet-pytorch albumentations fastapi uvicorn mediapipe scipy bm3d
```

### Heavy Training (RTX A6000 VM)
Used for training FTCA and TS-CAN models on large-scale datasets.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python facenet-pytorch albumentations fastapi uvicorn mediapipe scipy wandb
```

---

## Datasets

| Dataset | Use | Attack Type |
|---|---|---|
| FaceForensics++ (c23 / c40) | FTCA training + eval | Deepfake |
| Celeb-DF v2 | FTCA generalisation eval | Deepfake |
| DFDC Preview | FTCA training | Deepfake |
| Replay-Attack (IDIAP) | Replay module training | Screen replay |
| MSU-MFSD | Replay module training | Screen replay |
| UBFC-rPPG | rPPG module training | Liveness ground truth |
| COHFACE | rPPG module training | Liveness ground truth |
| Custom OBS captures | Virtual injection training | Virtual camera injection |

The custom OBS dataset is generated in-house: deepfake sequences are injected through OBS virtual camera into a capture pipeline, producing samples that no existing public dataset covers.

---

## Data Pipeline

To ensure models learn behavioural and temporal artefacts rather than compression or hardware biases:

- **Batched MTCNN extraction** — isolates facial bounding boxes across 16-frame windows
- **Temporal windowing** — `T = 16` frames, stride `= 8`, aligned across all four modules
- **Aggressive augmentation via Albumentations:**
  - On-the-fly H.264 / JPEG compression degradation
  - Extreme colour jitter and Gaussian blur injection
  - Random horizontal flip, crop jitter
  - Simulated screen-capture noise (Moire injection for replay augmentation)

---

## Evaluation Metrics

| Metric | Target |
|---|---|
| AUC-ROC | > 0.97 on FF++ c23 |
| Equal Error Rate (EER) | < 3% across all attack types |
| False Acceptance Rate (FAR) | < 1% (banking deployment requirement) |
| False Rejection Rate (FRR) | < 5% (user experience requirement) |
| End-to-end latency | < 150 ms per 16-frame window (mixed CPU+GPU) |

### Ablation Study Design

| Variant | Modules Active |
|---|---|
| Baseline | Replay detector only |
| +rPPG | Baseline + liveness gate |
| +FTCA | Baseline + frequency-temporal detector |
| +PRNU | Baseline + sensor forensics |
| **Full system** | **All modules + fusion** |

Each ablation is evaluated against all four attack types independently.

---

## Implementation Phases

| Phase | Weeks | Deliverable |
|---|---|---|
| Literature review + environment setup | 1–2 | Shared repo, dependency lock, reading list |
| Dataset collection + augmentation pipeline | 3–4 | Unified DataLoader, augmentation config |
| rPPG module (CHROM + TS-CAN + gate) | 5–6 | rPPG liveness gate with SNR threshold tuning |
| FTCA module (architecture + training) | 7–8 | Trained FTCA block with ablation baseline |
| PRNU + Replay modules | 9 | Both detectors integrated and unit-tested |
| Fusion layer + FastAPI pipeline | 10–11 | End-to-end inference service, Docker image |
| Evaluation, ablation, benchmarking | 12–14 | Final metrics, paper draft, patent claims draft |

---

## Patent and Publication Targets

**Independent Claim 1 — Biological Liveness Gate:**
A method for authenticating a live video stream by extracting an rPPG signal from a facial ROI, computing pulse SNR within the physiological frequency band, and hard-rejecting streams below a threshold — operating without any additional hardware sensor.

**Independent Claim 2 — FTCA Architecture:**
A neural network for synthetic video detection comprising a 3D-CNN RGB branch, a per-frame DFT frequency encoding branch, and a cross-attention module in which the RGB branch queries the frequency branch to capture correlations between spatial artefacts and their temporal frequency evolution.

**Target venues:** IEEE/CVF CVPR or ECCV Workshop on Media Forensics · IEEE T-IFS · ACM MM

---

## Repository Structure *(planned)*

```
kyc-defensive/
├── data/
│   ├── loaders/          # PyTorch Dataset classes (temporal windowing)
│   └── augmentation/     # Albumentations pipelines
├── modules/
│   ├── rppg/             # CHROM extractor, TS-CAN, SNR gate
│   ├── ftca/             # 3D-CNN backbone, frequency encoder, cross-attention
│   ├── prnu/             # BM3D denoiser, fingerprint correlator
│   └── replay/           # Moire/flicker detector
├── fusion/               # Weighted score fusion layer
├── api/                  # FastAPI + WebSocket inference service
├── train/                # Training scripts per module + joint fine-tuning
├── eval/                 # Benchmarking and ablation scripts
├── docker/               # Dockerfile + compose config
└── notebooks/            # Exploratory analysis and visualisation
```

---

*Internal working document — NIT Patna Minor Project · ECE Department*