# Defensive KYC: Presentation Attack Detection (PAD) via Biologically-Anchored Telemetry and Spatio-Temporal Forensics

> **NIT Patna — Department of Electronics and Communication Engineering**
> Manmohan Vishwakarma · Ankur Raj

---

## Abstract

Remote Know Your Customer (KYC) verification is the standard onboarding layer for digital identity services. Yet, traditional video verification is fundamentally broken against modern threats: real-time deepfake pipelines, virtual-camera injection (OBS, ManyCam), and screen-replay attacks can impersonate a legitimate user without triggering standard liveness checks.

This project builds an **end-to-end Presentation Attack Detection (PAD) system** that operates on live video streams. The core insight is that synthetic generators optimize against human visual perception—they have no incentive to synthesize a coherent cardiovascular pulse, a consistent sensor-noise fingerprint, or plausible GAN-free frequency spectra. We exploit these blind spots simultaneously through a modular, 5-stage security gateway.

---

## The Threat Model (PAD)

| Attack Type | Vector | Why Naive Detectors Fail | Our System's Defense |
|---|---|---|---|
| **Virtual Camera Injection** | OBS injects directly into OS driver | No screen artifacts; bypasses replay detectors. | **Stream Telemetry & PRNU Forensics** |
| **Screen Replay** | Prerecorded video played to a physical camera | Moiré/flicker-only detectors miss digital injection. | **Moiré Frequency Analysis** |
| **Live Filter / Identity Swap** | Real human using real-time AR/Face-swap | Pulse exists (bypassing basic liveness). | **PRNU & FTCA Temporal Inconsistency** |
| **Fully Synthetic AI Video** | Diffusion/GAN talking-head generation | Single-frame detectors miss temporal glitches. | **Biological rPPG Gate & FTCA** |

---

## Novel Contributions

### 1. rPPG Physiological Liveness Gate *(Patent Anchor)*
Extracts the sub-visible color fluctuations in facial skin caused by the cardiac cycle using the CHROM algorithm and Welch's Power Spectral Density (PSD).
* **Gate logic:** If the Signal-to-Noise Ratio (SNR) of the pulse is below the physiological threshold, the session is **hard-rejected**.

### 2. Frequency-Temporal Cross-Attention (FTCA) Block *(Publication Anchor)*
A novel PyTorch architecture fusing a 3D-CNN (RGB Branch) with a differentiable per-frame 2D DFT (Frequency Branch) via a cross-attention module.
* **Key property:** Forces the model to learn *which* frequency artifacts correlate with *which* spatial-temporal movements, catching high-end GANs/Diffusion models.

### 3. PRNU Sensor-Noise Stream Forensics *(Deployment Anchor)*
Extracts the Photo-Response Non-Uniformity (PRNU) fingerprint. Virtual-camera software renders frames entirely in software, meaning they carry **no physical sensor signature**. Our system flags feeds lacking this physical CMOS noise.

---

## Unified System Architecture & Pipeline

```text
[Video Stream] -> [Stage 0: Stream Telemetry] (Drop OBS metadata anomalies)
                       |
                 [Stage 1: PRNU Forensics] (Verify physical camera sensor)
                       |
                 [Stage 2: Replay Analysis] (Detect Moiré screen grids)
                       |
                 [Stage 3: Biological Liveness] (CHROM rPPG Heartbeat Extraction)
                       |
                 [Stage 4: FTCA AI Detection] (3D-CNN Spatio-Temporal Analysis)
                       |
            [Weighted Fraud Risk Score & Final Decision]
```

## Technology Stack & Architecture

- **Core Engine:** Python 3.11, PyTorch (AMP Optimized for RTX A6000)
- **Signal Processing:** OpenCV, SciPy, NumPy FFT
- **Face Tracking:** MediaPipe Face Mesh (CPU optimized)
- **Data Augmentation:** Albumentations (H.264/JPEG degradation, color jitter)
- **Inference Gateway:** FastAPI + Uvicorn (Modular Headless API)

## Repository Structure

```plaintext
defensive-kyc/
├── modules/
│   ├── prnu_forensics.py      # Extracts hardware sensor noise
│   ├── moire_detector.py      # Detects screen replay grids via FFT
│   ├── rppg_extractor.py      # CHROM Biological Liveness
│   └── ftca_module.py         # 3D CNN + Frequency Cross-Attention
│
├── core_engine.py             # Traffic cop routing frames to modules
├── main.py                    # FastAPI End-to-End Security Gateway
├── dataset.py                 # PyTorch DataLoader & Augmentation
├── train.py                   # RTX A6000 FTCA Training Loop
└── requirements.txt
```