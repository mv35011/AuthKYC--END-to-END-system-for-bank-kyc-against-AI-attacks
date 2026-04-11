#!/bin/bash
echo "==================================================="
echo "[System] Starting Defensive KYC Pipeline"
echo "[System] Environment: RunPod A6000 (Ubuntu Linux)"
echo "==================================================="

echo -e "\n[1/3] Starting Pipeline Phase 1: Data Extraction..."
PYTHONPATH=$(pwd) python data/extractor.py

echo -e "\n[2/3] Starting Pipeline Phase 2: FTCA Model Training..."
PYTHONPATH=$(pwd) python data/train.py

echo -e "\n[3/3] Starting Pipeline Phase 3: Xception Baseline Training..."
PYTHONPATH=$(pwd) python data/train_baseline.py

echo -e "\n==================================================="
echo "[System] ALL TRAINING COMPLETED."
echo "==================================================="