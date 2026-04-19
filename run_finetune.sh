#!/bin/bash

# Exit immediately if any command fails
set -e

echo "==================================================="
echo " STARTING DOMAIN ADAPTATION FINE-TUNING PIPELINE "
echo "==================================================="

# Add the current root directory AND the finetune directory to Python's path
export PYTHONPATH="$(pwd):$(pwd)/finetune:$PYTHONPATH"

echo ""
echo "[PHASE 1/2] Launching Data Extractor & Physical Balancer..."
python finetune/data_extractor.py
echo ">>> Phase 1 Complete. Tensors are balanced."
echo ""

echo "[PHASE 2/2] Launching PyTorch Training Loop..."
python finetune/train.py
echo ">>> Phase 2 Complete. Golden Weights Generated."
echo ""

echo "==================================================="
echo " PIPELINE FINISHED: patent_ftca_v2.pth is ready. "
echo "==================================================="