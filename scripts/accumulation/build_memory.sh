#!/bin/bash
# Experience Accumulation: Build hierarchical memory from training data.
#
# Pipeline:
#   Step 1: Generate memory visualizations (visual.py --mode memory)
#   Step 2: Run vision-to-text on memory images (EPF_image / ETTh_image)
#   Step 3: Generate test visualizations (visual.py --mode test)
#   Step 4: Distill and summarize memory (EPF_summery_cat / ETTh_summery_cat)
#
# Usage:
#   bash scripts/accumulation/build_memory.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# ==========================================
# Step 1: Generate Memory Visualizations
# ==========================================
echo "=========================================="
echo " Step 1: Generating Memory Visualizations"
echo "=========================================="

echo "[EPF] Running visual.py --mode memory..."
python main/EPF/visual.py --mode memory

echo "[ETTh] Running visual.py --mode memory..."
python main/ETTh/visual.py --mode memory

# ==========================================
# Step 2: Vision-to-Text (reads memory images)
# ==========================================
echo "=========================================="
echo " Step 2: Vision-to-Text"
echo "=========================================="

echo "[EPF] Running EPF_image.py --mode memory..."
python entry/EPF/EPF_image.py --mode memory

echo "[ETTh] Running ETTh_image.py --mode memory..."
python entry/ETTh/ETTh_image.py --mode memory

# ==========================================
# Step 3: Generate Test Visualizations
# ==========================================
echo "=========================================="
echo " Step 3: Generating Test Visualizations"
echo "=========================================="

echo "[EPF] Running visual.py --mode test..."
python main/EPF/visual.py --mode test

echo "[ETTh] Running visual.py --mode test..."
python main/ETTh/visual.py --mode test

# ==========================================
# Step 3.5: Vision-to-Text on Test Images
# ==========================================
echo "=========================================="
echo " Step 3.5: Vision-to-Text (test images)"
echo "=========================================="

echo "[EPF] Running EPF_image.py --mode test..."
python entry/EPF/EPF_image.py --mode test

echo "[ETTh] Running ETTh_image.py --mode test..."
python entry/ETTh/ETTh_image.py --mode test

# ==========================================
# Step 3.7: Generate Origin Forecasts (one-shot reasoning)
# ==========================================
echo "=========================================="
echo " Step 3.7: Generating Origin Forecasts"
echo "=========================================="

echo "[EPF] Running EPF_168_24.py..."
python entry/EPF/EPF_168_24.py

echo "[ETTh] Running ETTh_96_96.py..."
python entry/ETTh/ETTh_96_96.py

# ==========================================
# Step 4: Memory Summarization
# ==========================================
echo "=========================================="
echo " Step 4: Memory Summarization"
echo "=========================================="

echo "[EPF] Running EPF_summary.py..."
python entry/EPF/EPF_summary.py

echo "[ETTh] Running ETTh_summary.py..."
python entry/ETTh/ETTh_summary.py

echo "=========================================="
echo " Memory construction complete."
echo "=========================================="
