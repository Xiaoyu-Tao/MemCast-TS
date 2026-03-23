#!/bin/bash
# Long-term Forecasting: ETTh1 dataset (look_back=96, pred_window=96).
# Runs few-shot reasoning with dynamic features and memory retrieval.
# Usage:
#   bash scripts/long_term/ETTh.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=========================================="
echo " Long-term Forecasting: ETTh1 (96->96)"
echo "=========================================="
python entry/ETTh/ETTh_96_96_image.py

echo "=========================================="
echo " Done."
echo "=========================================="
