#!/bin/bash
# Short-term Forecasting: NP dataset (look_back=168, pred_window=24).
# Runs few-shot reasoning with dynamic features and memory retrieval.
# Usage:
#   bash scripts/short_term/NP.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=========================================="
echo " Short-term Forecasting: NP (168->24)"
echo "=========================================="
python entry/EPF/EPF_168_24_image.py

echo "=========================================="
echo " Done."
echo "=========================================="
