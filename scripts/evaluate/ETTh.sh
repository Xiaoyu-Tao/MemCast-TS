#!/bin/bash
# Evaluate ETTh (ETTh1 dataset, look_back=96, pred_window=96).
# Selects the best prediction trajectory via LLM + heuristic scoring (uqmem).
# Usage:
#   bash scripts/evaluate/ETTh.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=========================================="
echo " Evaluate: ETTh1 (96->96)"
echo "=========================================="
python evaluate/ETTh/evaluate_ETT_uqmem.py

echo "=========================================="
echo " Done."
echo "=========================================="
