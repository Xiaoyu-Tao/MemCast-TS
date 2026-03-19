#!/bin/bash
# Evaluate EPF (NP dataset, look_back=168, pred_window=24).
# Selects the best prediction trajectory via LLM + heuristic scoring (uqmem).
# Usage:
#   bash scripts/evaluate/EPF.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=========================================="
echo " Evaluate: EPF / NP (168->24)"
echo "=========================================="
python evaluate/EPF/evaluate_EPF_uqmem.py

echo "=========================================="
echo " Done."
echo "=========================================="
