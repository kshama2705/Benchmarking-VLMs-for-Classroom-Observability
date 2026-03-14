#!/bin/bash
# Qwen2.5-VL-7B-Instruct inference on DAiSEE (all 3 prompts + self-consistency)
# Run from: CVPR 2026 Workshop/  (project root)
#
# Usage:
#   source venv/bin/activate
#   bash daisee/run_qwen25vl.sh          # or wherever this script lives
#
# If frame paths in sampled_test.csv don't match your machine, set NEW_FRAMES_DIR
# in scripts/07_qwen25vl_inference.py before running.

set -e

SCRIPT="daisee/scripts/07_qwen25vl_inference.py"

echo "=== P1 run 1 (temp=0.0) ==="
python "$SCRIPT" --prompt-variant 1 --run 1

echo "=== P2 run 1 (temp=0.0) ==="
python "$SCRIPT" --prompt-variant 2 --run 1

echo "=== P3 run 1 (temp=0.0) ==="
python "$SCRIPT" --prompt-variant 3 --run 1

# Self-consistency runs (temperature=0.7) — pick the best prompt after run 1
# and uncomment the relevant block.

# echo "=== P2 run 2 (temp=0.7) ==="
# python "$SCRIPT" --prompt-variant 2 --run 2
# echo "=== P2 run 3 (temp=0.7) ==="
# python "$SCRIPT" --prompt-variant 2 --run 3

echo "=== All done. Evaluate with: ==="
echo "python scripts/02_evaluate.py --full --model qwen25vl --results-dir daisee/results/"
