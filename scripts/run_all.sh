#!/bin/bash
# Master script to run all DAiSEE experiments
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$BASE_DIR/venv/bin/activate"

source "$VENV"

echo "============================================"
echo "Step 1: Sample frames from DAiSEE test set"
echo "============================================"
python "$SCRIPT_DIR/01_sample_and_extract_frames.py"

echo ""
echo "============================================"
echo "Step 2: CLIP inference (3 prompts × 1 run each first)"
echo "============================================"
for p in 1 2 3; do
    echo "--- CLIP prompt $p, run 1 ---"
    python "$SCRIPT_DIR/03_clip_inference.py" --prompt-variant $p --run 1
done

echo ""
echo "============================================"
echo "Step 3: BLIP-VQA inference (3 prompts × 1 run each)"
echo "============================================"
for p in 1 2 3; do
    echo "--- BLIP-VQA prompt $p, run 1 ---"
    python "$SCRIPT_DIR/04_blip_vqa_inference.py" --prompt-variant $p --run 1
done

echo ""
echo "============================================"
echo "Step 4: GPT-4o inference (3 prompts × 1 run each)"
echo "  Requires: export OPENAI_API_KEY=your_key"
echo "============================================"
for p in 1 2 3; do
    echo "--- GPT-4o prompt $p, run 1 ---"
    python "$SCRIPT_DIR/05_gpt4o_inference.py" --prompt-variant $p --run 1
done

echo ""
echo "============================================"
echo "Step 5: LLaVA inference (3 prompts × 1 run each)"
echo "============================================"
for p in 1 2 3; do
    echo "--- LLaVA prompt $p, run 1 ---"
    python "$SCRIPT_DIR/06_llava_inference.py" --prompt-variant $p --run 1
done

echo ""
echo "============================================"
echo "Step 6: Self-consistency runs (run 2 and 3 for best prompt)"
echo "  Run these manually after checking which prompt works best:"
echo "  python 03_clip_inference.py --prompt-variant BEST --run 2"
echo "  python 03_clip_inference.py --prompt-variant BEST --run 3"
echo "  (repeat for each model)"
echo "============================================"

echo ""
echo "============================================"
echo "Step 7: Evaluate all results"
echo "============================================"
for model in clip blip_vqa gpt4o llava; do
    echo "--- Evaluating $model ---"
    python "$SCRIPT_DIR/02_evaluate.py" --full --model $model --results-dir "$BASE_DIR/results/"
done

echo ""
echo "Done! All results in $BASE_DIR/results/"
