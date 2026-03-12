#!/bin/bash
# Qwen2.5-VL-7B-Instruct inference across all 3 prompts
# Temperature set to 0.0 for deterministic responses

ROOT=/home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined
OUT_DIR=/home/ubuntu/SCB-05-Dataset/runs/output_files
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

for PROMPT_NUM in 1 2 3; do
  echo "=== Running Qwen2.5-VL prompt${PROMPT_NUM} ==="
  python inference/qwen25vl_inference.py \
    --root "$ROOT" \
    --split val \
    --gt_csv scb_ground_truth_val.csv \
    --prompt_file prompts/prompt${PROMPT_NUM}.txt \
    --output_csv "${OUT_DIR}/qwen25vl_prompt${PROMPT_NUM}_run1.csv" \
    --hf_model "$MODEL" \
    --temperature 0.0
done
