# First run with LLaVA1.5-7B
# temp set to 0.0 for deterministic responses

python inference/llava_ollama_inference.py \
  --root /home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined \
  --split val \
  --gt_csv scb_ground_truth_val.csv \
  --prompt_file prompts/prompt3.txt \
  --output_csv SCB-05-Dataset/runs/output_files/llava_prompt3_run1.csv \
  --model llava:7b \
  --temperature 0.0 