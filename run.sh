# First run with first prompt
# temp set to 0.0 for deterministic responses
python inference/gpt4o_inference.py \
  --root /home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined \
  --split val \
  --gt_csv scb_ground_truth_val.csv \
  --prompt_file prompts/prompt1.txt \
  --output_csv SCB-05-Dataset/runs/output_files/gpt4o_prompt1_run1.csv \
  --temperature 0.0
