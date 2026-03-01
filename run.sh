# First run with first prompt
# temp set to 0.0 for deterministic responses
python inference/gpt4o_inference.py \
  --root /home/ubuntu/SCB-05-Dataset/0.355k_university_yolo_Dataset \
  --split val \
  --gt_csv scb_val_engagement_gt.csv \
  --prompt_file prompts/prompt1.txt \
  --output_csv output_files/gpt4o_prompt1_run1.csv \
  --temperature 0.0
