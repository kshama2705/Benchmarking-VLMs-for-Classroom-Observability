python inference/blipvqa_inference.py \
  --root /home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined \
  --split val \
  --gt_csv scb_ground_truth_val.csv \
  --prompt_file prompts/prompt3.txt \
  --output_csv SCB-05-Dataset/runs/output_files/blip_vqa_prompt3_run1.csv \
  --hf_model Salesforce/blip-vqa-base 