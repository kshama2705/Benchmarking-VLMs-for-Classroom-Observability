# Evaluate over all the files in runs/output_files

for file in /home/ubuntu/Benchmarking-VLMs-for-Classroom-Observability/SCB-05-Dataset/runs/output_files/*.csv
do
  echo "Evaluating $file"
  
  python evaluation/evaluate_metrics.py \
    --pred_csv "$file" \
    --out_folder /home/ubuntu/Benchmarking-VLMs-for-Classroom-Observability/SCB-05-Dataset/runs/results
done