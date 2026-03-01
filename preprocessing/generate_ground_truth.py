import os
import csv
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Generate SCB engagement ground truth CSV from YOLO labels.")
    
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of SCB dataset (contains images/ and labels/)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train/val/test)"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        default="scb_ground_truth.csv",
        help="Name of output CSV file (saved inside labels/)"
    )
    
    args = parser.parse_args()

    # Construct paths
    labels_dir = os.path.join(args.root, "labels", args.split)
    output_dir = os.path.join(args.root, "labels")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, args.filename)

    # Mapping SCB class IDs to DAiSEE 0-3 scale
    class_to_level = {
        '3': 0, '4': 0,  # Phone, Bowing
        '5': 1,          # Leaning
        '1': 2, '2': 2,  # Reading, Writing
        '0': 3           # Raising hand
    }

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'ground_truth'])
        
        for filename in os.listdir(labels_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(labels_dir, filename)
                levels = []
                
                with open(filepath, 'r') as f:
                    for line in f:
                        class_id = line.split()[0]
                        if class_id in class_to_level:
                            levels.append(class_to_level[class_id])
                
                if levels:
                    majority_level = Counter(levels).most_common(1)[0][0]
                else:
                    majority_level = "Unknown"
                    
                image_id = filename.replace('.txt', '.png')
                writer.writerow([image_id, majority_level])

    print(f"Ground truth CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()