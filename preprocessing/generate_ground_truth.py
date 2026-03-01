import os
import csv
import argparse

def mean_to_bin(mean_score: float) -> int:
    # Binning thresholds (0-3)
    if mean_score < 0.75:
        return 0
    elif mean_score < 1.75:
        return 1
    elif mean_score < 2.75:
        return 2
    else:
        return 3

def main():
    parser = argparse.ArgumentParser(description="Generate SCB engagement ground truth CSV (mean + bins).")

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

    labels_dir = os.path.join(args.root, "labels", args.split)
    output_dir = os.path.join(args.root, "labels")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, args.filename)

    # SCB class → engagement mapping
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
                    mean_score = sum(levels) / len(levels)
                    ground_truth = mean_to_bin(mean_score)
                else:
                    ground_truth = "Unknown"

                image_id = filename.replace('.txt', '.png')
                writer.writerow([image_id, ground_truth])

    print(f"Binned-mean ground truth CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()