import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="Path to the inference CSV")
    parser.add_argument("--out_folder", required=True, help="Directory to save the metrics CSV")
    args = parser.parse_args()

    # Read predictions
    df = pd.read_csv(args.pred_csv)

    y_true = df["ground_truth"].astype(int)
    y_pred = df["predicted"].astype(int)

    # Calculate metrics matching the DAiSEE paper
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    kappa_qw = cohen_kappa_score(y_true, y_pred, weights="quadratic") # Fixed to Quadratic Weighted
    mse = mean_squared_error(y_true, y_pred) 

    # Print to console
    print(f"Results for: {os.path.basename(args.pred_csv)}")
    print("Accuracy:       ", round(acc, 4))
    print("Macro F1:       ", round(f1_macro, 4))
    print("Cohen Kappa (QW):", round(kappa_qw, 4))
    print("MSE:            ", round(mse, 4))

    # Prepare data structure for saving
    metrics_data = {
        "model_run": [os.path.basename(args.pred_csv)],
        "accuracy": [round(acc, 4)],
        "f1_macro": [round(f1_macro, 4)],
        "kappa_qw": [round(kappa_qw, 4)],
        "mse": [round(mse, 4)]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Ensure the output folder exists
    os.makedirs(args.out_folder, exist_ok=True)

    # Construct the save path (e.g., metrics_clip_prompt1_run1.csv)
    save_filename = f"metrics_{os.path.basename(args.pred_csv)}"
    save_path = os.path.join(args.out_folder, save_filename)

    # Save to CSV
    metrics_df.to_csv(save_path, index=False)
    print(f"\nSaved metrics to: {save_path}")

if __name__ == "__main__":
    main()