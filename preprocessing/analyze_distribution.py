import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to scb_engagement_val_gt.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if "ground_truth" not in df.columns:
        raise ValueError("CSV must contain 'ground_truth' column")

    # Ensure numeric
    df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")

    # Drop NaNs if any
    df = df.dropna(subset=["ground_truth"])

    total = len(df)

    distribution = df["ground_truth"].value_counts().sort_index()

    print("Total samples:", total)
    print("\nDistribution:")
    for label, count in distribution.items():
        percentage = (count / total) * 100
        print(f"Level {int(label)}: {count} samples ({percentage:.2f}%)")

    print("\nRaw dict format:")
    print(distribution.to_dict())


if __name__ == "__main__":
    main()