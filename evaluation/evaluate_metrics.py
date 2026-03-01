import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    y_true = df["ground_truth"].astype(int)
    y_pred = df["predicted"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1_macro, 4))
    print("Cohen Kappa:", round(kappa, 4))


if __name__ == "__main__":
    main()