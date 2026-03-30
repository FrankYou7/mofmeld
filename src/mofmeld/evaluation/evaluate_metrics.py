import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction CSV using MAD, RMSE, and R2."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the prediction CSV file.",
    )
    parser.add_argument(
        "--true_col",
        type=str,
        default="",
        help="Optional name of the ground-truth column. If omitted, the script will infer it.",
    )
    parser.add_argument(
        "--pred_col",
        type=str,
        default="",
        help="Optional name of the prediction column. If omitted, the script will infer it.",
    )
    return parser.parse_args()


def infer_columns(df: pd.DataFrame):
    """
    Support two common naming conventions:
    1. standard_value / model_value
    2. target / prediction
    """
    cols = set(df.columns)

    if {"standard_value", "model_value"}.issubset(cols):
        return "standard_value", "model_value"

    if {"target", "prediction"}.issubset(cols):
        return "target", "prediction"

    raise ValueError(
        "Could not infer column names. "
        "Please provide --true_col and --pred_col explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # MAD: Mean Absolute Deviation about the mean of predictions
    mad = float(np.mean(np.abs(y_pred - np.mean(y_pred))))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # R2
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAD": mad,
        "R2": r2,
        "RMSE": rmse,
    }


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.true_col and args.pred_col:
        true_col = args.true_col
        pred_col = args.pred_col
        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Specified columns not found. Available columns: {list(df.columns)}"
            )
    else:
        true_col, pred_col = infer_columns(df)

    y_true = pd.to_numeric(df[true_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    valid_mask = np.isfinite(y_true.to_numpy()) & np.isfinite(y_pred.to_numpy())
    y_true = y_true.to_numpy()[valid_mask]
    y_pred = y_pred.to_numpy()[valid_mask]

    if len(y_true) == 0:
        raise RuntimeError("No valid numeric rows found after cleaning.")

    metrics = compute_metrics(y_true, y_pred)

    print(f"CSV: {csv_path}")
    print(f"Ground-truth column: {true_col}")
    print(f"Prediction column: {pred_col}")
    print(f"Number of valid rows: {len(y_true)}")
    print("Metrics:")
    print(f"  MAD:  {metrics['MAD']:.6f}")
    print(f"  R2:   {metrics['R2']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")


if __name__ == "__main__":
    main()