import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a prediction CSV by a MOF subset list and compute MAD, RMSE, and R2."
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="Path to the prediction CSV. Must contain columns: id, target, prediction.",
    )
    parser.add_argument(
        "--mof_list",
        type=str,
        required=True,
        help=(
            "Path to the MOF subset file. "
            "Supported formats: "
            "(1) .csv with a column named 'mof_name', or "
            "(2) .txt with one MOF name per line."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Path to save the filtered prediction CSV.",
    )
    parser.add_argument(
        "--filter_bad_pred",
        action="store_true",
        help="Whether to filter extreme predictions.",
    )
    parser.add_argument(
        "--pred_abs_max",
        type=float,
        default=5.0,
        help="Absolute prediction threshold used when --filter_bad_pred is enabled.",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # MAD: Mean Absolute Deviation about the mean of predictions
    mad = float(np.mean(np.abs(y_pred - np.mean(y_pred))))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    return {"MAD": mad, "RMSE": rmse, "R2": r2}


def load_mof_names(mof_list_path: Path):
    suffix = mof_list_path.suffix.lower()

    if suffix == ".csv":
        mof_df = pd.read_csv(mof_list_path)
        if "mof_name" not in mof_df.columns:
            raise ValueError(
                f"MOF list CSV must contain a column named 'mof_name', "
                f"but got {mof_df.columns.tolist()}"
            )
        mof_names = (
            mof_df["mof_name"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
            .dropna()
            .unique()
            .tolist()
        )
        return mof_names

    if suffix == ".txt":
        with open(mof_list_path, "r", encoding="utf-8") as f:
            mof_names = [
                line.strip()
                for line in f
                if line.strip()
            ]
        return list(dict.fromkeys(mof_names))

    raise ValueError(
        f"Unsupported MOF list format: {mof_list_path}. "
        "Please provide either a .csv file with column 'mof_name' "
        "or a .txt file with one MOF name per line."
    )


def main():
    args = parse_args()

    pred_csv = Path(args.pred_csv)
    mof_list_path = Path(args.mof_list)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(pred_csv)
    required_cols = {"id", "target", "prediction"}
    if not required_cols.issubset(set(pred_df.columns)):
        raise ValueError(
            f"PRED_CSV must contain columns {required_cols}, but got {pred_df.columns.tolist()}"
        )

    mof_names = load_mof_names(mof_list_path)
    mof_set = set(mof_names)

    pred_df["id"] = pred_df["id"].astype(str).str.strip()
    sub_df = pred_df[pred_df["id"].isin(mof_set)].copy()

    matched = set(sub_df["id"].unique().tolist())
    missing = sorted(list(mof_set - matched))
    print(
        f"[INFO] mof_list unique={len(mof_set)} | "
        f"matched_rows={len(sub_df)} | "
        f"matched_unique_ids={len(matched)} | "
        f"missing_ids={len(missing)}"
    )
    if len(missing) > 0:
        print("[WARN] First 20 missing ids:", missing[:20])

    sub_df["target"] = pd.to_numeric(sub_df["target"], errors="coerce")
    sub_df["prediction"] = pd.to_numeric(sub_df["prediction"], errors="coerce")

    before = len(sub_df)
    sub_df = sub_df.dropna(subset=["target", "prediction"])
    sub_df = sub_df[
        np.isfinite(sub_df["target"].to_numpy()) &
        np.isfinite(sub_df["prediction"].to_numpy())
    ]

    if args.filter_bad_pred:
        sub_df = sub_df[sub_df["prediction"].abs() <= args.pred_abs_max]

    after = len(sub_df)
    print(f"[INFO] after cleaning: {after}/{before} rows kept")

    if after == 0:
        raise RuntimeError("No rows left after filtering. Check id matching or relax filters.")

    y_true = sub_df["target"].to_numpy()
    y_pred = sub_df["prediction"].to_numpy()
    metrics = compute_metrics(y_true, y_pred)

    print("[METRICS] on filtered subset:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    sub_df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv} | rows={len(sub_df)}")


if __name__ == "__main__":
    main()