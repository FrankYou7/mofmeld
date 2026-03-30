import argparse
import glob
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chgnet.data.dataset import CIFData, collate_graphs
from chgnet.model.model import CHGNet
from chgnet.trainer.trainer import Trainer

# Avoid "unable to mmap ..." in multiprocessing tensor sharing
torch.multiprocessing.set_sharing_strategy("file_system")


TARGET_CONFIG = {
    "pld": {
        "property": "pld",
        "mode": "direct",
    },
    "lcd": {
        "property": "lcd",
        "mode": "direct",
    },
    "surface_area": {
        "property": "surface_area_m2g",
        "mode": "direct",
    },
    "void_fraction": {
        "property": "void_fraction",
        "mode": "direct",
    },
    "co2_0p01bar": {
        "property": "co2_uptake_0p01bar",
        "mode": "co2",
        "target_pressure_atm": 0.009869232667160128,
    },
    "co2_2p5bar": {
        "property": "co2_uptake_2p5bar",
        "mode": "co2",
        "target_pressure_atm": 2.467308166790032,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CHGNet baseline model for MOF property prediction."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing train/val/test CIF folders and labels.json.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="",
        help="Path to labels.json. If omitted, defaults to <base_dir>/labels.json.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for checkpoints, predictions, and metrics.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=list(TARGET_CONFIG.keys()),
        help="Target property to train.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory for DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=42,
        help="Torch random seed.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Data random seed used by CHGNet trainer.",
    )

    return parser.parse_args()


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_labels(labels_path: Path) -> dict:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ids_from_cif_dir(cif_dir: Path):
    cifs = glob.glob(str(cif_dir / "*.cif"))
    ids = [Path(x).stem for x in cifs]
    ids.sort()
    return ids


def extract_uptake_at_pressure_atm(sample: dict, target_p_atm: float, tol: float = 1e-3):
    """
    Extract CO2 uptake at a target pressure (atm) from sample["co2_isotherms"].
    Preference is given to the 298 K isotherm if available.
    """
    iso_list = sample.get("co2_isotherms", None)
    if not iso_list:
        return None

    chosen = None
    for iso in iso_list:
        if float(iso.get("temperature", -1)) == 298.0:
            chosen = iso
            break
    if chosen is None:
        chosen = iso_list[0]

    if str(chosen.get("unit_pressure", "")).lower() != "atm":
        return None

    pts = chosen.get("points", [])
    if not pts:
        return None

    best = None
    best_diff = 1e18
    for pt in pts:
        p = float(pt.get("pressure", np.nan))
        u = pt.get("uptake", None)
        if u is None or np.isnan(p):
            continue
        diff = abs(p - target_p_atm)
        if diff < best_diff:
            best_diff = diff
            best = float(u)

    if best is None or best_diff > tol:
        return None

    return best


def subset_labels(all_labels: dict, ids: list, target_cfg: dict):
    """
    Build the split-specific labels dictionary.
    For direct targets, use the existing property field directly.
    For CO2 targets, derive a new label from co2_isotherms.
    """
    sub = {}
    missing = 0
    missing_target = 0

    mode = target_cfg["mode"]
    property_name = target_cfg["property"]

    for mid in ids:
        if mid not in all_labels:
            missing += 1
            continue

        rec = all_labels[mid]

        if mode == "direct":
            if property_name not in rec:
                missing_target += 1
                continue
            rec2 = dict(rec)
            rec2[property_name] = float(rec[property_name])
            sub[mid] = rec2

        elif mode == "co2":
            u = extract_uptake_at_pressure_atm(
                rec,
                target_cfg["target_pressure_atm"],
                tol=1e-3,
            )
            if u is None:
                missing_target += 1
                continue

            rec2 = dict(rec)
            rec2[property_name] = float(u)
            sub[mid] = rec2

        else:
            raise ValueError(f"Unknown mode: {mode}")

    if missing > 0:
        print(f"[WARN] {missing} ids not found in labels.json and were skipped.")
    if missing_target > 0:
        print(f"[WARN] {missing_target} ids missing target values and were skipped.")

    return sub


def build_loader(
    split: str,
    base_dir: Path,
    all_labels: dict,
    target_cfg: dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    cif_dir = base_dir / split / "cifs"
    split_ids = get_ids_from_cif_dir(cif_dir)
    split_labels = subset_labels(all_labels, split_ids, target_cfg)

    print(
        f"[INFO] {split}: cif_files={len(split_ids)} "
        f"labels_subset={len(split_labels)} dir={cif_dir}"
    )

    labels_filename = f"labels_{split}_{target_cfg['property']}.json"
    labels_path = cif_dir / labels_filename
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(split_labels, f)

    ds = CIFData(
        cif_path=str(cif_dir),
        labels=labels_filename,
        targets="e",
        energy_key=target_cfg["property"],
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_graphs,
        drop_last=False,
    )
    return ds, loader, list(split_labels.keys())


@torch.no_grad()
def eval_mae_rmse_r2(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for graphs, targets in loader:
        graphs = [g.to(device) for g in graphs]
        out = model(graphs)
        pred = out["e"].detach().cpu().numpy().reshape(-1)
        true = targets["e"].detach().cpu().numpy().reshape(-1)
        y_pred.append(pred)
        y_true.append(true)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, r2


@torch.no_grad()
def save_predictions_csv(model, loader, ids_all, device, out_csv_path: Path):
    model.eval()
    rows = []
    offset = 0

    for graphs, targets in loader:
        bs = targets["e"].shape[0]
        ids = ids_all[offset: offset + bs]
        offset += bs

        graphs = [g.to(device) for g in graphs]
        out = model(graphs)
        pred = out["e"].detach().cpu().numpy().reshape(-1)
        true = targets["e"].detach().cpu().numpy().reshape(-1)

        for mid, t, p in zip(ids, true, pred):
            rows.append({"id": mid, "target": float(t), "prediction": float(p)})

    pd.DataFrame(rows).to_csv(out_csv_path, index=False)
    print(f"[SAVE] {out_csv_path} (rows={len(rows)})")


def save_checkpoint(trainer: Trainer, epoch: int, best_val_mae: float, path: Path):
    ckpt = {
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "model": trainer.model.state_dict(),
    }
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        ckpt["optimizer"] = trainer.optimizer.state_dict()
    if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
        try:
            ckpt["scheduler"] = trainer.scheduler.state_dict()
        except Exception:
            pass

    torch.save(ckpt, path)
    print(f"[CKPT] saved -> {path}")


def try_resume(trainer: Trainer, ckpt_last: Path):
    if not ckpt_last.exists():
        return 0, float("inf")

    ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])

    if "optimizer" in ckpt and hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        try:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[WARN] optimizer resume failed (ignored): {e}")

    if "scheduler" in ckpt and hasattr(trainer, "scheduler") and trainer.scheduler is not None:
        try:
            trainer.scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[WARN] scheduler resume failed (ignored): {e}")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_mae = float(ckpt.get("best_val_mae", float("inf")))
    print(f"[RESUME] loaded {ckpt_last}, start_epoch={start_epoch}, best_val_mae={best_val_mae}")
    return start_epoch, best_val_mae


def main():
    args = parse_args()
    seed_everything(args.torch_seed)

    base_dir = Path(args.base_dir)
    labels_path = Path(args.labels_path) if args.labels_path else base_dir / "labels.json"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_last = save_dir / "checkpoint_last.pt"
    ckpt_best = save_dir / "checkpoint_best.pt"

    target_cfg = TARGET_CONFIG[args.target]
    property_name = target_cfg["property"]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = args.device
    all_labels = load_all_labels(labels_path)

    train_ds, train_loader, train_ids = build_loader(
        split="train",
        base_dir=base_dir,
        all_labels=all_labels,
        target_cfg=target_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_ds, val_loader, val_ids = build_loader(
        split="val",
        base_dir=base_dir,
        all_labels=all_labels,
        target_cfg=target_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_ds, test_loader, test_ids = build_loader(
        split="test",
        base_dir=base_dir,
        all_labels=all_labels,
        target_cfg=target_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print("train/val/test sizes:", len(train_ds), len(val_ds), len(test_ds))

    model = CHGNet.load().to(device)

    trainer = Trainer(
        model=model,
        targets="e",
        energy_loss_ratio=1.0,
        force_loss_ratio=0.0,
        stress_loss_ratio=0.0,
        mag_loss_ratio=0.0,
        optimizer="AdamW",
        scheduler="CosLR",
        criterion="MSE",
        epochs=1,  # keep 1; outer loop controls epochs and checkpointing
        learning_rate=args.learning_rate,
        use_device=device,
        print_freq=50,
        torch_seed=args.torch_seed,
        data_seed=args.data_seed,
    )

    # Disable chgnet==0.2.2 internal checkpoint saving
    def _no_save_checkpoint(self, epoch, mae_error, save_dir=None):
        return
    trainer.save_checkpoint = types.MethodType(_no_save_checkpoint, trainer)

    start_epoch, best_val_mae = try_resume(trainer, ckpt_last)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== Epoch {epoch}/{args.epochs - 1} ==========")

        t_epoch0 = time.time()

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            save_dir=str(save_dir),
            save_test_result=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_train_done = time.time()

        t_val0 = time.time()
        val_mae, val_rmse, val_r2 = eval_mae_rmse_r2(trainer.model.to(device), val_loader, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_val_done = time.time()

        print(f"[VAL] {property_name} MAE={val_mae:.6f} RMSE={val_rmse:.6f} R2={val_r2:.6f}")
        print(
            f"[TIME] epoch={epoch} "
            f"train+internal_val={t_train_done - t_epoch0:.1f}s "
            f"extra_val={t_val_done - t_val0:.1f}s "
            f"epoch_total={t_val_done - t_epoch0:.1f}s"
        )

        save_checkpoint(trainer, epoch, best_val_mae, ckpt_last)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(trainer, epoch, best_val_mae, ckpt_best)
            print(f"[BEST] updated best_val_mae={best_val_mae:.6f}")

    print("\n========== Final Test (load best) ==========")
    if ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.model.to(device)

    test_mae, test_rmse, test_r2 = eval_mae_rmse_r2(trainer.model, test_loader, device)
    print(f"[TEST] {property_name} MAE={test_mae:.6f} RMSE={test_rmse:.6f} R2={test_r2:.6f}")

    out_csv_train = save_dir / f"prediction_results_train_set_{property_name}.csv"
    out_csv_val = save_dir / f"prediction_results_val_set_{property_name}.csv"
    out_csv_test = save_dir / f"prediction_results_test_set_{property_name}.csv"

    save_predictions_csv(trainer.model, test_loader, test_ids, device, out_csv_test)
    save_predictions_csv(trainer.model, train_loader, train_ids, device, out_csv_train)
    save_predictions_csv(trainer.model, val_loader, val_ids, device, out_csv_val)

    metrics = {
        "target": args.target,
        "property": property_name,
        "best_val_mae": float(best_val_mae),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }

    if target_cfg["mode"] == "co2":
        metrics["target_pressure_atm"] = float(target_cfg["target_pressure_atm"])

    with open(save_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved to:", save_dir)


if __name__ == "__main__":
    main()