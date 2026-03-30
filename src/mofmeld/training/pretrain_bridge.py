import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.mofmeld.data.pretrain_dataset import build_pretrain_loaders
from src.mofmeld.models.mof_bridge import MOFMultiModal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-I pretraining for the MOFMeld bridge module."
    )

    # Paths
    parser.add_argument("--llama_path", type=str, required=True,
                        help="Path to the MOFLLaMA model or checkpoint directory.")
    parser.add_argument("--embed_root", type=str, required=True,
                        help="Directory containing precomputed structure embeddings.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing prediction.jsonl, correlation.jsonl, and association.jsonl.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for saving checkpoints and training logs.")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="Optional checkpoint path for resuming training.")

    # Training hyperparameters
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-batch size.")
    parser.add_argument("--accum_steps", type=int, default=32,
                        help="Gradient accumulation steps.")
    parser.add_argument("--total_steps", type=int, default=260000,
                        help="Total number of training steps.")
    parser.add_argument("--save_every", type=int, default=50000,
                        help="Checkpoint save interval in steps.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps for the cosine scheduler.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers.")
    parser.add_argument("--prediction_max_length", type=int, default=64,
                        help="Maximum target length for the generation task.")
    parser.add_argument("--text_max_length", type=int, default=None,
                        help="Optional maximum text length for correlation/association tasks.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def get_mean_pooled_emb(model, input_ids, attention_mask):
    """
    Mean-pool the last hidden states of the frozen LLM backbone.
    """
    outputs = model.llm.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1]          # [B, L, H]
    mask = attention_mask.unsqueeze(-1)         # [B, L, 1]
    emb = (hidden * mask).sum(1) / mask.sum(1)  # [B, H]
    return emb


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = MOFMultiModal(
        bridge_ckpt=None,
        llama_path=args.llama_path,
    ).to(device)

    # Only train the bridge-side parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler()

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # Resume if requested
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_step = checkpoint["step"]

        loss_path = output_dir / f"loss_record_step{start_step}.pkl"
        if loss_path.exists():
            with open(loss_path, "rb") as f:
                loss_record = pickle.load(f)
        else:
            loss_record = {"pred": [], "corr": [], "match": []}
    else:
        print("No checkpoint provided. Starting pretraining from scratch.")
        start_step = 0
        loss_record = {"pred": [], "corr": [], "match": []}

    # Build dataloaders for the three pretraining tasks
    pred_loader, corr_loader, assoc_loader = build_pretrain_loaders(
        tokenizer=model.tokenizer,
        embed_root=args.embed_root,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prediction_max_length=args.prediction_max_length,
        text_max_length=args.text_max_length,
    )

    pred_iter = cycle(pred_loader)
    corr_iter = cycle(corr_loader)
    assoc_iter = cycle(assoc_loader)

    model.train()
    optimizer.zero_grad()

    for step in range(start_step, args.total_steps):
        with autocast():
            # Task 1: structure-conditioned generation
            batch = next(pred_iter)
            loss_pred = model.forward_pred(
                batch["vec"].to(device),
                batch["input_ids"].to(device),
                batch["labels"].to(device),
            )

            # Task 2: structure-text contrastive learning
            batch = next(corr_iter)
            txt_emb = get_mean_pooled_emb(
                model,
                batch["text_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss_corr = model.forward_corr(
                batch["vec"].to(device),
                txt_emb,
            )

            # Task 3: structure-text association / matching
            batch = next(assoc_iter)
            with torch.no_grad():
                txt_emb = get_mean_pooled_emb(
                    model,
                    batch["text_ids"].to(device),
                    batch["attention_mask"].to(device),
                )

            logits = model.forward_match(
                batch["vec"].to(device),
                txt_emb,
            )
            loss_match = F.binary_cross_entropy_with_logits(
                logits,
                batch["label"].to(device),
            )

            # Joint multi-task loss
            loss = loss_pred + loss_corr + loss_match
            loss = loss / args.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if step % 100 == 0:
            print(
                f"step {step} | "
                f"loss_pred {loss_pred.item():.4f} | "
                f"loss_corr {loss_corr.item():.4f} | "
                f"loss_match {loss_match.item():.4f}"
            )
            loss_record["pred"].append(float(loss_pred.item()))
            loss_record["corr"].append(float(loss_corr.item()))
            loss_record["match"].append(float(loss_match.item()))

        if step > 0 and step % args.save_every == 0:
            save_path = output_dir / f"mofmeld_bridge_pretrain_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Checkpoint saved to: {save_path}")

            loss_path = output_dir / f"loss_record_step{step}.pkl"
            with open(loss_path, "wb") as f:
                pickle.dump(loss_record, f)

    print("Stage-I pretraining completed.")


if __name__ == "__main__":
    main()