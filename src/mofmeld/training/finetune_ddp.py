import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.mofmeld.data.finetune_dataset import MOFMeldFinetuneDataset
from src.mofmeld.models.mof_bridge import MOFMultiModal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-II DDP fine-tuning for the MOFMeld bridge module."
    )

    # Paths
    parser.add_argument("--llama_path", type=str, required=True,
                        help="Path to the fine-tuned MOFLLaMA checkpoint or model directory.")
    parser.add_argument("--bridge_ckpt", type=str, default="",
                        help="Optional path to the pretrained bridge checkpoint.")
    parser.add_argument("--embed_root", type=str, required=True,
                        help="Directory containing precomputed structure embeddings.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="JSONL file containing stage-II fine-tuning samples.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for saving fine-tuning checkpoints and logs.")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU batch size.")
    parser.add_argument("--accum_steps", type=int, default=16,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--total_steps", type=int, default=250500,
                        help="Total number of training steps.")
    parser.add_argument("--save_every", type=int, default=50000,
                        help="Checkpoint save interval in steps.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for the scheduler.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum token length for each sample.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers.")
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


def main():
    args = parse_args()

    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Initialize model
    model = MOFMultiModal(
        bridge_ckpt=None,
        llama_path=args.llama_path,
    )

    if args.bridge_ckpt and Path(args.bridge_ckpt).exists():
        if local_rank == 0:
            print(f"Loading pretrained bridge checkpoint from: {args.bridge_ckpt}")
        checkpoint = torch.load(args.bridge_ckpt, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.bridge.load_state_dict(checkpoint, strict=True)
    else:
        if local_rank == 0:
            print("No pretrained bridge checkpoint provided. Training starts from scratch.")

    model = model.to(device)

    # Freeze the LLM backbone explicitly
    for name, param in model.named_parameters():
        if "llm" in name:
            param.requires_grad = False

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer, scaler, scheduler
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

    # Dataset and distributed dataloader
    tokenizer = model.module.tokenizer if hasattr(model, "module") else model.tokenizer

    dataset = MOFMeldFinetuneDataset(
        tokenizer=tokenizer,
        embed_root=args.embed_root,
        data_path=args.data_path,
        max_length=args.max_length,
    )

    sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    train_iter = cycle(train_loader)

    # Training loop
    model.train()
    loss_record = []
    optimizer.zero_grad()

    for step in range(args.total_steps):
        sampler.set_epoch(step)
        batch = next(train_iter)

        struct_vec = batch["vec"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast():
            loss = model(struct_vec, input_ids, attention_mask, labels)
            loss = loss / args.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if step % 100 == 0 and local_rank == 0:
            current_loss = float(loss.item())
            print(f"step {step} | loss {current_loss:.4f}")
            loss_record.append(current_loss)

        if step > 0 and step % args.save_every == 0 and local_rank == 0:
            save_path = output_dir / f"mofmeld_bridge_finetune_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                },
                save_path,
            )

            loss_path = output_dir / f"mofmeld_bridge_loss_step_{step}.pkl"
            with open(loss_path, "wb") as f:
                pickle.dump(loss_record, f)

            print(f"Checkpoint saved to: {save_path}")

    if local_rank == 0:
        print("Stage-II fine-tuning completed.")


if __name__ == "__main__":
    main()