import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class MOFMeldPretrainDataset(Dataset):
    """
    Dataset for stage-I pretraining of MOFMeld.

    Supported tasks:
        - "prediction"
        - "correlation"
        - "association"

    Expected JSONL format:
        For prediction.jsonl:
            {
                "embedding_path": "...",
                "target": "..."
            }

        For correlation.jsonl and association.jsonl:
            {
                "embedding_path": "...",
                "text": "...",
                "label": "positive" / "negative" / 1 / 0
            }

    Expected embedding file format:
        A torch file containing:
            {
                "embedding": <tensor of shape [64]>
            }
    """

    def __init__(self, jsonl_path: str | Path, task: str, embed_root: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.task = task
        self.embed_root = Path(embed_root)

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        # For contrastive learning, keep only positive pairs.
        # Negative pairs are formed implicitly through in-batch negatives.
        if self.task == "correlation":
            self.data = [
                d for d in self.data
                if d.get("label") in (None, "positive", 1)
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        embedding_file = self.embed_root / Path(item["embedding_path"]).name
        embedding_obj = torch.load(embedding_file, map_location="cpu")
        struct_vec = embedding_obj["embedding"].float()  # [64]

        if self.task == "prediction":
            return {
                "vec": struct_vec,
                "target": item["target"],
            }

        return {
            "vec": struct_vec,
            "text": item["text"],
            "label": item.get("label"),
        }


def collate_prediction(batch, tokenizer, num_query: int = 32, max_length: int = 64) -> dict:
    """
    Collate function for the structure-conditioned generation task.

    Returns:
        - vec: [B, 64]
        - input_ids: [B, T]
        - labels: [B, 32+T], with the bridge-query positions masked as -100
    """
    vec = torch.stack([b["vec"] for b in batch])
    targets = [b["target"] for b in batch]

    tok = tokenizer(
        targets,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    ignore_labels = torch.full((vec.shape[0], num_query), -100, dtype=torch.long)
    labels = torch.cat([ignore_labels, tok.input_ids], dim=1)

    return {
        "vec": vec,
        "input_ids": tok.input_ids,
        "labels": labels,
    }


def collate_text(batch, tokenizer, max_length: int | None = None) -> dict:
    """
    Collate function for the correlation and association tasks.

    Returns:
        - vec: [B, 64]
        - text_ids: [B, T]
        - attention_mask: [B, T]
        - label: [B]
    """
    vec = torch.stack([b["vec"] for b in batch])
    texts = [b["text"] for b in batch]

    tok = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    def parse_label(label):
        if isinstance(label, str):
            if label.lower() == "positive":
                return 1.0
            if label.lower() == "negative":
                return 0.0
            try:
                return float(label)
            except Exception:
                return 0.0
        return float(label)

    label_tensor = torch.tensor(
        [parse_label(b["label"]) for b in batch],
        dtype=torch.float,
    )

    return {
        "vec": vec,
        "text_ids": tok.input_ids,
        "attention_mask": tok.attention_mask,
        "label": label_tensor,
    }


def build_pretrain_loaders(
    tokenizer,
    embed_root: str | Path,
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    prediction_max_length: int = 64,
    text_max_length: int | None = None,
):
    """
    Build dataloaders for the three stage-I pretraining tasks.

    Expected files under data_dir:
        - prediction.jsonl
        - correlation.jsonl
        - association.jsonl

    Returns:
        pred_loader, corr_loader, assoc_loader
    """
    data_dir = Path(data_dir)

    pred_set = MOFMeldPretrainDataset(
        jsonl_path=data_dir / "prediction.jsonl",
        task="prediction",
        embed_root=embed_root,
    )
    corr_set = MOFMeldPretrainDataset(
        jsonl_path=data_dir / "correlation.jsonl",
        task="correlation",
        embed_root=embed_root,
    )
    assoc_set = MOFMeldPretrainDataset(
        jsonl_path=data_dir / "association.jsonl",
        task="association",
        embed_root=embed_root,
    )

    pred_loader = DataLoader(
        pred_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_prediction(
            b,
            tokenizer=tokenizer,
            num_query=32,
            max_length=prediction_max_length,
        ),
        num_workers=num_workers,
    )

    corr_loader = DataLoader(
        corr_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_text(
            b,
            tokenizer=tokenizer,
            max_length=text_max_length,
        ),
        num_workers=num_workers,
    )

    assoc_loader = DataLoader(
        assoc_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_text(
            b,
            tokenizer=tokenizer,
            max_length=text_max_length,
        ),
        num_workers=num_workers,
    )

    return pred_loader, corr_loader, assoc_loader