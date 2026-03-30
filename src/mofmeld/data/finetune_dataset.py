import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class MOFMeldFinetuneDataset(Dataset):
    """
    Dataset for stage-II fine-tuning of MOFMeld.

    Expected JSONL format:
        Each line should be a JSON object with at least:
        - "embedding_path": path or filename of the saved structure embedding
        - "input": input question or prompt
        - "output": target answer text

    Expected embedding file format:
        A torch file containing a dictionary with key:
        - "embedding": a tensor of shape [64]
    """

    def __init__(
        self,
        tokenizer,
        embed_root: str | Path,
        data_path: str | Path,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.embed_root = Path(embed_root)
        self.data_path = Path(data_path)
        self.max_length = max_length

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        if self.tokenizer.pad_token_id is None:
            # Fallback for tokenizers without an explicit pad token
            if self.tokenizer.eos_token_id is not None:
                self.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(
                    "The tokenizer does not have a pad_token_id or eos_token_id."
                )
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Load the precomputed structure embedding
        embedding_file = self.embed_root / Path(item["embedding_path"]).name
        embedding_obj = torch.load(embedding_file, map_location="cpu")
        struct_vec = embedding_obj["embedding"].float()  # [64]

        question = item["input"]
        answer = item["output"]

        # Prompt format used for fine-tuning
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)

        # Only compute loss on the answer tokens
        labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        return {
            "vec": struct_vec,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_finetune_loader(
    tokenizer,
    embed_root: str | Path,
    data_path: str | Path,
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for stage-II fine-tuning.

    Args:
        tokenizer: Hugging Face tokenizer
        embed_root: Directory containing precomputed structure embeddings
        data_path: JSONL file containing fine-tuning samples
        batch_size: Batch size
        max_length: Maximum token length
        shuffle: Whether to shuffle samples
        num_workers: Number of dataloader workers
        drop_last: Whether to drop the last incomplete batch

    Returns:
        A PyTorch DataLoader instance.
    """
    dataset = MOFMeldFinetuneDataset(
        tokenizer=tokenizer,
        embed_root=embed_root,
        data_path=data_path,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )