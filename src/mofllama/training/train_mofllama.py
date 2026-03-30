import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for MOFLLaMA."
    )

    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base LLaMA model directory.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training JSONL file in instruction/message format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned checkpoints.",
    )

    # Data / tokenization
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.99,
        help="Fraction of data kept for training after split.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Checkpoint save interval in steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of saved checkpoints.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw_data(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def preprocess_messages(example):
    """
    Convert a message-style sample into a single training text string.

    Expected format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    """
    messages = example["messages"]
    prompt = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    return {"text": prompt.strip()}


def tokenize_function(example, tokenizer, max_length: int):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    args = parse_args()
    set_seed(args.seed)

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading training data from: {data_path}")
    raw_data = load_raw_data(data_path)

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(preprocess_messages)

    # Keep the original workflow: split and use only the train portion
    split_dataset = dataset.train_test_split(
        test_size=1.0 - args.train_split_ratio,
        seed=args.seed,
    )
    train_dataset = split_dataset["train"]

    tokenized_dataset = train_dataset.map(
        lambda example: tokenize_function(example, tokenizer, args.max_length),
        remove_columns=train_dataset.column_names,
    )

    print(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    print(f"MOFLLaMA fine-tuning completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()