import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-I pretraining task files from QMOF QA-style JSONL data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input QMOF JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving prediction.jsonl, correlation.jsonl, and association.jsonl.",
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=3,
        help="Number of negative association samples generated per positive sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation_path = output_dir / "correlation.jsonl"
    prediction_path = output_dir / "prediction.jsonl"
    association_path = output_dir / "association.jsonl"

    print(f"Loading data from: {input_path}")
    all_data = load_jsonl(input_path)
    print(f"Loaded {len(all_data)} samples.")

    # Group samples by embedding_path
    mof2samples = defaultdict(list)
    for item in all_data:
        mof2samples[item["embedding_path"]].append(item)

    prediction = []
    correlation = []
    association = []

    all_by_embedding = defaultdict(list)
    for item in all_data:
        all_by_embedding[item["embedding_path"]].append(item)

    all_embedding_paths = list(all_by_embedding.keys())

    for emb_path, samples in tqdm(mof2samples.items(), desc="Building pretraining tasks"):
        # Candidate negative embedding groups: all embeddings except the current one
        negative_embedding_paths = [p for p in all_embedding_paths if p != emb_path]

        for sample in samples:
            sample_id = sample["id"]
            output_text = sample["output"]

            # Task 1: Prediction (structure -> text generation)
            prediction.append(
                {
                    "embedding_path": emb_path,
                    "target": output_text,
                    "id": sample_id,
                }
            )

            # Task 2: Correlation (positive pairs only for contrastive learning)
            correlation.append(
                {
                    "embedding_path": emb_path,
                    "text": output_text,
                    "label": "positive",
                    "id": sample_id,
                }
            )

            # Task 3: Association / matching (positive pair)
            association.append(
                {
                    "embedding_path": emb_path,
                    "text": output_text,
                    "label": 1,
                    "id": sample_id,
                }
            )

            # Task 3: Association / matching (negative pairs)
            if negative_embedding_paths:
                chosen_negative_embeddings = random.sample(
                    negative_embedding_paths,
                    k=min(args.neg_per_pos, len(negative_embedding_paths)),
                )

                for j, neg_emb_path in enumerate(chosen_negative_embeddings):
                    neg_sample = random.choice(all_by_embedding[neg_emb_path])
                    association.append(
                        {
                            "embedding_path": emb_path,
                            "text": neg_sample["output"],
                            "label": 0,
                            "id": f"{sample_id}-neg-{j}",
                        }
                    )

    write_jsonl(prediction_path, prediction)
    write_jsonl(correlation_path, correlation)
    write_jsonl(association_path, association)

    print(f"Saved prediction task data to: {prediction_path}")
    print(f"Saved correlation task data to: {correlation_path}")
    print(f"Saved association task data to: {association_path}")
    print("Stage-I pretraining task construction completed.")


if __name__ == "__main__":
    main()