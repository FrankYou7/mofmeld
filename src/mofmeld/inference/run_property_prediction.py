import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.mofmeld.models.mof_bridge import MOFMultiModal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MOFMeld property prediction from precomputed structure embeddings."
    )

    parser.add_argument(
        "--llama_path",
        type=str,
        default="checkpoints/MOFLLaMA",
        help="Path to the MOFLLaMA checkpoint directory.",
    )
    parser.add_argument(
        "--bridge_ckpt",
        type=str,
        default="checkpoints/finetune_result.pt",
        help="Path to the fine-tuned MOFMeld bridge checkpoint.",
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        default="data/cifs/test_cif",
        help="Directory containing CIF files. CIF names are used to locate matching embedding files.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings/qmof_hmof_embeddings",
        help="Directory containing precomputed embedding .pt files.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Give me the CO2 adsorption capacity under 298.0 K and 2.467308 atm of this mof.",
        help="Question prompt used for property prediction.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/mofmeld_inference/mofmeld_property_predictions.jsonl",
        help="Path to save inference results in JSONL format.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum prompt token length.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warm-up inference before timing.",
    )

    return parser.parse_args()


def load_model(llama_path: Path, bridge_ckpt: Path, device: torch.device) -> MOFMultiModal:
    print("Loading MOFMeld model...")

    model = MOFMultiModal(
        bridge_ckpt=None,
        llama_path=str(llama_path),
    )

    checkpoint = torch.load(bridge_ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    return model


def resolve_embedding_path(cif_path: Path, embedding_dir: Path) -> Path:
    """
    Match a CIF file to its corresponding embedding file by stem name.
    Example:
        demo_mof_001.cif -> demo_mof_001.pt
    """
    return embedding_dir / f"{cif_path.stem}.pt"


@torch.no_grad()
def answer_from_embedding_file(
    model: MOFMultiModal,
    embedding_file: Path,
    question: str,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
):
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

    embedding_obj = torch.load(embedding_file, map_location="cpu")
    struct_vec = embedding_obj["embedding"].float().unsqueeze(0).to(device)

    tokenizer = model.tokenizer
    device_type = "cuda" if device.type == "cuda" else "cpu"

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    if device_type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16 if device_type == "cuda" else torch.float32,
    ):
        query_emb = model.bridge(struct_vec).to(dtype=model.llm.dtype)
        txt_emb = model.llm.model.embed_tokens(input_ids)
        full_embeds = torch.cat([query_emb, txt_emb], dim=1).to(dtype=model.llm.dtype)

        bridge_mask = torch.ones(
            (1, query_emb.shape[1]),
            dtype=attention_mask.dtype,
            device=device,
        )
        full_attention_mask = torch.cat([bridge_mask, attention_mask], dim=1)

        generated_ids = model.llm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device_type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    prompt_text_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    if gen_text.startswith(prompt_text_decoded):
        answer = gen_text[len(prompt_text_decoded):].strip()
    else:
        answer = gen_text.strip()

    return answer, elapsed


def main():
    args = parse_args()

    llama_path = Path(args.llama_path)
    bridge_ckpt = Path(args.bridge_ckpt)
    cif_dir = Path(args.cif_dir)
    embedding_dir = Path(args.embedding_dir)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    if not llama_path.exists():
        raise FileNotFoundError(
            f"MOFLLaMA checkpoint directory not found: {llama_path}\n"
            "Please download it from Zenodo and place it under checkpoints/."
        )

    if not bridge_ckpt.exists():
        raise FileNotFoundError(
            f"MOFMeld bridge checkpoint not found: {bridge_ckpt}\n"
            "Please download it from Zenodo and place it under checkpoints/."
        )

    model = load_model(llama_path, bridge_ckpt, device)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if len(cif_files) == 0:
        raise FileNotFoundError(f"No CIF files found in: {cif_dir}")

    print(f"Found {len(cif_files)} CIF files in {cif_dir}")
    print(f"Embedding directory: {embedding_dir}")

    if args.warmup:
        print("Running warm-up inference...")
        warmup_embedding = resolve_embedding_path(cif_files[0], embedding_dir)
        _ = answer_from_embedding_file(
            model=model,
            embedding_file=warmup_embedding,
            question=args.question,
            device=device,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )

    results = []
    times = []
    total_start = time.time()

    for cif_path in tqdm(cif_files, desc="Running MOFMeld inference"):
        mof_name = cif_path.stem
        embedding_file = resolve_embedding_path(cif_path, embedding_dir)

        try:
            answer, elapsed = answer_from_embedding_file(
                model=model,
                embedding_file=embedding_file,
                question=args.question,
                device=device,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            times.append(elapsed)

            results.append(
                {
                    "mof_name": mof_name,
                    "cif_file": str(cif_path),
                    "embedding_file": str(embedding_file),
                    "question": args.question,
                    "model_answer": answer,
                    "inference_time_sec": round(elapsed, 6),
                }
            )

        except Exception as e:
            results.append(
                {
                    "mof_name": mof_name,
                    "cif_file": str(cif_path),
                    "embedding_file": str(embedding_file),
                    "question": args.question,
                    "model_answer": f"ERROR: {str(e)}",
                    "inference_time_sec": None,
                }
            )

    total_elapsed = time.time() - total_start

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved inference results to: {output_jsonl}")

    valid_times = [t for t in times if t is not None]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print("\n===== Inference Timing Summary =====")
        print(f"Number of CIF files: {len(cif_files)}")
        print(f"Successful inferences: {len(valid_times)}")
        print(f"Total elapsed time (excluding warm-up): {total_elapsed:.4f} s")
        print(f"Average inference time per sample: {avg_time:.4f} s")
        print(f"Min inference time per sample: {min(valid_times):.4f} s")
        print(f"Max inference time per sample: {max(valid_times):.4f} s")
        print(f"Device used: {device}")


if __name__ == "__main__":
    main()