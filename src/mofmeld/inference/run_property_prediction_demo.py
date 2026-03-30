import argparse
import time
from pathlib import Path

import torch

from src.mofmeld.models.mof_bridge import MOFMultiModal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive MOFMeld demo for single-MOF property prediction."
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
        default="data_demo/mofmeld/sample_cifs",
        help="Directory containing demo CIF files.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data_demo/mofmeld/sample_embeddings",
        help="Directory containing demo embedding .pt files.",
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


def resolve_files(mof_name: str, cif_dir: Path, embedding_dir: Path):
    cif_path = cif_dir / f"{mof_name}.cif"
    embedding_path = embedding_dir / f"{mof_name}.pt"
    return cif_path, embedding_path


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

    print("\n===== MOFMeld Interactive Demo =====")
    print(f"Available CIF directory: {cif_dir}")
    print(f"Available embedding directory: {embedding_dir}")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        mof_name = input("Enter MOF name (e.g. hmof-9): ").strip()
        if mof_name.lower() == "exit":
            print("Exiting demo.")
            break
        if not mof_name:
            print("MOF name cannot be empty.\n")
            continue

        question = input("Enter your question: ").strip()
        if question.lower() == "exit":
            print("Exiting demo.")
            break
        if not question:
            print("Question cannot be empty.\n")
            continue

        cif_path, embedding_path = resolve_files(mof_name, cif_dir, embedding_dir)

        if not cif_path.exists():
            print(f"CIF file not found: {cif_path}\n")
            continue

        if not embedding_path.exists():
            print(f"Embedding file not found: {embedding_path}\n")
            continue

        print(f"\nUsing CIF file: {cif_path.name}")
        print(f"Using embedding file: {embedding_path.name}")
        print("Running inference...\n")

        try:
            answer, elapsed = answer_from_embedding_file(
                model=model,
                embedding_file=embedding_path,
                question=question,
                device=device,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )

            print("===== Result =====")
            print(f"MOF name: {mof_name}")
            print(f"Question: {question}")
            print(f"Model answer: {answer}")
            print(f"Inference time: {elapsed:.4f} s\n")

        except Exception as e:
            print(f"Error during inference: {e}\n")


if __name__ == "__main__":
    main()