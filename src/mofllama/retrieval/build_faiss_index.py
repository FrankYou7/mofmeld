import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FAISS retrieval index from MOF KG triples."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/mofllama/kg/mofllama_kg_triples.jsonl",
        help="Path to the KG triples JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_demo/mofllama/retrieval_store/embedding",
        help="Directory to save the FAISS index.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="checkpoints/instructor_xl",
        help="Embedding model local path or Hugging Face model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for embedding, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of triples processed per batch when adding documents.",
    )
    return parser.parse_args()


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_triples(jsonl_path: Path, logger) -> List[Document]:
    """
    Load KG triples from a JSONL file and convert them to LangChain Documents.

    Expected JSONL fields per line:
    - head
    - relation
    - tail
    - source_file (optional)
    """
    documents = []
    total_size = jsonl_path.stat().st_size
    processed_size = 0
    start_time = time.time()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading triples", unit="triple"):
            processed_size += len(line.encode("utf-8"))
            item = json.loads(line)

            head = item["head"]
            relation = item["relation"]
            tail = item["tail"]
            source = item.get("source_file", "unknown")

            triple_text = f"{head} {relation} {tail}"

            doc = Document(
                page_content=triple_text,
                metadata={
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "source": source,
                },
            )
            documents.append(doc)

            if len(documents) % 1000 == 0:
                elapsed_time = time.time() - start_time
                speed = len(documents) / max(elapsed_time, 1e-8)
                progress = processed_size / max(total_size, 1) * 100
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Speed: {speed:.1f} triples/s | "
                    f"Processed: {len(documents)} triples"
                )

                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory used: {gpu_memory_used:.2f} GB")

    logger.info(f"Loaded {len(documents)} triples in total.")
    return documents


def create_batch_iterator(documents: List[Document], batch_size: int, logger):
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        yield batch

        current = min(i + batch_size, total)
        if current % 1000 == 0 or current == total:
            logger.info(f"Embedded {current}/{total} triples")
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {gpu_memory_used:.2f} GB")


def main():
    args = parse_args()
    logger = setup_logger()

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    if torch.cuda.is_available() and "cuda" in args.device:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        logger.warning("CUDA is not available or CPU mode is selected. Using CPU.")

    logger.info("Initializing embedding model...")
    embedder = HuggingFaceInstructEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
    )

    logger.info(f"Loading triples from: {input_jsonl}")
    documents = load_triples(input_jsonl, logger)

    if not documents:
        logger.warning("No triples were loaded. Exiting.")
        return

    logger.info("Building FAISS vector store...")
    start_time = time.time()

    vectorstore = None
    for batch in create_batch_iterator(documents, args.batch_size, logger):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedder)
        else:
            vectorstore.add_documents(batch)

        if torch.cuda.is_available() and "cuda" in args.device:
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    logger.info(f"Vector indexing completed in {elapsed_time:.2f} seconds")

    vectorstore.save_local(str(output_dir))
    logger.info(f"FAISS index saved to: {output_dir}")
    logger.info("Generated files typically include index.faiss and index.pkl")


if __name__ == "__main__":
    main()