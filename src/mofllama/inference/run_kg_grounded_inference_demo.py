import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive demo for MOFLLaMA + KG-grounded retrieval and answering."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/MOFLLaMA",
        help="Path to the fine-tuned MOFLLaMA checkpoint directory.",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="checkpoints/instructor_xl",
        help="Path to the retrieval embedding model.",
    )
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="data_demo/mofllama/retrieval_store/embedding",
        help="Path to the FAISS vector store.",
    )
    parser.add_argument(
        "--citation_csv",
        type=str,
        default="data_demo/mofllama/metadata/citations_metadata.csv",
        help="Path to the citation metadata CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the LLM, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--embed_device",
        type=str,
        default="cpu",
        help="Device for the retrieval embedding model.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved documents.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of newly generated tokens.",
    )

    return parser.parse_args()


def format_citation(row):
    authors = (
        row["authors"].split(",")[0] + " et al."
        if row["authors"] != "Unknown"
        else "Unknown Author"
    )
    year = row["publicationDate"][:4] if row["publicationDate"] != "Unknown" else "n.d."
    title = row["title"]
    journal = row["sourceTitle"]
    return f'{authors} ({year}), "{title}", published in *{journal}*'


def build_citation_map(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("Unknown")
    return dict(zip(df["File Attachments"], df.apply(format_citation, axis=1)))


def load_llm(model_path: Path, device: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map="auto" if "cuda" in device else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
        return_full_text=False,
    )

    return tokenizer, pipe


def load_vectorstore(embedding_model_path: Path, embed_device: str, vector_store_path: Path):
    embedding_model = SentenceTransformerEmbeddings(
        model_name=str(embedding_model_path),
        device=embed_device,
    )
    vectorstore = FAISS.load_local(
        str(vector_store_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


TEMPLATE = """
You are a scientific assistant with expertise in MOF (Metal–Organic Framework) materials.

Based on the provided context, please:
1. Summarize the general principles, patterns, or factors related to the question.
2. If possible, provide examples from the context or your own knowledge to illustrate each point.
3. If the context lacks sufficient information, supplement your answer with your scientific knowledge.
4. Write your answer in clear, well-organized bullet points. Avoid repetition and hallucination.

Context:
{context}

Question: {question}

Answer:
"""


def query_mofllama(user_query, vectorstore, citation_map, tokenizer, pipe, top_k: int):
    docs = vectorstore.similarity_search(user_query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])

    prompt_text = TEMPLATE.format(context=context, question=user_query)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )

    answer = pipe(chat_prompt)[0]["generated_text"].strip()

    sources = []
    for doc in docs:
        filename = doc.metadata.get("source", "Unknown")
        citation = citation_map.get(filename, f"[No citation info] {filename}")
        sources.append("- " + citation)

    references = "\n".join(sources)
    return answer, references


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    embedding_model_path = Path(args.embedding_model_path)
    vector_store_path = Path(args.vector_store_path)
    citation_csv = Path(args.citation_csv)

    if not model_path.exists():
        raise FileNotFoundError(
            f"MOFLLaMA checkpoint directory not found: {model_path}\n"
            "Please download it from Zenodo and place it under checkpoints/."
        )

    if not embedding_model_path.exists():
        raise FileNotFoundError(
            f"Embedding model path not found: {embedding_model_path}\n"
            "Please prepare the retrieval embedding model under checkpoints/."
        )

    if not vector_store_path.exists():
        raise FileNotFoundError(
            f"FAISS vector store not found: {vector_store_path}\n"
            "Please place the demo retrieval store under data_demo/mofllama/retrieval_store/."
        )

    if not citation_csv.exists():
        raise FileNotFoundError(
            f"Citation CSV not found: {citation_csv}\n"
            "Please place the demo citation metadata under data_demo/mofllama/metadata/."
        )

    logging.basicConfig(level=logging.INFO)

    print("Loading citation metadata...")
    citation_map = build_citation_map(citation_csv)

    print("Loading MOFLLaMA...")
    tokenizer, pipe = load_llm(
        model_path=model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    print("Loading retrieval store...")
    vectorstore = load_vectorstore(
        embedding_model_path=embedding_model_path,
        embed_device=args.embed_device,
        vector_store_path=vector_store_path,
    )

    print("\n===== MOFLLaMA + KG Interactive Demo =====")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question: ").strip()
        if user_query.lower() == "exit":
            print("Exiting demo.")
            break
        if not user_query:
            print("Question cannot be empty.\n")
            continue

        print("\nRetrieving and generating answer...\n")
        answer, references = query_mofllama(
            user_query=user_query,
            vectorstore=vectorstore,
            citation_map=citation_map,
            tokenizer=tokenizer,
            pipe=pipe,
            top_k=args.top_k,
        )

        print("===== Answer =====")
        print(answer)
        print("\n===== References =====")
        print(references)
        print("")


if __name__ == "__main__":
    main()