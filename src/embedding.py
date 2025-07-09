# src/embedding.py

import os
import sys
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import argparse
from typing import Set, List

# === CLI & notebook compatibility ===
def get_batch_size(default: int = 3000) -> int:
    if hasattr(sys, 'argv'):
        argv = [arg for arg in sys.argv[1:] if not arg.startswith('-f')]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=default)
    args, _ = parser.parse_known_args(argv)
    return args.batch_size

# === Config ===
CHUNKED_PATH = "data/processed/chunked/chunked_narratives.csv"
VECTOR_DB_DIR = "vector_store/chroma/"
CHECKPOINT_PATH = "vector_store/embedded_ids.txt"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = get_batch_size()

def load_embedded_ids(path: str) -> Set[str]:
    """Load previously embedded chunk_ids."""
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_embedded_ids(path: str, chunk_ids: List[str]):
    """Append newly embedded chunk_ids to checkpoint."""
    with open(path, "a") as f:
        for cid in chunk_ids:
            f.write(f"{cid}\n")

def embed_and_index_streaming(
    chunked_path: str,
    persist_path: str,
    model_name: str,
    checkpoint_path: str
):
    os.makedirs(persist_path, exist_ok=True)

    print("Loading previously embedded chunk_ids...", flush=True)
    embedded_ids = load_embedded_ids(checkpoint_path)

    print("Initializing model and vector store...", flush=True)
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = Chroma(
        collection_name="complaints",
        embedding_function=embedder,
        persist_directory=persist_path
    )

    for i, chunk_df in enumerate(pd.read_csv(chunked_path, chunksize=BATCH_SIZE)):
        print(f"\nProcessing batch {i + 1} (batch size = {BATCH_SIZE})...", flush=True)

        # Filter out already embedded chunks
        chunk_df["chunk_id"] = chunk_df["chunk_id"].astype(str)
        chunk_df = chunk_df[~chunk_df["chunk_id"].isin(embedded_ids)]

        if chunk_df.empty:
            print("All chunks in this batch are already embedded. Skipping.", flush=True)
            continue

        documents = [
            Document(
                page_content=row["chunk_text"],
                metadata={
                    "product": row["product"],
                    "complaint_id": row["complaint_id"],
                    "chunk_id": row["chunk_id"]
                }
            )
            for _, row in chunk_df.iterrows()
        ]

        print(f"Embedding {len(documents)} new chunks...", flush=True)
        vector_db.add_documents(documents)
        vector_db.persist()

        # Save newly processed chunk IDs
        new_ids = list(chunk_df["chunk_id"].astype(str))
        save_embedded_ids(checkpoint_path, new_ids)

        print(f"Batch {i + 1} embedded and persisted.", flush=True)

    print("\n All batches completed.", flush=True)
    print(f"Vector store saved to: {persist_path}", flush=True)
    print(f"Checkpoint file updated: {checkpoint_path}", flush=True)

def main():
    print("=" * 60, flush=True)
    print(f"Starting batch embedding process...", flush=True)
    print(f"Using model: {EMBED_MODEL_NAME}", flush=True)
    print(f"Chunked data path: {CHUNKED_PATH}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print("=" * 60, flush=True)
    embed_and_index_streaming(
        chunked_path=CHUNKED_PATH,
        persist_path=VECTOR_DB_DIR,
        model_name=EMBED_MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH
    )

if __name__ == "__main__":
    main()
