import os
import sys
import shutil
import pandas as pd
import argparse
from typing import Set, List
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# === CLI compatibility ===
def get_batch_size(default: int = 3000) -> int:
    if hasattr(sys, "argv"):
        argv = [arg for arg in sys.argv[1:] if not arg.startswith("-f")]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=default)
    args, _ = parser.parse_known_args(argv)
    return args.batch_size

# === Config ===
CHUNKED_PATH = "data/processed/chunked/chunked_complaints.csv"

# Local (fast) storage
LOCAL_CHROMA_DIR = "/content/chroma_index/"
# Permanent backup on Drive
DRIVE_CHROMA_DIR = "vector_store/chroma_index/"

CHECKPOINT_PATH = "vector_store/embedded_ids.txt"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = get_batch_size()

# === Checkpoint Utils ===
def load_embedded_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_embedded_ids(path: str, chunk_ids: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for cid in chunk_ids:
            f.write(f"{cid}\n")

def verify_progress(chunked_path, checkpoint_path):
    total_chunks = sum(1 for _ in open(chunked_path)) - 1
    embedded_ids = load_embedded_ids(checkpoint_path)
    embedded_count = len(embedded_ids)
    remaining = total_chunks - embedded_count
    print("="*60)
    print(" Progress Check")
    print(f"   Total chunks      : {total_chunks:,}")
    print(f"   Already embedded  : {embedded_count:,}")
    print(f"   Remaining         : {remaining:,}")
    print("="*60)
    return embedded_ids, total_chunks, remaining

# === Safe Sync to Drive ===
def sync_to_drive():
    os.makedirs(DRIVE_CHROMA_DIR, exist_ok=True)
    shutil.copytree(LOCAL_CHROMA_DIR, DRIVE_CHROMA_DIR, dirs_exist_ok=True)
    print(" Synced local Chroma index to Google Drive.")

# === Embedding + Chroma ===
def embed_and_index_chroma(chunked_path, checkpoint_path):
    embedded_ids, total_rows, remaining = verify_progress(chunked_path, checkpoint_path)
    if remaining <= 0:
        print(" All chunks are already embedded! Nothing to do.")
        return

    print(" Initializing embedding model...", flush=True)
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Ensure local Chroma directory
    os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)

    # Load or initialize Chroma
    print(" Loading / Initializing Chroma index...")
    vector_db = Chroma(
        persist_directory=LOCAL_CHROMA_DIR,
        embedding_function=embedder
    )

    processed_rows = len(embedded_ids)

    for i, chunk_df in enumerate(pd.read_csv(chunked_path, chunksize=BATCH_SIZE)):
        print(f"\n Processing batch {i + 1} (batch size = {BATCH_SIZE})...", flush=True)

        required_cols = {"chunk_text", "product", "complaint_id", "chunk_id"}
        if not required_cols.issubset(chunk_df.columns):
            missing = required_cols - set(chunk_df.columns)
            print(f" Missing columns: {missing}. Skipping batch.", flush=True)
            continue

        # Filter out already embedded
        chunk_df["chunk_id"] = chunk_df["chunk_id"].astype(str)
        chunk_df = chunk_df[~chunk_df["chunk_id"].isin(embedded_ids)]

        if chunk_df.empty:
            print(" Skipping (all chunks already embedded)", flush=True)
            continue

        documents, ids = [], []
        for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Creating documents"):
            documents.append(
                Document(
                    page_content=row["chunk_text"],
                    metadata={
                        "product": row["product"],
                        "complaint_id": row["complaint_id"],
                        "chunk_id": row["chunk_id"]
                    },
                )
            )
            ids.append(row["chunk_id"])

        print(f" Embedding {len(documents)} new chunks...", flush=True)
        vector_db.add_documents(documents, ids=ids)
        vector_db.persist()

        save_embedded_ids(checkpoint_path, ids)
        embedded_ids.update(ids)

        processed_rows += len(documents)
        print(f" Batch {i + 1} embedded. Progress: {processed_rows}/{total_rows} rows.", flush=True)

        # Sync to Drive after each batch
        sync_to_drive()

    print("\n All batches completed and Chroma index saved!", flush=True)

# === Main ===
def main():
    print("=" * 60, flush=True)
    print(" Chroma Batch Embedding Started", flush=True)
    print(f"Model: {EMBED_MODEL_NAME}", flush=True)
    print(f"Chunked data: {CHUNKED_PATH}", flush=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print("=" * 60, flush=True)

    embed_and_index_chroma(
        chunked_path=CHUNKED_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )

if __name__ == "__main__":
    main()
