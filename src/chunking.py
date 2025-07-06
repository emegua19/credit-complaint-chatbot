# src/chunking.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os

# === Config ===
INPUT_PATH = "data/processed/filtered/filtered_complaints.csv"
OUTPUT_PATH = "data/processed/chunked/chunked_narratives.csv"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_filtered_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def chunk_narratives(df: pd.DataFrame, chunk_size=CHUNK_SIZE, 
                     overlap=CHUNK_OVERLAP) -> pd.DataFrame:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = splitter.split_text(row["cleaned_narrative"])
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{row.name}_{i}",
                "product": row["Product"],
                "complaint_id": row.get("Complaint ID", row.name),
                "chunk_text": chunk,
                "original_narrative": row["cleaned_narrative"]
            })
    return pd.DataFrame(records)

def save_chunked_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print(" Loading filtered complaints...")
    df = load_filtered_data(INPUT_PATH)

    print(" Splitting narratives into chunks...")
    chunked_df = chunk_narratives(df)

    print(f" Saving {len(chunked_df)} chunks to {OUTPUT_PATH}")
    save_chunked_data(chunked_df, OUTPUT_PATH)
    print(" Done.")

if __name__ == "__main__":
    main()
