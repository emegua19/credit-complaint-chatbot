# src/embedding.py

import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

# === Config ===
CHUNKED_PATH = "data/processed/chunked/chunked_narratives.csv"
VECTOR_DB_DIR = "vector_store/chroma/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunked_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def embed_and_index(df: pd.DataFrame, persist_path: str, model_name: str):
    os.makedirs(persist_path, exist_ok=True)
    
    # Load embedding model
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    # Convert each row into a LangChain Document with metadata
    documents = [
        Document(
            page_content=row["chunk_text"],
            metadata={
                "product": row["product"],
                "complaint_id": row["complaint_id"],
                "chunk_id": row["chunk_id"]
            }
        )
        for _, row in df.iterrows()
    ]

    # Create and persist vector store
    db = Chroma.from_documents(documents, embedding=embedder,
                                persist_directory=persist_path)
    db.persist()

    print(f" Vector store saved to {persist_path}")

def main():
    print(" Loading chunked narratives...")
    df = load_chunked_data(CHUNKED_PATH)

    print(" Embedding and indexing...")
    embed_and_index(df, VECTOR_DB_DIR, EMBED_MODEL_NAME)

if __name__ == "__main__":
    main()
