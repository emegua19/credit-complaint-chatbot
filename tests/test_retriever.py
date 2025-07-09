# tests/test_retriever.py

import os
import sys

# === Add project root to sys.path ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.retriever import ComplaintRetriever


def test_retrieval_k_results():
    # === Adjust this path based on your setup ===
    VECTOR_DB = os.path.join(project_root, "vector_store", "chroma")

    if not os.path.exists(VECTOR_DB):
        raise FileNotFoundError(f" Vector DB not found at: {VECTOR_DB}\nMake sure to run embedding.py first.")

    retriever = ComplaintRetriever(vector_store_path=VECTOR_DB, default_top_k=5)

    query = "Why are customers frustrated with virtual currency?"
    results = retriever.retrieve(query=query)

    assert isinstance(results, list), " Retrieval output is not a list"
    assert len(results) == 5, f" Expected 5 results, but got {len(results)}"
    
    print(" test_retrieval_k_results passed!")

    for i, doc in enumerate(results, 1):
        print(f"\n Result {i}")
        print(" Product:", doc.metadata.get("product", "N/A"))
        print(" Complaint ID:", doc.metadata.get("complaint_id", "N/A"))
        print(" Text Preview:\n", doc.page_content[:150], "...")


if __name__ == "__main__":
    test_retrieval_k_results()
