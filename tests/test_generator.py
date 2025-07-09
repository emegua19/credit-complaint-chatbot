# test_generator.py

import os
import sys
from typing import List
from langchain_core.documents import Document

# === Add project source path ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from retriever import ComplaintRetriever
from generator import generate_answer  # Assuming generate_answer is a function

def test_generate_answer():
    """
    Test the generate_answer function with sample data.
    """
    # Initialize retriever with local vector store path
    vector_store_path = os.path.join(project_root, "vector_store", "chroma")
    try:
        retriever = ComplaintRetriever(vector_store_path=vector_store_path)
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return

    # Test cases
    test_cases = [
        ("What are common problems with credit card services?", None),
        ("Why are users unhappy with savings accounts?", "Savings account"),
        ("Do you have data on rare complaints?", None)  # Should trigger "not enough information"
    ]

    for question, product_filter in test_cases:
        print(f"\nTesting question: {question}")
        try:
            # Retrieve chunks
            chunks = retriever.retrieve(query=question, product_filter=product_filter)
            print(f"Retrieved {len(chunks)} chunks")

            # Generate answer
            answer = generate_answer(chunks, question)
            print(f"Generated Answer: {answer}")

            # Basic validation
            assert isinstance(answer, str), "Answer must be a string"
            assert len(answer) > 0, "Answer must not be empty"
        except Exception as e:
            print(f"Error during test: {e}")

def main():
    print("Starting generator tests...")
    test_generate_answer()
    print("Generator tests completed.")

if __name__ == "__main__":
    main()