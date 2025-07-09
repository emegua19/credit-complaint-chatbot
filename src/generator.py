# src/generator.py

from typing import List
from langchain_core.documents import Document
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import sys
import os

# === Config ===
MODEL_NAME = "google/flan-t5-base"  # Public T5 model for local use
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store", "chroma")

# === Prompt Template ===
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

# === Load LLM pipeline ===
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=300)
except Exception as e:
    print(f" Error loading model: {e}")
    sys.exit(1)

# Dynamically import retriever (assuming it's in the same src directory)
sys.path.append(os.path.dirname(__file__))
from retriever import ComplaintRetriever

def generate_answer(chunks: List[Document], question: str) -> str:
    """
    Generate an answer based on retrieved context chunks and the user question.
    """
    if not chunks:
        return "I don't have enough information to answer."
    context = "\n\n".join([doc.page_content for doc in chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        response = llm(prompt, return_text=True, clean_up_tokenization_spaces=True)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        print(f" Error generating answer: {e}")
        return "Error processing your request."

def main():
    # Initialize retriever with local vector store path
    try:
        retriever = ComplaintRetriever(vector_store_path=VECTOR_STORE_PATH)
    except Exception as e:
        print(f" Error initializing retriever: {e}")
        sys.exit(1)

    # Example usage
    question = "What are common problems with credit card services?"
    chunks = retriever.retrieve(query=question)
    answer = generate_answer(chunks, question)
    print("\nFinal Answer:\n", answer)

if __name__ == "__main__":
    main()