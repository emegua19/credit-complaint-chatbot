# src/generator.py

from typing import List
from langchain.docstore.document import Document
from transformers import pipeline

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # or another HF-supported model

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
llm = pipeline("text-generation", model=MODEL_NAME, max_new_tokens=300, do_sample=True)

def generate_answer(chunks: List[Document], question: str) -> str:
    """
    Generate an answer to the user's question based on retrieved context chunks.
    """
    context = "\n\n".join([doc.page_content for doc in chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    response = llm(prompt, return_full_text=False)[0]["generated_text"]
    return response.strip()

# Example usage:
if __name__ == "__main__":
    from retriever import retrieve_relevant_chunks
    
    question = "What are common problems with credit card services?"
    chunks = retrieve_relevant_chunks(question)
    answer = generate_answer(chunks, question)
    print("\nFinal Answer:\n", answer)
