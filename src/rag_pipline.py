# rag_pipeline.py

import os
from typing import List, Optional
import pandas as pd
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# === Config ===
CHROMA_DIR = "/content/drive/MyDrive/credit-complaint-chatbot/vector_store/chroma_index/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# === Prompt Template ===
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. 
Your task is to answer questions about customer complaints.

Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""

# === Retriever ===
def get_relevant_chunks(query: str, k: int = TOP_K, product: Optional[str] = None) -> List[Document]:
    """Retrieve top-k relevant chunks as LangChain Documents from Chroma."""
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )

    # Apply product filter if specified
    filter_dict = {"product": product} if product else None

    docs = vector_db.similarity_search(query, k=k, filter=filter_dict)
    return docs   # return Document objects directly

# === Generator ===
def get_llm():
    """Load HuggingFace generation pipeline."""
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",   # can downgrade to flan-t5-base if OOM
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

def generate_answer(query: str, product: Optional[str] = None, k: int = TOP_K):
    """Generate an answer using retrieved chunks + LLM."""
    docs = get_relevant_chunks(query, k=k, product=product)

    if not docs:
        return "⚠️ No relevant context retrieved.", []

    # Build context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    # Run generator
    llm = get_llm()
    answer = llm.invoke(final_prompt)

    return answer, docs

# === Evaluation ===
def format_sources(docs: List[Document], max_chars: int = 150) -> List[str]:
    """Format retrieved sources for display (trimmed)."""
    sources = []
    for d in docs:
        text_preview = d.page_content[:max_chars].replace("\n", " ")
        cid = d.metadata.get("complaint_id", "N/A")
        prod = d.metadata.get("product", "N/A")
        sources.append(f"{text_preview}... [Product: {prod}, Complaint ID: {cid}]")
    return sources

def evaluate_pipeline(questions: List[str], product: Optional[str] = None, save_path: str = "evaluation_results.md"):
    """Run evaluation and save results to Markdown."""
    rows = []

    for q in questions:
        print(f"\n🔎 Question: {q}")
        answer, docs = generate_answer(q, product=product, k=TOP_K)

        srcs = format_sources(docs)
        print(f"🤖 Answer: {answer}")
        print("📚 Sources:")
        for s in srcs:
            print(f"   - {s}")

        rows.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": "\n".join(srcs[:2]),  # keep top 2 for clarity
            "Quality Score (1-5)": "",
            "Comments": ""
        })

    # Save evaluation table
    df = pd.DataFrame(rows)
    md_table = df.to_markdown(index=False)

    with open(save_path, "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write(md_table)

    print(f"✅ Evaluation saved to {save_path}")

# === Main ===
if __name__ == "__main__":
    import torch

    sample_queries = [
        "Credit card late payment issues",
        "Unauthorized transactions in my bank account",
        "Problems with BNPL refunds",
        "Difficulties with personal loan repayment",
        "Delays in money transfers abroad",
    ]

    evaluate_pipeline(sample_queries, save_path="evaluation_results.md")
