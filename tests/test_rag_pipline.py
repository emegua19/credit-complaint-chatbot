# test_rag_pipeline.py
# Quick test for RAG pipeline (retriever + generator)

import os
from app_gradio_interactive import (
    get_vector_db,
    get_relevant_chunks,
    generate_answer,
    format_sources
)

# === Test Queries ===
test_queries = [
    "Credit card late payment issues",
    "Unauthorized transactions in my bank account",
    "Problems with BNPL refunds",
    "Difficulties with personal loan repayment",
    "Delays in international money transfers",
]

# Optional product filter for testing
test_product = ""  # leave empty to test all

# Load vector DB
vector_db = get_vector_db()
print(f"✅ Vector DB loaded. Total documents: {len(vector_db._collection.get()['metadatas'])}\n")

# Loop through queries
for q in test_queries:
    print(f"🔎 Query: {q}")
    
    # Retrieve top 3 relevant documents
    docs = get_relevant_chunks(q, k=3, product=test_product)
    print(f"Retrieved {len(docs)} documents:")
    for i, d in enumerate(docs):
        print(f"  Doc {i+1} | ID: {d.metadata.get('complaint_id','N/A')} | Product: {d.metadata.get('product','N/A')}")
        print(f"    Preview: {d.page_content[:150]}...\n")
    
    # Generate answer
    answer, docs_for_answer = generate_answer(q, product=test_product, k=3)
    print(f"🤖 Answer:\n{answer}\n")
    
    # Print formatted sources
    sources = format_sources(docs_for_answer)
    print(f"📚 Sources:\n{sources}\n")
    print("="*80 + "\n")
