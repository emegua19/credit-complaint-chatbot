# Intelligent Complaint Analysis Chatbot (RAG-Powered)

This project builds a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial to help internal teams (Product, Compliance, Support) analyze customer complaints quickly and accurately.

##  Project Goal
Enable users to ask questions like:
> "Why are customers unhappy with BNPL?"

...and receive fast, grounded answers from real complaint data.

##  Project Structure
```

credit-complaint-chatbot/
├── data/               # Raw and cleaned data
├── notebooks/          # Jupyter notebooks for EDA and testing
├── src/                # Data processing, embedding, retrieval code
├── app/                # Streamlit or Gradio chatbot app
├── vector\_store/       # Saved embeddings/index
├── reports/            # Interim and final reports
├── tests/              # Unit tests
├── config.yaml         # Config settings
├── README.md           # Project overview
├── requirements.txt    # Python dependencies

```

## ⚙️ Tools Used
- Python
- LangChain
- FAISS / ChromaDB
- SentenceTransformers
- Streamlit / Gradio
- Git & GitHub
