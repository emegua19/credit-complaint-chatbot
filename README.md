# Intelligent Complaint Analysis Chatbot (RAG-Powered)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot to help internal teams at **CrediTrust Financial** analyze real customer complaints using natural language queries. It is built using large language models (LLMs), vector databases, and semantic search.

---

## Problem

CrediTrust receives millions of complaints across financial products like credit cards, personal loans, and Buy Now, Pay Later (BNPL).  
Manually reviewing these complaints is slow and often incomplete, leading to missed trends and unresolved issues.

---

## Solution

This chatbot allows users to ask questions such as:

> _“Why are users unhappy with BNPL?”_

And receive fast, grounded responses based on actual complaint narratives, powered by:

- **Text chunking and embedding**
- **Semantic search using a vector database**
- **Answer generation via LLMs**

---

##  Visual Overview

```

```
                 +----------------------------------+
                 |     CrediTrust Financial         |
                 |  (Digital finance company)       |
                 +----------------------------------+
                               |
         ┌────────────────────┴────────────────────┐
         |                                         |
 +------------------+                   +------------------------+
 |     Asha         |                   |    Internal Teams      |
 |  Product Manager |                   |  (Support, Compliance) |
 +------------------+                   +------------------------+
         |                                         |
 Needs fast insight into complaints        Struggle with data overload
         |                                         |
         └──────────────┬──────────────────────────┘
                        ↓
          +-----------------------------+
          |        PROJECT GOAL         |
          |  Build a chatbot using RAG  |
          +-----------------------------+
                        ↓
    +----------------------------------------------+
    |               Chatbot Must:                  |
    |----------------------------------------------|
    | ✓ Understand natural language queries         |
    | ✓ Retrieve relevant complaint chunks          |
    | ✓ Generate concise, grounded answers          |
    | ✓ Provide traceable evidence                  |
    +----------------------------------------------+
                        ↓
    +----------------------------------------------+
    |              Success Metrics                 |
    |----------------------------------------------|
    | • Speed: Answers in seconds                   |
    | • Usability: No technical skills required     |
    | • Proactivity: Reveal hidden issues early     |
    +----------------------------------------------+
```

```

---

## 📦 Project Structure

```

credit-complaint-chatbot/
├── data/                   # Raw and processed complaint data
│   ├── raw/                # Original CFPB CSV (\~5.6 GB)
│   └── processed/          # Filtered & chunked (\~1.12 GB)
├── notebooks/              # Task 1: EDA and cleaning
├── src/                    # Modular Python source code
│   ├── chunking.py         # Text splitting into 300-char chunks
│   ├── embedding.py        # Embedding & indexing with ChromaDB
│   ├── retriever.py        # Query vector + top-k semantic retriever
│   ├── generator.py        # LLM-based answer generation
│   └── preprocessing.py    # Narrative cleaning logic
├── vector\_store/           # Saved vector database (ChromaDB)
├── app/                    # Chat interface (Streamlit/Gradio)
├── reports/                # Interim & final documentation
├── tests/                  # Unit tests for core logic
├── requirements.txt        # Python dependencies
├── README.md               # You're here!
└── .github/workflows/      # GitHub CI pipeline

````

---

##  Interim Deliverables

### Task 1: EDA & Preprocessing
- Filtered 5.6M CFPB complaints → 479,110 clean records
- Focused on 5 key product categories (BNPL, loans, etc.)
- Removed nulls, normalized text, calculated lengths
- Saved cleaned `.csv` (1.12 GB) and `.csv.gz` versions

### Task 2: Chunking & Embedding
- Used `RecursiveCharacterTextSplitter` with 300/50 window
- Embedded with `all-MiniLM-L6-v2` (fast, lightweight model)
- Stored in ChromaDB with complaint ID + product metadata

---

## ⚙️ Tech Stack

| Component            | Tool / Library                        |
|----------------------|----------------------------------------|
| Text Cleaning        | `re`, `pandas`                         |
| Chunking             | LangChain `RecursiveCharacterTextSplitter` |
| Embedding            | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store         | `ChromaDB` via `langchain-community`  |
| Chat UI (planned)    | `Streamlit` or `Gradio`                |
| Testing & CI         | `unittest`, GitHub Actions             |

---

##  Setup Instructions

```bash
# Clone the project
git clone https://github.com/emegua19/credit-complaint-chatbot.git
cd credit-complaint-chatbot

# Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

##  Reporting

| Report                     | Location                            |
| -------------------------- | ----------------------------------- |
| Interim Report (Tasks 1–2) | `reports/interim/interim_report.md` |
| Final Report (Tasks 3–4)   | `reports/final/final_report.md`     |
| Visualizations             | `reports/figures/`                  |

---

##  What's Next (Final Phase)

* Implement semantic search with `retriever.py`
* Add generative LLM response in `generator.py`
* Build a user-facing chatbot UI
* Evaluate output quality and user experience

---

## Author

Yitbarek Geletu
10 Academy – Week 6 RAG Project

