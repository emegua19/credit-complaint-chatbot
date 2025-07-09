# Intelligent Complaint Analysis Chatbot (RAG-Powered)

This project delivers a Retrieval-Augmented Generation (RAG) chatbot designed for CrediTrust Financial to enable fast, insightful analysis of real customer complaints through natural language queries.

Built using large language models (LLMs), semantic search, and vector databases, the chatbot streamlines complaint review and decision-making for internal teams.

---

## Problem

CrediTrust handles millions of customer complaints spanning financial products like credit cards, loans, and Buy Now, Pay Later (BNPL). Manual analysis is:

* Time-consuming
* Inconsistent
* Prone to missing trends and unresolved issues

---

## Solution

With this chatbot, internal teams can ask:

> "Why are users unhappy with BNPL?"

And instantly receive:

* Concise, grounded answers
* Based on real complaint narratives
* Delivered in seconds

### Powered by:

* Text chunking and embedding
* Semantic search via ChromaDB
* LLM-based generation (Flan-T5)

---

## Table of Contents

* [Problem](#problem)
* [Solution](#solution)
* [Visual Overview](#visual-overview)
* [Project Structure](#project-structure)
* [Final Deliverables](#final-deliverables)
* [Tech Stack](#tech-stack)
* [Setup Instructions](#setup-instructions)
* [Usage Instructions](#usage-instructions)
* [License](#license-information)
* [Contact](#contact-information)

---

## Visual Overview

```
CrediTrust Teams (Support/Product)
          ↓
   Natural Language Questions
          ↓
  Semantic Retrieval (ChromaDB)
          ↓
 LLM Answer Generation (Flan-T5)
          ↓
  Grounded Answer from Complaints
          ↓
     Business Decisions
```

---

## Project Structure

```
credit-complaint-chatbot/
├── .github/                    # GitHub Actions workflows
│   └── workflows/ci.yml        # CI pipeline for testing
├── .gitignore                  # Ignored files (e.g., venv/)
├── LICENSE                     # Project license (e.g., MIT)
├── README.md                   # Project overview and setup
├── config/                     # Configuration files
│   └── settings.yaml           # Model and path settings
├── data/                       # Raw and processed complaint data
│   ├── raw/                    # Original CFPB CSV (~5.6 GB)
│   │   └── complaints.csv
│   └── processed/              # Filtered & chunked (~1.12 GB)
│       ├── chunked/            # Chunked narratives
│       └── filtered/           # Filtered datasets
├── notebooks/                  # Exploratory data analysis
│   ├── eda.ipynb
│   └── narrative_analysis.ipynb
├── src/                        # Core Python modules
│   ├── chunking.py             # Text splitting
│   ├── embedding.py            # Embedding with ChromaDB
│   ├── generator.py            # LLM answer generation
│   ├── retriever.py            # Semantic retrieval
│   └── __init__.py             # Package initialization
├── tests/                      # Unit and integration tests
│   ├── test_generator.py       # Generator tests
│   ├── test_retriever.py       # Retriever tests
│   └── __init__.py             # Package initialization
├── vector_store/               # Persistent vector database
│   ├── chroma/                 # ChromaDB index files
│   └── embedded_ids.txt        # Embedded IDs mapping
├── app/                        # Chat interface
│   └── app.py                  # Gradio-based UI
├── reports/                    # Documentation and reports
│   ├── interim/                # Submitted interim report
│   └── final/                  # Final report
├── requirements.txt            # Python dependencies
└── venv/                       # Virtual environment (ignored)
```

---

## Final Deliverables

### Task 1: EDA and Preprocessing

* Cleaned 5.6M complaints to 479,110 usable entries
* Removed nulls, normalized narratives, calculated word stats
* Saved compressed .csv (\~1.12 GB)

### Task 2: Chunking and Embedding

* Used RecursiveCharacterTextSplitter (chunk=300, overlap=50)
* Embedded using sentence-transformers/all-MiniLM-L6-v2
* Stored in ChromaDB with metadata: product, complaint\_id

### Task 3: RAG Pipeline Evaluation

| Question                                            | Answer Summary                                  | Product         |
| --------------------------------------------------- | ----------------------------------------------- | --------------- |
| Why are customers dissatisfied with credit cards?   | Hidden fees, unauthorized charges, poor support | Credit card     |
| What are common problems with Buy Now, Pay Later?   | Late fees, unclear repayment terms              | BNPL            |
| Are users experiencing delays with money transfers? | Yes, delays and failed transactions reported    | Money transfers |

Insight: RAG performs well on common categories; BNPL and transfer responses indicate future tuning opportunities.

### Task 4: Interactive Chat Interface

* Built `app.py` using Gradio
* User input box, product filter, submit and clear buttons
* Clean interface suitable for non-technical users
* Includes `test_generator.py` and `test_retriever.py` for validation

---

## Tech Stack

| Component      | Library / Tool                           |
| -------------- | ---------------------------------------- |
| Data Prep      | pandas, nltk, re, matplotlib             |
| Chunking       | LangChain RecursiveCharacterTextSplitter |
| Embedding      | sentence-transformers MiniLM-L6-v2       |
| Vector Store   | ChromaDB via LangChain                   |
| LLM Generator  | transformers (Flan-T5)                   |
| Chat UI        | gradio                                   |
| Testing and CI | pytest, GitHub Actions                   |

---

## Setup Instructions

```bash
# Clone the repository
$ git clone https://github.com/emegua19/credit-complaint-chatbot.git
$ cd credit-complaint-chatbot

# Create a virtual environment (optional)
$ python -m venv .venv
$ source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt

# Generate vector store (if not already done)
$ python src/embedding.py
```

---

## Usage Instructions

```bash
# Run the chatbot interface
$ python app/app.py
```

Then open your browser to: [http://localhost:7860](http://localhost:7860)

Example Question:

> Why are users unhappy with personal loans?

Select a product category from the dropdown to filter results (optional).

---

## License Information

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact Information

* Author: Yitbarek Geletaw
* Email: ebnenode@gmail.com
* 10 Academy: Week 6 AI Mastery RAG Project

