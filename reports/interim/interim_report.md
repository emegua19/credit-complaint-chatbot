# Interim Report: Credit Complaint Chatbot (Tasks 1–2)

**Author**: Yitbarek Geletaw  
**Date**: August 17, 2025  
**GitHub**: [https://github.com/emegua19/credit-complaint-chatbot](https://github.com/emegua19/credit-complaint-chatbot)

---

## Project Overview

The Credit Complaint Chatbot project develops an AI-powered tool for **CrediTrust** to analyze consumer complaints from the **Consumer Financial Protection Bureau (CFPB)** dataset. This addresses a major business challenge: manual complaint analysis is slow, inconsistent, and prevents teams from making timely product and compliance decisions.

The chatbot leverages a **Retrieval-Augmented Generation (RAG) pipeline** to enable semantic search over millions of complaint narratives. This allows Product, Risk, and Compliance teams to query the data in natural language and instantly obtain grounded insights.

This interim report summarizes:

- **Task 1: Data Loading, Preprocessing, and Exploratory Data Analysis (EDA)** – Preparing a clean dataset, exploring complaint distributions, and identifying key business insights.  
- **Task 2: Chunking and Embedding** – Transforming narratives into semantically searchable embeddings using a structured pipeline.

---

## Task 1: Exploratory Data Analysis and Preprocessing

### Dataset and Preprocessing

The raw dataset (`data/raw/complaints.csv`, **5.63 GB**) was processed using `src/loader.py` with **chunked loading (`chunksize=10,000`)** to handle memory limits.

Steps included:

- **Filtering**: Retained complaints for five products:
  - Credit Card  
  - Consumer Loan  
  - Payday/Title/Personal Loan  
  - Checking or Savings Account  
  - Money Transfer / Virtual Currency / Money Service
- Removed empty or null narratives.
- **Cleaning**: Lowercasing, removing special characters, and normalizing whitespace.
- **Output**: Filtered dataset saved to `data/processed/filtered/filtered_complaints.csv.gz`.

---

### EDA Findings

- **Total complaints after filtering**: 479,110  
- **Product distribution**: Credit Card and Checking/Savings dominate  
- **Narrative lengths**: Average 202 words (range: 20–1,000+ words)  
- **Visualizations**:  
  - Product distribution  
  - Narrative length distribution  
  - Complaint trends over time  
  - Product vs. issue type heatmap  
  - Company distribution  
  - System diagram  

---

### Summary of EDA Insights

The analysis shows that consumer complaints are not evenly distributed across financial products. **Credit Card (39.5%)** and **Checking/Savings Accounts (32.4%)** dominate, together representing more than 70% of filtered complaints. This concentration highlights that customer frustrations are heavily tied to everyday financial products, which should be CrediTrust’s immediate focus for improving service quality.

Narrative length distribution reveals a wide variation, from very short reports (less than 50 words) to highly detailed descriptions exceeding 1,000 words. The average length of 202 words indicates that most complaints are moderately detailed, requiring an approach that can handle both concise and lengthy narratives. This has direct implications for the **chunking strategy** in Task 2, ensuring important context is not lost during preprocessing.

The **temporal trend analysis** suggests spikes in complaint volume at certain periods, which may correspond to product updates, regulatory shifts, or economic conditions. Recognizing these trends will allow CrediTrust to link consumer concerns to external factors and proactively address systemic issues rather than treating them as isolated cases.

Finally, the **company-level breakdown** highlights a small number of firms receiving a disproportionate share of complaints. This insight is valuable for competitive benchmarking and compliance monitoring. If the chatbot can quickly surface such concentration patterns, it will provide CrediTrust with a powerful tool for both risk management and market positioning.

---

### Business Relevance

EDA revealed that **Credit Card** and **Checking/Savings** complaints dominate (72% of filtered records). This insight directs CrediTrust to focus chatbot testing on these areas first, aligning technical progress with urgent business priorities.

---

## Task 2: Chunking and Embedding

### Chunking

Implemented in `src/chunking.py` using **LangChain’s RecursiveCharacterTextSplitter**:

- **Configuration**: Chunk size = 300 words, overlap = 50 words  
- **Output**: `data/processed/chunked/chunked_narratives.csv.gz`  
- **Rationale**: Balances context preservation and memory efficiency, ensuring meaningful retrieval for long narratives  

---

### Embedding

Implemented in `src/embedding.py`:

- **Model**: `all-MiniLM-L6-v2` (384-dim embeddings)  
- **Vector Store**: ChromaDB with metadata (`chunk_id`, `product`, `complaint_id`)  
- **Output**: Stored in `vector_store/chroma/`  
- **Rationale**: MiniLM balances semantic accuracy with efficiency for large-scale datasets  

---

## Challenges and Solutions

- **Large Dataset (5.63 GB)**: Solved via chunked loading (`chunksize=10,000`)  
- **Time Constraints**: Interim embedding done on subset (1,000 complaints); full embedding planned for final  
- **Visualization Issues**: Debugged blank plots and ensured reproducibility via modular scripts  

---

## Next Steps

- Scale embeddings to all **5.6M complaints**  
- Implement `src/retriever.py` and `src/generator.py` for semantic search and answer generation  
- Build **Gradio interface** in `app/app.py`  
- Add **evaluation metrics** (retrieval accuracy, response quality) for final report  

---

## Repository Structure (Relevant to Interim)

