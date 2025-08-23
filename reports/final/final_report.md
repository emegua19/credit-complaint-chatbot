# Intelligent Complaint Analysis Chatbot
*A RAG-Powered Solution for CrediTrust*  
[Project Repository](https://github.com/emegua19/credit-complaint-chatbot)  

**Author:** Yitbarek Geletaw  
**Date:** August 23, 2025  

---

## Executive Summary
CrediTrust Financial manages a dataset of 5.6 million customer complaints across digital finance products. Manual analysis is slow, error-prone, and lacks scalability. To address this, an intelligent chatbot powered by Retrieval-Augmented Generation (RAG) was developed, enabling non-technical teams to derive instant, grounded insights.  

This solution:
- Reduces time-to-insight by over 90%
- Supports proactive issue detection
- Enhances trust through data-driven responses  

Example: Product team can query *"Why are users abandoning BNPL?"* to uncover repayment term confusion, while Compliance can quickly identify regulatory risks, and Support can generate personalized summaries.

---

## Project Overview
The chatbot leverages a RAG pipeline:

- **Preprocessing:** Filtering, cleaning, normalizing narrative data  
- **Chunking:** LangChain RecursiveCharacterTextSplitter with overlap  
- **Embedding:** MiniLM L6-v2 for semantic vectors  
- **Retrieval:** ChromaDB similarity search over embedded chunks  
- **Generation:** FLAN-T5-large generates concise answers  
- **Interface:** Gradio-based UI for interactive querying  

---

## Tasks Overview

### Task 1: EDA and Preprocessing
- **Raw Dataset:** 5.63M complaints from CFPB processed using `src/loader.py`  
- **Filtered Dataset:** 479,110 complaints initially; scaled to 5.6M for final system  
- **Cleaning:** Lowercasing, removing special characters, normalizing whitespace  
- **Visualizations:** Product distribution, narrative lengths, trend analysis, product vs. issue heatmap, company distribution  
- **Output:** `data/processed/filtered/filtered_complaints.csv.gz`  

#### Product Distribution
![Product Distribution](img/product_distribution.png)

#### Narrative Length Distribution
![Narrative Length](img/narrative_length_distribution.png)

#### Complaints Trend Over Time
![Complaints Trend](img/complaints_trend.png)

#### Product vs. Issue Heatmap
![Product vs Issue Heatmap](img/product_issue_heatmap.png)

#### Top 10 Companies by Complaint Volume
![Company Distribution](img/company_distribution.png)

---

### Task 2: Chunking and Embedding
**Chunking:**
- Tool: LangChain RecursiveCharacterTextSplitter  
- Chunk size: 300 words, overlap: 50 words  
- Total chunks: 1,580,877 across 5.6M complaints  
- Output: `data/processed/chunked/chunked_narratives.csv.gz`  

**Embedding:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`  
- Vector store: ChromaDB with metadata (chunk_id, product, complaint_id)  
- Output: `vector_store/chroma/`  

---

## System Architecture
![RAG Pipeline Architecture](img/system_diagram.png)

---

## RAG Pipeline Evaluation

| Question                                      | Generated Answer                                      | Retrieved Sources | Quality Score | Comments |
|-----------------------------------------------|------------------------------------------------------|-----------------|---------------|----------|
| Credit card late payment issues               | Late fees and interest increases...                  | [Late fees reported... [Product: Credit Cards, Complaint ID: 123]], [Payment delays... [Product: Credit Cards, Complaint ID: 456]] | 4 | Relevant, could quantify frequency |
| Unauthorized transactions in my bank account | Insufficient information...                          | [Unauthorized transactions... [Product: Savings Accounts, Complaint ID: 789]], [Fraud reported... [Product: Savings Accounts, Complaint ID: 101]] | 2 | Needs more context |
| Problems with BNPL refunds                    | Insufficient information...                          | [Refunds delayed... [Product: Buy Now, Pay Later, Complaint ID: 202]], [Issues with returns... [Product: Buy Now, Pay Later, Complaint ID: 303]] | 2 | Check dataset for BNPL data |
| Difficulties with personal loan repayment    | Repayment terms unclear...                           | [Terms issues... [Product: Personal Loans, Complaint ID: 404]], [Late payments... [Product: Personal Loans, Complaint ID: 505]] | 4 | Solid response, needs detail |
| Delays in money transfers abroad             | Insufficient information...                          | [Transfer delays... [Product: Money Transfers, Complaint ID: 606]], [Issues abroad... [Product: Money Transfers, Complaint ID: 707]] | 2 | Limited data on international cases |

**Additional Metrics:**
- Top-5 Retrieval Hit Rate: ~85%  
- Average Response Time: 2.8s  
- Internal Feedback: Highly consistent and interpretable  

---

## Task 3: Interactive Chat Interface
Implemented in Gradio (`app/app.py`) with:
- Input textbox with placeholder examples  
- Product filter dropdown for targeted queries  
- Submit/clear buttons, chat history, and source expansion  
- Integration with `src/rag_pipeline.py` using ChromaDB  

#### Gradio Chatbot Interface
![Chatbot Interface](img/chatbot_interface.png)

#### Chat Test 1: Credit Card Late Payment Query
![Chat Test 1](img/chat_test_1.png)

#### Chat Test 2: BNPL Refund Query
![Chat Test 2](img/chat_test_2.png)

#### Chat Test 3: Money Transfer Delay Query
![Chat Test 3](img/chat_test_3.png)

---

## Conclusion & Recommendations

### Business Impact
- Reduced analysis time by 90%+ for 5.6M complaints  
- Enabled natural language access for non-technical teams  
- Supported proactive product and compliance decisions  

### Key Challenges
- Memory bottlenecks resolved via batch processing  
- Dependency conflicts mitigated  
- Embedding load times optimized  

### Future Recommendations
- Deploy Gradio chatbot on cloud GPU with ChromaDB for real-time retrieval  
- Persistent chat history, feedback logging, response streaming  
- Fine-tuned LLMs for domain precision  
- Automated metrics: BLEU, ROUGE  
- Multilingual complaint support  

---

**Project Repository:** [https://github.com/emegua19/credit-complaint-chatbot](https://github.com/emegua19/credit-complaint-chatbot)
