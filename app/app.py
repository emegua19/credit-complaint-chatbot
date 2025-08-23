# app.py

import os
import sys
import gradio as gr
from typing import List
from langchain_core.documents import Document
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# === Config ===
MODEL_NAME = "google/flan-t5-base"
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "src", "..", "vector_store", "chroma")
THEME = gr.themes.Default(primary_hue="emerald", secondary_hue="lime")  # Professional color scheme

# Dynamically import components
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipline import ComplaintRetriever
from rag_pipline import generate_answer  # Assuming generate_answer is exported

# === Load LLM pipeline ===
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=300)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Initialize retriever
try:
    retriever = ComplaintRetriever(vector_store_path=VECTOR_STORE_PATH)
except Exception as e:
    print(f"Error initializing retriever: {e}")
    raise

def get_answer(question: str, product_filter: str = "All") -> tuple[str, str]:
    """
    Retrieve chunks and generate an answer with a sample source.
    """
    try:
        # Adjust product_filter for "All" case
        product_filter = product_filter if product_filter != "All" else None
        chunks = retriever.retrieve(query=question, product_filter=product_filter)
        answer = generate_answer(chunks, question)
        source = chunks[0].page_content[:200] + "..." if chunks else "No source available"
        return answer, source
    except Exception as e:
        return f"Error: {e}", "No source available"

# Gradio interface with attractive design
with gr.Blocks(theme=THEME, title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Assistant
        ### Get Expert Insights on Customer Complaints
        Ask about financial issues and explore complaint data with ease.
        """
    )
    
    with gr.Row(variant="panel").style(height=50):
        gr.Image(value="https://via.placeholder.com/150", height=50)  # Placeholder logo
    
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are common problems with credit cards?",
                lines=2,
                elem_id="question_input"
            ).style(container=True, border_color="#4CAF50")
            product_filter_input = gr.Dropdown(
                label="Filter by Product",
                choices=["All", "Credit card", "Savings account", "Mortgage", "Debt collection", "Payday loan"],
                value="All",
                elem_id="product_filter_input"
            ).style(container=True)
            submit_btn = gr.Button("Get Answer", variant="primary").style(full_width=True)
        
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                elem_id="answer_output"
            ).style(container=True, border_color="#2E7D32", padding=10)
            source_output = gr.Textbox(
                label="Sample Source",
                interactive=False,
                elem_id="source_output"
            ).style(container=True, border_color="#2E7D32", padding=10)

    submit_btn.click(
        fn=get_answer,
        inputs=[question_input, product_filter_input],
        outputs=[answer_output, source_output],
        _js="() => [document.getElementById('question_input').value, document.getElementById('product_filter_input').value]"
    )

    # Add professional CSS styling
    demo.css = """
    .gradio-container {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 15px;
        padding: 20px;
    }
    #question_input, #answer_output, #source_output {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .gradio-button {
        background-color: #4CAF50;
        color: white;
        transition: background-color 0.3s;
    }
    .gradio-button:hover {
        background-color: #45a049;
    }
    """

demo.launch()