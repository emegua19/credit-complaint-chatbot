#!/bin/bash

# Data folders
mkdir -p data/raw
mkdir -p data/processed/filtered
mkdir -p data/processed/chunked

# Notebooks
mkdir -p notebooks
touch notebooks/eda.ipynb

# Source code
mkdir -p src/utils
touch src/data_loader.py
touch src/preprocessing.py
touch src/chunking.py
touch src/embedding.py
touch src/retriever.py
touch src/generator.py
touch src/utils/text_cleaning.py
touch src/utils/logging.py

# Vector store
mkdir -p vector_store/chroma
touch vector_store/index_metadata.json

# App
mkdir -p app
touch app/app.py

# Reports
mkdir -p reports/interim
mkdir -p reports/final
mkdir -p reports/figures
touch reports/interim/interim_report.md
touch reports/final/final_report.md

# Tests
mkdir -p tests
touch tests/test_preprocessing.py
touch tests/test_retriever.py
touch tests/test_generator.py

# Root files
touch requirements.txt
touch .gitignore
touch README.md

# GitHub CI
mkdir -p .github/workflows
touch .github/workflows/ci.yml

echo " Project structure created successfully!"
