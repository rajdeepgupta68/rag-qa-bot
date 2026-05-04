---
title: RAG Q&A Bot
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
---

# RAG Q&A Bot

A Retrieval Augmented Generation (RAG) Q&A bot built with:
- Llama 3 via Groq (HuggingFace Inference)
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Gradio for the UI

## How to use
1. Upload a PDF, TXT or DOCX document
2. Ask any question about the document
3. Get answers with citations

## Run locally
```bash
pip install -r requirements.txt
python src/ingest.py
python src/app.py
```