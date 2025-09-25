# AI Document Summarizer & Q&A System

This project summarizes documents and answers questions using RAG with LlamaIndex.

## Setup
1. Install dependencies: `pip install -r requirements.txt` (bana le requirements.txt).
2. Set OPENAI_API_KEY environment variable.
3. Place PDFs in `data/` folder.

## Run
- Index: `python rag_indexer.py`
- Q&A: `python rag_main.py`
- Summarize: `python rag_summarizer.py` 