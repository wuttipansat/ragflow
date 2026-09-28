# RagFlow

RagFlow is a simple Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents. It retrieves relevant text from ChromaDB and sends the context to a free OpenRouter model to generate answers with source references.

## Features

- Extract text from PDF documents
- Clean and split text into overlapping chunks
- Generate multilingual embeddings
- Store and search vectors with ChromaDB
- Retrieve relevant chunks using cosine similarity
- Generate answers through OpenRouter
- Support Thai and English questions
- Display source files, pages, and similarity scores

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- Multilingual E5
- NumPy
- ChromaDB
- OpenRouter API
- PyYAML
- Pytest

## Get Started

Clone the repository:

```bash
git clone https://github.com/wuttipansat/ragflow.git
cd ragflow
```

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install the project:

```powershell
pip install -e ".[dev]"
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Place the input PDF at:

```text
data/raw/sample.pdf
```

Run the pipeline:

```powershell
python scripts/ingest.py
python scripts/index_chunks.py
python scripts/ask.py
```

## Future Improvements

- Support multiple PDF files
- Add token-aware and semantic chunking
- Add hybrid search and reranking
- Evaluate retrieval and answer quality
- Add conversation history
- Add a Streamlit web interface
- Support additional document formats
- Add automated indexing when documents change

## Author

Developed by Wuttipan Satienpaisan
