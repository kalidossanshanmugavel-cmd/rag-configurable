# Configurable Manual RAG

A simple manual RAG project demonstrating four main stages:

1. Ingestion
2. Retrieval
3. Augmentation
4. Generation

## Architecture

PDF -> Load -> Chunk -> Embedding -> FAISS
                                  |
User Query -> Query Embedding -> Retrieval
                                  |
                                  v
                          Retrieved Chunks
                                  |
                                  v
                             Augmentation
                                  |
                                  v
                                Prompt
                                  |
                                  v
                              Generation
                                  |
                                  v
                            Final Answer

## Providers

### Embeddings

- Hugging Face / Sentence Transformers
- OpenAI
- Gemini

### LLM

- Ollama
- OpenAI
- Gemini

Embedding provider and LLM provider are independent.

Example:

Hugging Face Embedding + Gemini LLM

## Setup

### 1. Create virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

### 2. Install packages

    pip install -r requirements.txt

### 3. Download NLTK tokenizer data

Run once:

    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

### 4. Environment variables

Copy `.env.example` to `.env`.

Add only the API keys required by your selected providers.

Ollama and local Hugging Face SentenceTransformer models do not require an API key.

### 5. Add PDF documents

Place one or more PDF files inside:

    data/

### 6. Select models

Edit `config.py`.

Example:

    "embedding_provider": "huggingface",
    "embedding_model": "all-MiniLM-L6-v2",

    "llm_provider": "gemini",
    "llm_model": "gemini-3.6-flash",

Use model IDs that are available to your provider/account.

### 7. Run

    python main.py

## Ollama

If using Ollama, install/run Ollama separately and pull the model configured in `config.py`.

Example:

    ollama pull llama3.2

Then set:

    "llm_provider": "ollama",
    "llm_model": "llama3.2",

## Important

The same embedding provider/model must be used for both document embeddings and query embeddings.

Do not create the FAISS index with one embedding model and query it with another embedding model.
