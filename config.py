import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # huggingface | openai | gemini
    "embedding_provider": "huggingface",
    "embedding_model": "all-mpnet-base-v2",

    # ollama | openai | gemini
    "llm_provider": "gemini",
    "llm_model": "gemini-3.5-flash-lite",

    # RAG settings
    "chunk_size": 3,
    "chunk_overlap": 1,
    "top_k": 3,

    # Ollama local server
    "ollama_url": "http://localhost:11434/api/generate",
}

# Example embedding models:
#
# Hugging Face:
#   all-MiniLM-L6-v2
#   all-MiniLM-L12-v2
#   all-mpnet-base-v2
#   paraphrase-MiniLM-L6-v2
#   multi-qa-MiniLM-L6-cos-v1
#
# OpenAI:
#   text-embedding-3-small
#   text-embedding-3-large
#
# Gemini:
#   gemini-embedding-001
#
# Example LLMs:
#
# Ollama:
#   llama3.2
#   gemma3
#
# OpenAI:
#   Set "llm_model" to a model available to your OpenAI account.
#
# Gemini:
#   Set "llm_model" to a model available to your Gemini API account.

API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "huggingface": None,
    "ollama": None,
}

EMBEDDING_API_KEY = API_KEYS.get(CONFIG["embedding_provider"])
LLM_API_KEY = API_KEYS.get(CONFIG["llm_provider"])
