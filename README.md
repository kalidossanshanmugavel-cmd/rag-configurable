## Configurable RAG Architecture

```mermaid
flowchart TD

    CONFIG["config.py<br/>Embedding Provider + Model<br/>LLM Provider + Model"]

    CONFIG --> ING

    subgraph ING["1. INGESTION - ingestion.py"]
        PDF["PDF Documents"]
        LOAD["Load Documents"]
        CLEAN["Clean Text"]
        CHUNK["Sentence Chunking"]
        EMB["Embedding Model<br/>Hugging Face / OpenAI / Gemini"]
        NORM["Normalize Embeddings"]
        PDF --> LOAD --> CLEAN --> CHUNK --> EMB --> NORM
    end

    NORM --> FAISS["FAISS Vector Store<br/>IndexFlatIP<br/>Cosine Similarity"]

    QUESTION["User Question"]

    subgraph RET["2. RETRIEVAL - retrieval.py"]
        QEMB["Same Embedding Model"]
        QNORM["Normalize Query Vector"]
        SEARCH["FAISS Similarity Search"]
        TOPK["Top-K Relevant Chunks"]
        QEMB --> QNORM --> SEARCH --> TOPK
    end

    QUESTION --> QEMB
    FAISS --> SEARCH

    subgraph AUG["3. AUGMENTATION - augmentation.py"]
        CONTEXT["Build Context"]
        PROMPT["Create Prompt<br/>Question + Retrieved Context"]
        CONTEXT --> PROMPT
    end

    TOPK --> CONTEXT
    QUESTION --> PROMPT

    subgraph GEN["4. GENERATION - generation.py"]
        LLM["Configured LLM<br/>Gemini / OpenAI / Ollama"]
        ANSWER["Final Answer"]
        LLM --> ANSWER
    end

    CONFIG --> LLM
    PROMPT --> LLM
```

### RAG Flow

**Ingestion**

`PDF → Load → Clean → Chunk → Embedding → Normalize → FAISS`

**Retrieval**

`Question → Same Embedding Model → Normalize → FAISS Search → Top-K Chunks`

**Augmentation**

`Question + Retrieved Chunks → Context → Prompt`

**Generation**

`Prompt → Configured LLM → Final Answer`

> **Important:** Document embeddings and query embeddings must use the same embedding model.  
> The embedding provider and LLM provider can be different.