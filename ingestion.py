# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import faiss
import numpy as np

from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from google import genai

from config import (
    CONFIG,
    EMBEDDING_API_KEY
)


# ============================================================
# 1. LOAD PDF DOCUMENTS
# ============================================================

def load_documents():

    data_path = Path("data")

    pdf_files = list(
        data_path.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF files found in data folder."
        )

    documents = []

    for pdf_path in pdf_files:

        print(
            f"Loading: {pdf_path.name}"
        )

        reader = PdfReader(
            pdf_path
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text and page_text.strip():

                documents.append({

                    "source": pdf_path.name,

                    "page": page_number,

                    "text": page_text

                })

    print(
        f"\nTotal PDFs: {len(pdf_files)}"
    )

    print(
        f"Total Pages: {len(documents)}"
    )

    return documents


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

def create_chunks(documents):

    chunk_size = CONFIG[
        "chunk_size"
    ]

    overlap = CONFIG[
        "chunk_overlap"
    ]

    if overlap >= chunk_size:

        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    for document in documents:

        source = document["source"]

        page = document["page"]

        text = document["text"]


        # ----------------------------------------------------
        # CLEAN TEXT
        # ----------------------------------------------------

        lines = text.splitlines()

        clean_lines = [

            line.strip()

            for line in lines

            if line.strip()

        ]

        text = " ".join(
            clean_lines
        )


        # ----------------------------------------------------
        # SENTENCE TOKENIZATION
        # ----------------------------------------------------

        sentences = sent_tokenize(
            text
        )


        # ----------------------------------------------------
        # CREATE SENTENCE-BASED CHUNKS
        # ----------------------------------------------------

        start = 0

        while start < len(sentences):

            end = (
                start
                + chunk_size
            )

            chunk_sentences = (
                sentences[start:end]
            )

            if not chunk_sentences:

                break

            chunk_text = " ".join(
                chunk_sentences
            )

            chunks.append({

                "text": chunk_text,

                "source": source,

                "page": page

            })


            # Move forward with overlap

            start = (
                end
                - overlap
            )


    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    provider = CONFIG[
        "embedding_provider"
    ]

    model_name = CONFIG[
        "embedding_model"
    ]


    # ========================================================
    # SELECT EMBEDDING PROVIDER
    # ========================================================

    match provider:


        # ----------------------------------------------------
        # HUGGING FACE
        # ----------------------------------------------------

        case "huggingface":

            model = SentenceTransformer(
                model_name
            )

            embeddings = model.encode(

                texts,

                convert_to_numpy=True

            )


        # ----------------------------------------------------
        # OPENAI
        # ----------------------------------------------------

        case "openai":

            if not EMBEDDING_API_KEY:

                raise ValueError(
                    "OPENAI_API_KEY is not configured"
                )

            client = OpenAI(
                api_key=EMBEDDING_API_KEY
            )

            response = client.embeddings.create(

                model=model_name,

                input=texts

            )

            embedding_list = []

            for item in response.data:

                embedding_list.append(
                    item.embedding
                )

            embeddings = np.array(
                embedding_list
            )


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        case "gemini":

            if not EMBEDDING_API_KEY:

                raise ValueError(
                    "GEMINI_API_KEY is not configured"
                )

            client = genai.Client(
                api_key=EMBEDDING_API_KEY
            )

            response = (
                client.models.embed_content(

                    model=model_name,

                    contents=texts

                )
            )


            # ------------------------------------------------
            # VALIDATE GEMINI RESPONSE
            # ------------------------------------------------

            if not response.embeddings:

                raise ValueError(
                    "Gemini did not return embeddings"
                )

            embedding_list = []

            for item in response.embeddings:

                if item.values is None:

                    raise ValueError(
                        "Gemini returned an empty embedding"
                    )

                embedding_list.append(
                    item.values
                )

            embeddings = np.array(
                embedding_list
            )


        # ----------------------------------------------------
        # INVALID PROVIDER
        # ----------------------------------------------------

        case _:

            raise ValueError(

                f"Unsupported embedding provider: {provider}"

            )


    # ========================================================
    # CONVERT TO FLOAT32
    # ========================================================

    embeddings = np.asarray(

        embeddings,

        dtype="float32"

    )


    # ========================================================
    # NORMALIZE EMBEDDINGS
    #
    # Required when using IndexFlatIP for cosine similarity.
    # After L2 normalization:
    #
    # Inner Product ≈ Cosine Similarity
    # ========================================================

    faiss.normalize_L2(
        embeddings
    )


    return embeddings


# ============================================================
# 4. CREATE VECTOR STORE
# ============================================================

def create_vector_store():


    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    documents = load_documents()


    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        documents
    )


    if not chunks:

        raise ValueError(
            "No chunks were created"
        )


    # --------------------------------------------------------
    # GET TEXT FROM CHUNKS
    # --------------------------------------------------------

    chunk_texts = [

        chunk["text"]

        for chunk in chunks

    ]


    # --------------------------------------------------------
    # CREATE DOCUMENT EMBEDDINGS
    # --------------------------------------------------------

    embeddings = create_embeddings(
        chunk_texts
    )


    print(
        f"\nEmbedding Provider: "
        f"{CONFIG['embedding_provider']}"
    )

    print(
        f"Embedding Model: "
        f"{CONFIG['embedding_model']}"
    )

    print(
        f"Embedding Shape: "
        f"{embeddings.shape}"
    )


    # --------------------------------------------------------
    # GET EMBEDDING DIMENSION
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print(
        f"Embedding Dimension: "
        f"{dimension}"
    )


    # --------------------------------------------------------
    # CREATE FAISS INDEX
    #
    # IndexFlatIP = Inner Product
    #
    # Since embeddings are normalized,
    # this behaves like cosine similarity.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )


    # --------------------------------------------------------
    # ADD DOCUMENT VECTORS
    # --------------------------------------------------------

    index.add(
        embeddings
    )


    print(
        "\nFAISS vector index created"
    )

    print(
        "Similarity: Cosine Similarity"
    )

    print(
        f"Vectors stored in FAISS: "
        f"{index.ntotal}"
    )


    return index, chunks