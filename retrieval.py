# ============================================================
# IMPORTS
# ============================================================

from config import CONFIG

from ingestion import create_embeddings


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    index,
    chunks
):


    # ========================================================
    # 1. CREATE QUERY EMBEDDING
    #
    # create_embeddings() also normalizes the query vector.
    # ========================================================

    query_embedding = create_embeddings(
        [query]
    )


    # ========================================================
    # 2. GET TOP-K
    # ========================================================

    top_k = min(

        CONFIG["top_k"],

        len(chunks)

    )


    # ========================================================
    # 3. SEARCH FAISS
    #
    # IndexFlatIP returns similarity scores.
    #
    # Higher score = more similar
    # ========================================================

    scores, indices = index.search(

        query_embedding,

        top_k

    )


    # ========================================================
    # 4. GET RETRIEVED CHUNKS
    # ========================================================

    retrieved_chunks = []


    for position, chunk_index in enumerate(
        indices[0]
    ):


        # FAISS may return -1 if no result exists

        if chunk_index < 0:

            continue


        chunk = chunks[
            chunk_index
        ].copy()


        # Add similarity score

        chunk["similarity"] = float(

            scores[0][position]

        )


        retrieved_chunks.append(
            chunk
        )


    return retrieved_chunks