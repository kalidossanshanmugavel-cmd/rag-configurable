# ============================================================
# IMPORTS
# ============================================================

from ingestion import create_vector_store
from retrieval import retrieve
from augmentation import create_prompt
from generation import generate_answer


# ============================================================
# 1. INGESTION
# ============================================================

index, chunks = create_vector_store()


# ============================================================
# 2. USER QUERY
# ============================================================

query = input(
    "\nAsk your question: "
)


# ============================================================
# 3. RETRIEVAL
# ============================================================

retrieved_chunks = retrieve(
    query,
    index,
    chunks
)


# ============================================================
# DISPLAY RETRIEVED CHUNKS
# ============================================================

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")


for i, chunk in enumerate(
    retrieved_chunks,
    start=1
):

    print(f"\nResult {i}")

    print(
        f"Source     : {chunk['source']}"
    )

    print(
        f"Page       : {chunk['page']}"
    )

    print(
        f"Similarity : {chunk['similarity']:.4f}"
    )

    print(
        f"Content    : {chunk['text']}"
    )


# ============================================================
# 4. AUGMENTATION
# ============================================================

prompt = create_prompt(
    query,
    retrieved_chunks
)


# ============================================================
# 5. GENERATION
# ============================================================

answer = generate_answer(
    prompt
)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)