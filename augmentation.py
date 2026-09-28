def create_prompt(query, retrieved_chunks):
    """Combine retrieved context and user question into the final prompt."""

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"""Source: {chunk['source']}
            Page: {chunk['page']}

            Content:
            {chunk['text']}"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the context provided below.

==============================
CONTEXT
==============================

{context}

==============================
QUESTION
==============================

{query}

==============================
RULES
==============================

1. Use only the provided context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is not available in the context, say:
   "The information is not available in the provided documents."
5. Keep the answer clear and concise.
6. Mention the source document and page when possible.
"""

    return prompt
