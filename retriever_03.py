import ollama

"""
    
    Reranking

"""

# 1. Configuration
RERANK_MODEL = "sam860/qwen3-reranker:0.6b-Q8_0"


def get_rerank_score(query, document):
    """
    Prompts the reranker model to return a relevance score.
    Some reranker models in Ollama are fine-tuned to output a direct score
    or can be prompted to provide one.
    """
    prompt = f"Query: {query}\nDocument: {document}\nRelevance Score (0-10):"

    response = ollama.generate(
        model=RERANK_MODEL,
        prompt=prompt,
        options={"temperature": 0},  # Keep it deterministic
        stream=False
    )

    # Simple extraction logic: find the first number in the response
    # In production, use a regex or a model specifically for /api/rerank if available
    try:
        score_text = response['response'].strip()
        # Extract the first digit found as a crude score
        score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_text)))
        return score
    except ValueError:
        return 0.0


def rerank_documents(query, documents, top_n=3):
    """
    Takes a list of documents and re-sorts them based on Ollama's evaluation.
    """
    scored_docs = []
    for doc in documents:
        score = get_rerank_score(query, doc)
        scored_docs.append((score, doc))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Return top N documents
    return [doc for score, doc in scored_docs[:top_n]]


# --- Example Usage ---
query = "How do I optimize RAG retrieval?"
retrieved_chunks = [
    "To optimize RAG, focus on chunking and hybrid search.",
    "RAG stands for Retrieval-Augmented Generation.",
    "The best way to improve RAG is using a cross-encoder reranker.",
    "Artificial intelligence is a broad field of study."
]

print("Reranking with Ollama...")
final_results = rerank_documents(query, retrieved_chunks, top_n=2)

print(f"Query: {query}")
for i, doc in enumerate(final_results, 1):
    print(f"Rank {i}: {doc}")
