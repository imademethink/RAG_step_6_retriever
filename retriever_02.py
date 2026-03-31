import math

"""   
 
  Hybrid Search
  Reciprocal Rank Fusion (RRF)

"""
# --- 1. DATA PREPARATION ---
documents = [
    {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "doc2", "text": "Artificial Intelligence is transforming retrieval systems with vector search."},
    {"id": "doc3", "text": "Foxes are agile and clever animals found in the wild."},
]


# --- 2. RETRIEVAL METHODS ---
def sparse_search(query, docs):
    """Simple BM25-style keyword matching simulation."""
    scores = []
    query_terms = query.lower().split()
    for doc in docs:
        # Count keyword overlap
        score = sum(1 for term in query_terms if term in doc["text"].lower())
        scores.append((doc["id"], score))
    # Return document IDs ordered by keyword relevance
    return [id for id, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


def dense_search(query, docs):
    """Semantic vector search simulation."""
    # In production, use embeddings (e.g., OpenAI or Sentence-Transformers)
    # This mock ranks based on document length similarity to query length
    scores = []
    for doc in docs:
        score = 1 / (1 + abs(len(query) - len(doc["text"])))
        scores.append((doc["id"], score))
    return [id for id, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


# --- 3. HYBRID FUSION (RRF) ---
def reciprocal_rank_fusion(results_list, k=60):
    """Combines multiple ranked lists into one using the RRF algorithm."""
    rrf_scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            # Standard RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort documents by their combined RRF score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# --- 4. EXECUTION ---
query = "quick fox"

# Get results from both 'heads'
lexical_results = sparse_search(query, documents)
semantic_results = dense_search(query, documents)

# Merge results
hybrid_results = reciprocal_rank_fusion([lexical_results, semantic_results])

print(f"Query: {query}")
print("Final Hybrid Rankings:")
for doc_id, score in hybrid_results:
    print(f"- {doc_id}: Score {score:.4f}")
