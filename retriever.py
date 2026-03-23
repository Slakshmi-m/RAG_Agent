"""
retriever.py — Hybrid Retrieval: Semantic Search + BM25 Keyword Search

WHY HYBRID?
───────────
Pure semantic search (embeddings) is great at understanding *intent*,
but struggles with exact string matches. In lighting standards, users
often search for specific norm codes like "EN 12464-1", "DIN 5035",
"ASR A3.4", or specific values like "500 lux". These are exact strings
that embedding models may not rank highly if the surrounding context
is different.

BM25 (Best Match 25) is a classical keyword ranking algorithm —
essentially a smarter TF-IDF. It excels at exact term matching and is
very fast. But it misses synonyms and cross-language matches.

HYBRID APPROACH (Reciprocal Rank Fusion):
──────────────────────────────────────────
1. Run semantic search → get top-K results with ranks
2. Run BM25 keyword search → get top-K results with ranks
3. Merge using Reciprocal Rank Fusion (RRF):
     score(chunk) = 1/(rank_semantic + k) + 1/(rank_bm25 + k)
   where k=60 is a smoothing constant (standard RRF parameter)
4. Re-rank by combined score → return top-K

This gives the best of both: semantic understanding + exact matching.

EXAMPLE:
  Query: "EN 12464-1 Büro 500 lux"
  
  Semantic only → finds "Bürobeleuchtung Anforderungen" chunks (good intent match)
  BM25 only     → finds chunks containing "EN 12464-1" literally (good exact match)
  Hybrid        → finds chunks that match BOTH — highest quality results
"""

import re
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# BM25 Index 

def build_bm25_index(chunks: List[Dict]) -> Tuple[BM25Okapi, List[Dict]]:
    """
    Build a BM25 index from the chunk corpus.

    Tokenisation strategy:
    - Lowercase
    - Split on whitespace and punctuation
    - Keep numbers intact (important for norm identifiers like "12464")
    - Keep hyphenated terms intact (e.g. "EN-12464", "Lux-Wert")

    Returns (bm25_index, chunks) — chunks list is kept aligned with
    the BM25 index so we can map result indices back to chunk dicts.
    """

    tokenised_corpus = [tokenise_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenised_corpus)
    return bm25, chunks


# Reciprocal Rank Fusion 

def reciprocal_rank_fusion(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Merge semantic and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where rank is 1-indexed and k=60 is the standard smoothing constant.

    A chunk that appears at rank 1 in semantic and rank 3 in BM25 gets:
      score = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

    A chunk appearing only in semantic at rank 1 gets:
      score = 1/(60+1) = 0.01639

    So a chunk that's strong in BOTH methods always beats a chunk
    that's only strong in one.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    # Score semantic results
    for rank, chunk in enumerate(semantic_results, start=1):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    # Sort by combined score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(scores[cid], 5)
        results.append(chunk)

    return results


# Retrieval Functions 

def semantic_search(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    top_k: int = 10,
) -> List[Dict]:
    """
    Retrieve top-K chunks by cosine similarity in the vector store.
    Returns list of chunk dicts with 'similarity' field.
    """
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][i]
        meta = results["metadatas"][0][i]
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page_num": meta["page_num"],
            "section": meta.get("section", ""),
            "content_type": meta.get("content_type", "prose"),
            "similarity": round(similarity, 4),
        })

    return retrieved


def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    chunks: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Retrieve top-K chunks by BM25 keyword score.
    Returns list of chunk dicts with 'bm25_score' field.
    """

    query_tokens = tokenise_for_bm25(query)
    scores = bm25_index.get_scores(query_tokens)

    # Get top-K indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    retrieved = []
    for idx in top_indices:
        if scores[idx] < 0.01:   # filter near-zero scores (irrelevant)
            break
        chunk = chunks[idx].copy()
        chunk["bm25_score"] = round(float(scores[idx]), 4)
        retrieved.append(chunk)

    return retrieved


def hybrid_search(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    bm25_index: BM25Okapi,
    bm25_chunks: List[Dict],
    top_k: int = 5,
    fetch_k: int = 15,     # candidates to fetch from each method before merging
) -> List[Dict]:
    """
    Full hybrid retrieval pipeline.

    Steps:
    1. Semantic search → top fetch_k candidates
    2. BM25 keyword search → top fetch_k candidates
    3. Merge with RRF
    4. Return final top_k

    The fetch_k > top_k pattern ("over-fetch then re-rank") is standard
    in hybrid RAG systems — you cast a wider net before merging.

    Args:
        query: user's natural language question
        collection: ChromaDB vector collection
        embed_model: SentenceTransformer instance
        bm25_index: pre-built BM25Okapi index
        bm25_chunks: original chunks list (aligned with BM25 index)
        top_k: final number of results to return
        fetch_k: how many candidates to fetch from each method

    Returns:
        List of top_k chunk dicts, sorted by RRF score (best first)
    """
    # Run both searches
    sem_results = semantic_search(query, collection, embed_model, top_k=fetch_k)
    kw_results = bm25_search(query, bm25_index, bm25_chunks, top_k=fetch_k)

    # Merge with RRF
    merged = reciprocal_rank_fusion(sem_results, kw_results)

    return merged[:top_k]


# Context Formatting 

def format_context_for_llm(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a prompt-ready context block.

    Norm table chunks get a special label so the LLM knows to treat
    them as structured data (not garbled prose).
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        content_label = (
            "NORM TABLE (structured data — cite values directly)"
            if chunk.get("content_type") == "norm_table"
            else "TEXT"
        )
        section_label = f" | Section: {chunk['section']}" if chunk.get("section") else ""
        score_label = (
            f"RRF: {chunk['rrf_score']}"
            if "rrf_score" in chunk
            else f"Sim: {chunk.get('similarity', '?')}"
        )

        header = (
            f"[Source {i} | Page {chunk['page_num']}{section_label} | "
            f"{content_label} | {score_label}]"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


# Confidence Scoring 

def retrieval_confidence(results: List[Dict]) -> str:
    """
    Assess retrieval confidence based on top result's original semantic similarity.
    Ignores RRF scores because their scale (max ~0.033) breaks the threshold logic.
    """
    if not results:
        return "low"

    # Search the top results for the highest original cosine similarity score
    # Default to 0 if the top results were only found via BM25
    highest_sim = max([res.get("similarity", 0.0) for res in results[:3]])

    # Standard sentence-transformer cosine similarity thresholds
    if highest_sim > 0.70:
        return "high"
    elif highest_sim > 0.50:
        return "medium"
    else:
        return "low"

# Tokenise for BM25 (keep norm identifiers intact)

def tokenise_for_bm25(text: str) -> List[str]:
    """Extract tokens, keeping technical identifiers like EN-12464-1 intact."""
    return re.findall(r"[\w][\w\-\.]*", text.lower())