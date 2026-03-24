import os
import chromadb
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from chunker import extract_and_chunk
from retriever import build_bm25_index, hybrid_search, format_context_for_llm, retrieval_confidence


# Configuration -----------------------------------------------------------------

PDF_PATH        = os.getenv("PDF_PATH", "Beleuchtungspraxis.pdf")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "beleuchtungspraxis"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K           = 3

# Load embedding model and vector store at startup -------------------------------

_embed_model = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_existing = [c.name for c in _client.list_collections()]

if COLLECTION_NAME in _existing:
    _collection = _client.get_collection(COLLECTION_NAME)
    print(f"Vector store loaded: {_collection.count()} chunks")
    # Load raw chunks for BM25 (including the text, not just vectors)
    _all_data = _collection.get(include=["documents", "metadatas"])
    _chunks = [
        {
            "id": _all_data["ids"][i],
            "text": _all_data["documents"][i],
            "page_num": _all_data["metadatas"][i]["page_num"],
            "section": _all_data["metadatas"][i].get("section", ""),
            "content_type": _all_data["metadatas"][i].get("content_type", "prose"),
            "chunk_idx": 0,
        }
        for i in range(len(_all_data["ids"]))
    ]
else:
    print(f"Vector store not found. Building from {PDF_PATH}...")
    _chunks = extract_and_chunk(PDF_PATH)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    BATCH_SIZE = 64
    for i in range(0, len(_chunks), BATCH_SIZE):
        batch = _chunks[i:i + BATCH_SIZE]
        embeddings = _embed_model.encode([c["text"] for c in batch]).tolist()
        _collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "page_num": c["page_num"],
                "section": c["section"],
                "content_type": c["content_type"],
            } for c in batch]
        )
    print(f"Built vector store: {_collection.count()} chunks")

import pickle

BM25_INDEX_PATH  = os.path.join(CHROMA_DIR, f"{COLLECTION_NAME}_bm25_index.pkl")
BM25_CHUNKS_PATH = os.path.join(CHROMA_DIR, f"{COLLECTION_NAME}_bm25_chunks.pkl")

if os.path.exists(BM25_INDEX_PATH) and os.path.exists(BM25_CHUNKS_PATH):
    print("Loading BM25 index from disk...")
    with open(BM25_INDEX_PATH, "rb") as f:
        _bm25_index = pickle.load(f)
    with open(BM25_CHUNKS_PATH, "rb") as f:
        _bm25_chunks = pickle.load(f)
    print(f"BM25 index loaded ({len(_bm25_chunks)} chunks)")
else:
    print("Building BM25 index...")
    _bm25_index, _bm25_chunks = build_bm25_index(_chunks)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(_bm25_index, f)
    with open(BM25_CHUNKS_PATH, "wb") as f:
        pickle.dump(_bm25_chunks, f)
    print(f"BM25 index built and saved ({len(_bm25_chunks)} chunks)")

print("All systems ready!\n")



# Semantic Cache -----------------------------------------------------------------

CACHE_THRESHOLD = 0.92   # cosine similarity for a cache hit
_cache: list = []        # list of {embedding, result, query}


def _cosine_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _check_cache(query_embedding):
    """Return cached result if a similar query exists, else None."""
    for entry in _cache:
        similarity = _cosine_similarity(query_embedding, entry["embedding"])
        if similarity >= CACHE_THRESHOLD:
            print(f"Cache hit! (similarity: {similarity:.3f}) — skipping retrieval")
            return entry["result"]
    return None


def _store_cache(query_embedding, query, result):
    """Store a new query+result in the cache."""
    _cache.append({"embedding": query_embedding, "query": query, "result": result})
    print(f"Cached query (total cached: {len(_cache)})")


# ADK Custom Tool ---------------------------------------------------------------  
 
def search_lighting_knowledge(query: str, room_type_in_german: str = "") -> str:
    """
    Search the TRILUX Beleuchtungspraxis for both general lighting standards 
    AND specific numerical norm tables in a single step.

    Args:
        query: The user's full lighting question.
        room_type_in_german: If the user mentions a specific room/area, provide 
                             its German translation (e.g., "Büro", "Industriehalle"). 
                             Leave empty if no specific room is mentioned.
    """
    # Step 1: Check cache using the main query
    query_embedding = _embed_model.encode([query])[0].tolist()
    cached = _check_cache(query_embedding)
    if cached:
        return cached  

    # Step 2: Broad Search (Finds general rules, prose, and context)
    broad_results = hybrid_search(
        query=query,
        collection=_collection,
        embed_model=_embed_model,
        bm25_index=_bm25_index,
        bm25_chunks=_bm25_chunks,
        top_k=3,
    )

    # Step 3: Targeted Table Search (Forces the DB to find the numerical tables)
    table_search_term = room_type_in_german if room_type_in_german else query
    targeted_query = f"{table_search_term} Beleuchtungsstärke lux Norm Tabelle EN 12464"
    
    table_results = hybrid_search(
        query=targeted_query,
        collection=_collection,
        embed_model=_embed_model,
        bm25_index=_bm25_index,
        bm25_chunks=_bm25_chunks,
        top_k=4,   
    )

    # Step 4: Combine, Deduplicate, and Prioritize Tables
    seen_ids = set()
    combined_results = []
    
    # Force norm tables to the very top of the context window
    for r in table_results:
        if r["id"] not in seen_ids and r.get("content_type") == "norm_table":
            combined_results.append(r)
            seen_ids.add(r["id"])
            
    # Fill the rest with the broad context and any remaining prose from the table search
    for r in broad_results + table_results:
        if r["id"] not in seen_ids:
            combined_results.append(r)
            seen_ids.add(r["id"])
            
    # Keep the top 6 most relevant, unique chunks overall
    final_results = combined_results[:6]

    # Step 5: Format and Return
    confidence = retrieval_confidence(broad_results) # Base confidence on the main query
    confidence_note = {
        "high":   "",
        "medium": "\n Note: Retrieval confidence is moderate.",
        "low":    "\n Note: Retrieval confidence is LOW.",
    }[confidence]

    context = format_context_for_llm(final_results)
    result = f"RETRIEVAL CONFIDENCE: {confidence.upper()}{confidence_note}\n\n{context}"

    # Cache the result
    _store_cache(query_embedding, query, result)

    return result
   

# Memory & Session Services -----------------------------------------------------

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory

session_service = InMemorySessionService()   
memory_service  = InMemoryMemoryService()   

AGENT_INSTRUCTION = """You are an expert lighting consultant with deep knowledge of TRILUX
lighting standards and norms. You help lighting professionals navigate the TRILUX
Beleuchtungspraxis — a comprehensive 580-page technical reference.

### WORKFLOW RULES:
1. ALWAYS call `search_lighting_knowledge` to answer the user's question. You only need to call this ONCE per question.
2. If the user mentions a specific room or application (like "office" or "warehouse"), translate that word to German and pass it to the `room_type_in_german` argument (e.g., "Büro", "Lager").
3. Use `load_memory` ONLY if the user explicitly references past conversations.
4. Base your answer ONLY on the retrieved context. Do not invent norms.
5. ALWAYS cite page numbers when referencing specific values.

### THE CLARIFICATION RULE:
If the user asks about a broad category (e.g., "Industry", "Office") without specifying the exact visual task (e.g., "rough assembly", "data entry"), you must:
- Provide the general requirements and the range of possible values found in the context.
- Ask the user a direct clarifying question to narrow down their specific task so you can provide the exact technical values (lux, UGR, Ra).

### CRITICAL LANGUAGE CONSTRAINT:
The retrieved context from your tools will be in GERMAN. 
However, you MUST detect the language of the USER'S original message and respond 100% in the user's language.
- Do NOT let the German context influence your output language. 
- Translate the technical German terms from the context into the user's language accurately.

### REQUIRED OUTPUT STRUCTURE:
Format every response using the following section headers. 
CRITICAL: Use the English headers if the user asked in English. Use the German headers if the user asked in German.

English Headers:
- Applicable Norms:
- Key Technical Values:
- Practical Recommendations:
- Caveats:
- Next Steps: (Use this section to ask your clarifying question if the user's prompt was too broad).

German Headers:
- Anwendbare Normen:
- Wichtige technische Werte:
- Praktische Empfehlungen:
- Einschränkungen:
- Nächste Schritte: (Use this section to ask your clarifying question if the user's prompt was too broad).
"""

root_agent = LlmAgent(
    name="beleuchtungspraxis_agent",
    model="gemini-2.5-flash",
    instruction=AGENT_INSTRUCTION,
    tools=[
        search_lighting_knowledge,
        load_memory,                # ADK built-in: recall past sessions
    ],
    description=(
        "An AI lighting consultant that answers questions about lighting norms "
        "and standards by retrieving relevant sections from the TRILUX "
        "Beleuchtungspraxis (580-page technical reference)."
    ),
)

