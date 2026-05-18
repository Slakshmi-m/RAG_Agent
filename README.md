# RAG Agent
### AI-Powered Retrieval System for Technical Documents

A hybrid retrieval agent for navigating 500+ pages of technical documentation in German.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF parsing | `pdfplumber` | Table extraction + layout-aware text |
| Embeddings | `sentence-transformers` | Free, local, multilingual (50+ languages) |
| Vector DB | `ChromaDB` | Local, persistent, zero infrastructure |
| Keyword search | `rank-bm25` | Exact matching for norm codes |
| Retrieval fusion | Reciprocal Rank Fusion | Standard hybrid RAG merge strategy |
| Agent framework | Google ADK | Native Gemini integration, built-in dev UI |
| LLM | Gemini 2.5 Flash | Free tier, strong German, large context window |

---

## Architecture

The focus here is on **high-quality data ingestion** and **hybrid retrieval** rather than a large multi-agent framework. Accuracy comes from getting the chunking and retrieval right before the LLM ever sees the data.

### 1. Table-Aware Chunking (`chunker.py`)

Standard text chunkers destroy tables. A document like *Beleuchtungspraxis* contains norm tables such as *Tabelle 4.1 - Lichttechnische Anforderungen*, which maps room types to Ēm (lux), Uo, Ra, and RUGR values. A naive sliding-window chunker splits these mid-row, breaking the room → value association and making the values meaningless to the LLM.

`pdfplumber` detects pages with tables, extracts them as intact Markdown, and tags them with a `norm_table` metadata label. Regular prose is chunked with a sliding-window approach (500 chars, 80 overlap) that respects sentence boundaries to avoid slicing norm identifiers like `EN 12464-1` in half.

```
Page detected
    │
    ├── Has tables? → Extract as intact Markdown → tag: norm_table
    └── Prose text → Sliding-window chunks      → tag: prose
```

### 2. Hybrid Search with RRF (`retriever.py`)

Pure semantic search struggles with exact string matches - specific lux values or DIN norm codes, for example. The solution is a custom **hybrid search** combining dense (semantic) and sparse (BM25) retrieval, merged via **Reciprocal Rank Fusion**:

```
User Query
    │
    ├── Semantic Search (ChromaDB)    → finds chunks by intent
    └── BM25 Keyword Search           → finds chunks by exact norm codes
              │
              ▼
    Reciprocal Rank Fusion
    score = 1/(rank_semantic + 60) + 1/(rank_bm25 + 60)
              │
              ▼
    Top-K reranked results
```

A custom regex tokenizer (`[\w][\w\-\.]*`) ensures technical identifiers like `EN-12464-1` and `ASR-A3.4` aren't split at punctuation boundaries by BM25.

| Method | Strength | Weakness |
|---|---|---|
| Semantic | Understands intent, cross-language | May miss exact norm codes |
| BM25 | Exact string matching | Misses synonyms, no cross-language |
| **Hybrid + RRF** | **Best of both** | Slight added complexity |

### 3. Agent Design (`agent.py`)

The agent uses `gemini-2.5-flash` via Google ADK's function calling.

**Unified tool:** To prevent quota exhaustion and reduce latency, broad semantic search and targeted norm table search are combined into a **single tool** (`search_lighting_knowledge`). This brings API calls per query down from 3-4 to 2.

**Semantic cache:** Frequently asked queries bypass retrieval entirely via a cosine-similarity cache (threshold: 0.92). Two queries above this threshold share the same cached result — so `"office lighting standards"` and `"Bürobeleuchtung Normen"` hit the same entry.

```
Query → embed → check cache
    ├── HIT  (sim ≥ 0.92) → return instantly, zero tokens
    └── MISS → hybrid retrieval → store in cache → return
```

**Memory:** Uses ADK's `InMemorySessionService` for within-session conversation history and `InMemoryMemoryService` for cross-session recall via the built-in `load_memory` tool.

---

## Edge Cases

### Cross-lingual bleed
When a user asks in English, the LLM tends to anchor to the German context and reply in German - even with explicit instructions. A strict **language lock directive** in the system prompt forces the model to translate technical terms on the fly and respond entirely in the user's language.

### Vague queries
A broad question like *"What are the rules for an office?"* is unanswerable precisely, DIN EN 12464-1 specifies different lux values for writing (500 lx), technical drawing (750 lx), and CAD work (500 lx). A **clarification rule** in the system prompt has the agent give the full range of applicable values, then ask the user to specify their exact visual task.

### API quota multiplication
Multi-tool agents multiply API usage. Each tool call triggers a separate LLM reasoning step. With separate tools, one user question was generating 3-4 API calls, exhausting a 20 req/day free tier after 5-6 questions. Merging all retrieval into one unified tool fixed this.

---

## Known Limitations

### Double-extraction ghost
`pdfplumber`'s `extract_text()` pulls all text from a page, including table cells. Pages with tables therefore end up with two representations in the vector DB: a clean Markdown table and a garbled string of the same data. RRF naturally ranks the clean version higher due to its structured vocabulary, so this rarely surfaces in practice, but the proper fix is to subtract table bounding boxes from prose extraction using `pdfplumber`'s bbox API.

### In-memory cache only
The semantic cache lives in a Python list in RAM and resets on every restart. The first query after a restart always runs full retrieval.

---

## What Works Well

**Accuracy and citation:** Hybrid retrieval reliably surfaces exact numerical values (lux levels, Ra, RUGR, Uo), and the agent cites source page numbers — every claim is traceable back to a specific page.

**Structured output:** The agent consistently follows a 5-part format regardless of query type: applicable norms → key technical values → practical recommendations → caveat → next steps.

**Cross-language retrieval:** The multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) maps English and German into the same vector space - `"office lighting"` retrieves `"Bürobeleuchtung"` chunks without any translation step.

**Hybrid precision:** BM25 catches exact norm codes like `EN 12464-1` that semantic search alone would miss.

---

## Potential Improvements

- **Cross-encoder re-ranking:** A `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage filter over RRF results for higher precision
- **Persistent cache:** Move the in-memory semantic cache to Redis or SQLite for cross-session and cross-instance sharing
- **Table bbox filtering:** Filter table bounding boxes from prose extraction to eliminate the double-extraction issue
- **Managed vector DB:** Replace local ChromaDB with Pinecone or Weaviate if scaling to many concurrent users
- **Evaluation dataset:** Build 20–30 labelled Q&A pairs to measure retrieval recall objectively
