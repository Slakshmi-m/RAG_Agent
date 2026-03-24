# Lighting practice RAG Agent
### AI-Powered Lighting Standards Retrieval System

An AI-powered agent that helps lighting professionals navigate the 580-page TRILUX *Beleuchtungspraxis* to identify and apply relevant lighting norms and standards for their specific scenarios.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd Lighting_Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your Gemini API key
cp agent/.env.example agent/.env
# Edit agent/.env and set GEMINI_API_KEY=your_key_here

# 3. Place the PDF in the project root
# Request from: https://www.trilux.com/de/beleuchtungspraxis/

# 4. Run the agent
adk run agent        # CLI chat
adk web              # Browser UI at http://localhost:8000
```

> **First run** builds the vector index (~5 min). Every subsequent run loads from disk instantly.

---

##  Project Structure

```
Lighting_Agent/
├── agent/
│   ├── agent.py          # ADK agent — tools, cache, memory, language detection
│   ├── __init__.py
│   └── .env              # GEMINI_API_KEY (not committed)
├── chunker.py            # Table-aware PDF chunking
├── retriever.py          # Hybrid BM25 + semantic search with RRF
├── requirements.txt
├── README.md
├── Beleuchtungspraxis.pdf       # Source document (request from TRILUX)
└── chroma_db/                   # Auto-created: vector index + BM25 pickle
```

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
| LLM | Gemini 2.5 Flash Lite | Free tier, strong German, large context window |


## System Architecture & Approach

My goal was to build a pragmatic, highly accurate system. Rather than over-engineering a massive multi-agent framework, I focused on **high-quality data ingestion** and **hybrid retrieval**.

### 1. Data Processing: Table-Aware Chunking (`chunker.py`)

Standard text chunkers destroy the rows and columns of technical tables.

**The Problem:** The Beleuchtungspraxis contains critical norm tables like *Tabelle 4.1 — Lichttechnische Anforderungen* with columns for room type, Ēm (lux), Uo, Ra, and RUGR. A naive sliding-window chunker splits these mid-row, breaking the room -> value association and making the values meaningless to the LLM.

**The Solution:** I used `pdfplumber` to detect pages with tables, extract them, and format them as intact Markdown tables. These are tagged with a `norm_table` metadata label so the LLM recognises them as structured data.

**Prose Handling:** Regular text is processed using a sliding-window chunker (500 chars, 80 overlap) that preserves sentence boundaries to prevent slicing important norm identifiers like `EN 12464-1` in half.

```
Page detected
    │
    ├── Has tables? → Extract as intact Markdown → tag: norm_table
    └── Prose text → Sliding-window chunks      → tag: prose
```

### 2. Retrieval: Hybrid Search with RRF (`retriever.py`)

Pure semantic search struggles with exact string matches (e.g. specific lux values or DIN norm codes).

**The Solution:** I implemented a custom **Hybrid Search** combining Dense (Semantic) and Sparse (BM25) retrieval, merged via **Reciprocal Rank Fusion (RRF)**:

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

**Custom Tokenizer:** I wrote a custom regex tokenizer for BM25 (`[\w][\w\-\.]*`) to ensure technical identifiers like `EN-12464-1` and `ASR-A3.4` are not split apart by standard punctuation rules.

**Why hybrid beats either alone:**

| Method | Strength | Weakness |
|---|---|---|
| Semantic | Understands intent, cross-language | May miss exact norm codes |
| BM25 | Exact string matching | Misses synonyms, no cross-language |
| **Hybrid + RRF** | **Best of both** | Slight added complexity |

### 3. Agent Design (`agent.py`)

The agent uses `gemini-2.5-flash` via Google ADK's function calling to query the retrieved context.

**Unified Tooling:** To prevent quota exhaustion and reduce latency, I combined broad semantic search and targeted norm table search into a **single unified tool** (`search_lighting_knowledge`). This reduces API calls from 3-4 per query down to 2.

**Semantic Caching:** Frequently asked queries bypass retrieval entirely via a **cosine-similarity cache** (threshold: 0.92). Two queries with similarity above this threshold share the same cached result - so `"office lighting standards"` and `"Bürobeleuchtung Normen"` hit the same cache entry.

```
Query → embed → check cache
    ├── HIT  (sim ≥ 0.92) → return instantly, zero tokens 
    └── MISS → hybrid retrieval → store in cache → return
```

**Memory:** Uses ADK's `InMemorySessionService` for within-session conversation history and `InMemoryMemoryService` for cross-session recall via the built-in `load_memory` tool.

---

## Handling Edge Cases

During development, I encountered and solved several complex RAG edge cases:

### Cross-Lingual Bleed
**Problem:** The user asks in English, but the retrieved chunks are in German. The LLM anchors to the German context and replies in German - even when explicitly instructed otherwise.

**Solution:**  A strict **Language Lock directive** is given in the system prompt, forcing the LLM to translate technical terms dynamically and respond entirely in the user's native language.

### Vague User Prompts
**Problem:** If a user asks a broad question like *"What are the rules for an office?"*, the agent cannot know which specific visual task to cite — DIN EN 12464-1 specifies different values for writing (500 lx), technical drawing (750 lx), and CAD work (500 lx).

**Solution:** A **Clarification Rule** in the system prompt: the agent provides the general range of applicable values but ends by asking the user to specify their exact visual task (e.g. *"Are you doing data entry, reading, or technical drawing?"*).

### API Quota Multiplier
**Problem:** Agentic systems with multiple tools multiply API usage. Each tool call requires the LLM to process the result and decide next steps - counting as a separate API request. With separate tools, one user question triggered 3-4 API calls, exhausting a 20 req/day free tier after only 5-6 questions.

**Solution:** Merged all retrieval into a single unified tool, reducing every query to exactly 2 API calls (one to invoke the tool, one to generate the answer).

---

## Evaluation & Limitations

### What Works Well 

**Accuracy & Citation:** The hybrid retrieval effectively surfaces exact numerical values (lux levels, Ra, RUGR, Uo), and the agent reliably cites the page numbers of its sources - every claim is traceable back to a specific page in the Beleuchtungspraxis.

**Formatting:** The agent strictly adheres to a highly readable 5-part structure regardless of query type:
- Applicable Norms
- Key Technical Values
- Practical Recommendations
- Caveat
- Next Steps (Used to ask clarifying question if the user's prompt was too broad)

**Cross-language Retrieval:** The multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) maps English and German queries into the same vector space — `"office lighting"` retrieves `"Bürobeleuchtung"` chunks without any translation step.

**Semantic Cache:** Repeat or semantically similar queries bypass retrieval entirely via a cosine-similarity cache (threshold 0.92) — instant response, zero tokens consumed.

**Hybrid Precision:** BM25 catches exact norm codes like EN 12464-1 that semantic search alone would miss. Together they consistently outperform either method alone.

### What Doesn't Work Perfectly Yet 

**The "Double-Extraction Ghost"**

Because `pdfplumber` extracts all text from a page via `extract_text()`, pages containing tables result in **duplicate data** in the vector database:
- One clean Markdown table (extracted specifically via `extract_tables()`)
- One garbled string of the same data (from the general page text extraction)

**Trade-off Decision:** Filtering out table bounding boxes from prose extraction requires complex coordinate math using `pdfplumber`'s bbox API. Given the time constraints of this task, I opted not to build this filter. The Hybrid Retrieval's RRF scoring naturally mitigates this - the clean Markdown table consistently ranks higher than the garbled prose duplicate due to its structured vocabulary.

**In-Memory Cache Only:** The semantic cache resets on every agent restart since it lives in a plain Python list in RAM. Warm cache state is lost between sessions — the first query after restart always runs full retrieval.

---

## Future Improvements

- **Cross-encoder re-ranking:** Add a `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage filter over RRF results for higher precision
- **Managed vector database:** Replace local ChromaDB with Pinecone or Weaviate if scaling to thousands of concurrent users
- **Persistent cache:** Move the in-memory semantic cache to Redis or SQLite for cross-session and cross-instance cache sharing
- **Evaluation dataset:** Build 20-30 labelled Q&A pairs to measure retrieval recall objectively — currently quality is assessed manually

---

