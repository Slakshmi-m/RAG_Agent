"""
chunker.py — Table-Aware PDF Chunking for Beleuchtungspraxis

The Beleuchtungspraxis contains two fundamentally different content types:

  1. PROSE: Running explanatory text (regular paragraphs, bullet points)
     → Handle with sliding-window character chunking

  2. NORM TABLES: Structured tables like "Tabelle 4.1 — Lichttechnische
     Anforderungen" with columns for room type, Ēm (lux), Uo, Ra, RUGR, etc.
     → These MUST be kept intact. Splitting a table row across two chunks
       loses the association between room name and its required values.

Strategy:
  - Use pdfplumber to detect pages that contain tables
  - For table pages: extract via pdfplumber.extract_tables(), format as
    Markdown, and store as a single chunk (never split)
  - For prose pages: use the standard sliding-window chunker
  - Tag every chunk with its content type so the LLM prompt can signal
    when it's citing a structured norm table vs. prose explanation
"""

import re
from typing import List, Dict, Optional
import pdfplumber


# Constants 

PROSE_CHUNK_SIZE = 500      # characters per prose chunk
PROSE_CHUNK_OVERLAP = 80    # overlap between consecutive prose chunks
MIN_CHUNK_LENGTH = 80       # ignore chunks shorter than this (page numbers etc.)

# Table detection: a page is "table-heavy" if pdfplumber finds at least this
# many rows across all extracted tables on the page.
TABLE_ROW_THRESHOLD = 3


# Section Heading Detection 

def detect_section_heading(text: str) -> str:
    """
    Extract the first section heading from a text block.

    Matches patterns found in Beleuchtungspraxis:
      - Numbered: "4.1.5 Lichttechnische Anforderungen"
      - ALL CAPS: "INHALTSVERZEICHNIS", "VORWORT"

    Returns the heading string (max 100 chars), or "" if none found.
    """
    lines = text.split("\n")[:6]
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+\.\d*\s+\w", line):
            return line[:100]
        if line.isupper() and 4 < len(line) < 60:
            return line
    return ""


# Table Formatting 

def format_table_as_markdown(table: List[List[Optional[str]]], table_index: int = 0) -> str:
    """
    Convert a pdfplumber table (list of rows) into a Markdown table string.

    pdfplumber returns tables as List[List[str|None]].
    None cells (merged cells, empty cells) are replaced with "-".

    Example output:
        | Ref.-Nr. | Art des Raumes | Ēm (lx) | Uo | Ra | RUGR |
        |---|---|---|---|---|---|
        | 1.1 | Korridore und Verkehrsflächen | 100 | 0.40 | 40 | 28 |
        | 1.2 | Treppen, Rolltreppen | 100 | 0.40 | 40 | 25 |
    """
    if not table or not table[0]:
        return ""

    # Normalise cells: strip whitespace, replace None/empty with "-"
    def clean(cell) -> str:
        if cell is None:
            return "-"
        cleaned = str(cell).strip().replace("\n", " ")
        return cleaned if cleaned else "-"

    rows = [[clean(cell) for cell in row] for row in table]

    # Determine column widths for alignment
    n_cols = max(len(row) for row in rows)

    # Pad rows to same width
    rows = [row + ["-"] * (n_cols - len(row)) for row in rows]

    lines = []

    # Header row (first row of table)
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * n_cols) + "|")

    # Data rows
    for row in rows[1:]:
        # Skip completely empty rows (artefacts from merged cells)
        if all(c == "-" for c in row):
            continue
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# Main Chunking Functions 

def chunk_page_prose(page_text: str, page_num: int, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Sliding-window chunker that respects word boundaries.
    """
    chunks = []
    text = page_text.strip()
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        
        # If we aren't at the end of the text, pull back to the nearest space
        # so we don't slice a word (like "Beleuchtungsstärke") in half.
        if end < len(text):
            while end > start and text[end] not in [' ', '\n']:
                end -= 1
            if end == start: # Fallback if there's an absurdly long string
                end = start + chunk_size 
                
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": chunk_text,
                "page_num": page_num,
                "section": detect_section_heading(chunk_text),
                "content_type": "prose",
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1

        # Advance start, but account for overlap
        start = end - overlap

    return chunks


def chunk_page_tables(page, page_num: int) -> List[Dict]:
    """
    Extract and format all tables from a pdfplumber page object.
    Each table becomes one chunk — never split.

    Also extracts any surrounding prose text (before/after tables)
    as separate prose chunks.

    Returns list of chunk dicts.
    """
    chunks = []

    # Extract all tables on this page
    tables = page.extract_tables()
    if not tables:
        return []

    for i, table in enumerate(tables):
        if not table:
            continue

        # Count non-empty rows to filter out junk (single-cell "tables")
        real_rows = [r for r in table if any(c and str(c).strip() for c in r)]
        if len(real_rows) < TABLE_ROW_THRESHOLD:
            continue

        md_table = format_table_as_markdown(table, table_index=i)
        if not md_table:
            continue

        chunks.append({
            "text": md_table,
            "page_num": page_num,
            "section": f"Table {i+1} on page {page_num}",
            "content_type": "norm_table",   # ← tagged as structured data
            "chunk_idx": i,
        })

    return chunks


def extract_and_chunk(
    pdf_path: str,
    prose_chunk_size: int = PROSE_CHUNK_SIZE,
    prose_overlap: int = PROSE_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Main entry point. Processes the full PDF with table-aware chunking.

    For each page:
      1. Try to extract tables with pdfplumber
      2. If tables found → store each as an intact Markdown chunk
      3. Extract prose text → sliding-window chunk it
      4. Tag everything with page number, section, content_type

    Returns:
        List of chunk dicts, each with keys:
          id, text, page_num, section, content_type
    """
    all_chunks = []
    chunk_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"📄 Processing {total} pages with table-aware chunker...")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Extract tables
            table_chunks = chunk_page_tables(page, page_num)

            # Extract prose text 
            prose_text = page.extract_text()
            prose_chunks = []

            if prose_text and len(prose_text.strip()) > MIN_CHUNK_LENGTH:
                prose_chunks = chunk_page_prose(
                    prose_text, page_num, prose_chunk_size, prose_overlap
                )

            # Assign global IDs and collect
            # Tables first (they're more specific/valuable), then prose
            for chunk in table_chunks + prose_chunks:
                chunk["id"] = f"chunk_{chunk_counter:05d}"
                all_chunks.append(chunk)
                chunk_counter += 1

            if (page_num) % 100 == 0:
                print(f"  ... {page_num}/{total} pages | {len(all_chunks)} chunks so far")

    # Summary stats
    n_tables = sum(1 for c in all_chunks if c["content_type"] == "norm_table")
    n_prose = sum(1 for c in all_chunks if c["content_type"] == "prose")
    print(f"\n Chunking complete!")
    print(f"   Total chunks : {len(all_chunks)}")
    print(f"   Norm tables  : {n_tables}  ← kept fully intact")
    print(f"   Prose chunks : {n_prose}")

    return all_chunks

