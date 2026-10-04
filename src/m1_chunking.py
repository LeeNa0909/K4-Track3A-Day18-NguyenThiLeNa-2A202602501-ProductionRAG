from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


@lru_cache(maxsize=1)
def _get_semantic_model():
    """Load the sentence embedding model once, only when semantic chunking is used."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """Group adjacent sentences when their embedding cosine similarity meets threshold."""
    from numpy import dot
    from numpy.linalg import norm

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    metadata = {} if metadata is None else metadata
    # A single sentence has no boundary to compare; avoid loading the model.
    if len(sentences) == 1:
        groups = [sentences]
    else:
        embeddings = _get_semantic_model().encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        )
        groups: list[list[str]] = [[sentences[0]]]
        for index in range(1, len(sentences)):
            previous, current = embeddings[index - 1], embeddings[index]
            similarity = float(dot(previous, current) / (norm(previous) * norm(current) + 1e-9))
            if similarity < threshold:
                groups.append([sentences[index]])
            else:
                groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]

def _split_text_at_boundaries(text: str, max_size: int) -> list[str]:
    """Split text to fit max_size, preferring whitespace and preserving content."""
    if max_size <= 0:
        raise ValueError("Chunk sizes must be positive integers")
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            # Keep boundary whitespace in the preceding piece so concatenation
            # of pieces reproduces the original text exactly.
            boundary = max(
                text.rfind(" ", start + 1, end),
                text.rfind("\n", start + 1, end),
                text.rfind("\t", start + 1, end),
            )
            if boundary >= start:
                end = boundary + 1
        pieces.append(text[start:end])
        start = end
    return pieces


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Create bounded parent chunks and smaller children linked by parent_id."""
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive integers")
    metadata = {} if metadata is None else metadata
    parents: list[Chunk] = []
    children: list[Chunk] = []
    current_parent = ""

    def add_parent(parent_text: str) -> None:
        if not parent_text.strip():
            return
        parent_id = f"parent_{len(parents)}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={
                    **metadata,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "chunk_index": len(parents),
                },
                parent_id=parent_id,
            )
        )
        for child_index, child_text in enumerate(_split_text_at_boundaries(parent_text, child_size)):
            if child_text.strip():
                children.append(
                    Chunk(
                        text=child_text,
                        metadata={
                            **metadata,
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "chunk_index": len(children),
                            "parent_chunk_index": len(parents) - 1,
                            "child_index": child_index,
                        },
                        parent_id=parent_id,
                    )
                )

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    for paragraph in paragraphs:
        # Split long paragraphs, while packing shorter ones together.
        for piece in _split_text_at_boundaries(paragraph, parent_size):
            if current_parent and len(current_parent) + 2 + len(piece) > parent_size:
                add_parent(current_parent)
                current_parent = ""
            if current_parent:
                current_parent += "\n\n" + piece
            else:
                current_parent = piece

            if len(current_parent) >= parent_size:
                add_parent(current_parent)
                current_parent = ""

    if current_parent:
        add_parent(current_parent)
    return parents, children

def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split Markdown at level 1-3 headings and attach each heading as section metadata."""
    metadata = {} if metadata is None else metadata
    chunks: list[Chunk] = []
    section_name = ""
    section_lines: list[str] = []
    in_fenced_block = False
    header_pattern = re.compile(r"^\s{0,3}(#{1,3})\s+(.+?)\s*#*\s*$")
    fence_pattern = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

    def flush_section() -> None:
        content = "".join(section_lines)
        if content.strip():
            chunks.append(
                Chunk(
                    text=content.strip(),
                    metadata={
                        **metadata,
                        "section": section_name,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

    for line in text.splitlines(keepends=True):
        header_match = None if in_fenced_block else header_pattern.match(line.rstrip("\r\n"))
        if header_match:
            flush_section()
            section_name = header_match.group(2).strip()
            section_lines = [line]
        else:
            section_lines.append(line)

        if fence_pattern.match(line):
            in_fenced_block = not in_fenced_block

    flush_section()
    return chunks

def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
