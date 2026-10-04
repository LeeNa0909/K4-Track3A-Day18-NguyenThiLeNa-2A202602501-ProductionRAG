"""Text enrichment helpers for the RAG ingestion pipeline."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, create_openrouter_client


@dataclass
class EnrichedChunk:
    """A source chunk plus its retrieval-oriented enrichments."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def summarize_chunk(text: str) -> str:
    """Return a short extractive summary that does not require an API."""
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text.strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Create answerable question-shaped retrieval hints from the chunk."""
    if n_questions <= 0:
        return []
    sentences = [sentence for sentence in _sentences(text) if len(sentence) > 10]
    return [f"{sentence.rstrip('.!?')}?" for sentence in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Attach the source title to the chunk while preserving its full text."""
    title = str(document_title).strip()
    return f"Source document: {title}\n\n{text}" if title else text


def extract_metadata(text: str) -> dict:
    """Provide lightweight metadata without requiring a model call."""
    sentences = _sentences(text)
    return {
        "topic": sentences[0][:160] if sentences else "general",
        "entities": [],
        "category": "general",
        "language": "vi",
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Use one OpenRouter request for summary, HyQA, context, and metadata.

    If no key is configured or the provider returns an unusable response, return
    local fallbacks so document indexing can continue without the LLM.
    """
    fallback_context = (
        f"This passage is from the document '{source}'."
        if source
        else "Relevant passage from the source document."
    )
    fallback = {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": fallback_context,
        "metadata": extract_metadata(text),
    }
    if not OPENROUTER_API_KEY:
        return fallback

    try:
        client = create_openrouter_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the supplied passage and return only a valid JSON object with exactly "
                        "these fields: summary (concise, in the passage's language), questions (an array "
                        "of 1-3 useful questions answerable from the passage), context (one short sentence "
                        "describing the passage's subject and location in its source document), and metadata "
                        "(an object with topic, entities, category, and language). Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Source document: {source or 'unknown'}\n\nPassage:\n{text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )
        content = response.choices[0].message.content or "{}"
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content, flags=re.IGNORECASE)
        result = json.loads(content)
        if not isinstance(result, dict):
            return fallback

        summary = result.get("summary")
        questions = result.get("questions")
        context = result.get("context")
        metadata = result.get("metadata")
        return {
            "summary": summary.strip() if isinstance(summary, str) and summary.strip() else fallback["summary"],
            "questions": [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            if isinstance(questions, list)
            else fallback["questions"],
            "context": context.strip() if isinstance(context, str) and context.strip() else fallback["context"],
            "metadata": metadata if isinstance(metadata, dict) else fallback["metadata"],
        }
    except Exception as exc:
        # Do not include request details or credentials in the log.
        print(f"  Enrichment API unavailable; using local fallback ({type(exc).__name__}).")
        return fallback


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks using one combined request by default, or local helpers.

    Each input chunk has ``text`` and optionally ``metadata``. Supported
    methods are ``combined``, ``summary``, ``hyqa``, ``contextual``, and
    ``metadata``.
    """
    methods = ["combined"] if methods is None else methods
    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        original_metadata = chunk.get("metadata", {}) or {}
        source = str(original_metadata.get("source", "") or "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result["summary"]
            questions = result["questions"]
            context = result["context"]
            parts = [part for part in (context, summary, text) if part]
            if questions:
                parts.append("Related questions: " + " ".join(questions))
            enriched_text = "\n\n".join(parts)
            auto_metadata = result["metadata"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**original_metadata, **auto_metadata},
                method="+".join(methods),
            )
        )

        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample_text = "Employees receive annual leave based on their service period. Additional leave is granted for long tenure."
    print("Summary:", summarize_chunk(sample_text))
    print("Questions:", generate_hypothesis_questions(sample_text))
    print("Contextual:", contextual_prepend(sample_text, "Employee Handbook 2024"))
    print("Metadata:", extract_metadata(sample_text))
