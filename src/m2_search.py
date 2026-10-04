from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text, then split compound-word underscores for BM25."""
    from underthesea import word_tokenize

    return word_tokenize(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25 index from chunk dictionaries."""
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = []
        self.documents = []
        for chunk in chunks:
            tokens = segment_vietnamese(chunk.get("text", "")).split()
            if tokens:
                self.corpus_tokens.append(tokens)
                self.documents.append(chunk)

        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Return positive BM25 matches, ordered by decreasing relevance."""
        if self.bm25 is None or top_k <= 0:
            return []

        query_tokens = segment_vietnamese(query).split()
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.documents[index]
            results.append(
                SearchResult(
                    text=chunk.get("text", ""),
                    score=score,
                    metadata=chunk.get("metadata", {}) or {},
                    method="bm25",
                )
            )
            if len(results) >= top_k:
                break
        return results



class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Encode chunks and replace the target Qdrant collection contents."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload={**(chunk.get("metadata", {}) or {}), "text": texts[index]},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search Qdrant by embedding similarity using query_points()."""
        if top_k <= 0:
            return []
        if hasattr(self.client, "collection_exists") and not self.client.collection_exists(collection):
            return []

        encoded_query = self._get_encoder().encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )
        query_vector = encoded_query.tolist() if hasattr(encoded_query, "tolist") else list(encoded_query)
        response = self.client.query_points(
            collection, query=query_vector, limit=top_k
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(
                SearchResult(
                    text=text,
                    score=float(point.score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results



def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists with RRF: score(d) = sum(1 / (k + rank + 1))."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    fused: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(
                result.text,
                {"score": 0.0, "result": result},
            )
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]



class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
