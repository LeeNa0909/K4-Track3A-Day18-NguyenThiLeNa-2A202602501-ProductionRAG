from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Score each query-document pair and return the best top_k documents."""
        if not documents or top_k <= 0:
            return []

        pairs = [(query, document.get("text", "")) for document in documents]
        scores = self._load_model().predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [scores]
        else:
            scores = list(scores)
        if len(scores) != len(documents):
            raise ValueError(
                f"Cross-Encoder returned {len(scores)} scores for {len(documents)} documents"
            )

        ranked = sorted(
            zip(scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0) or 0.0),
                rerank_score=float(score),
                metadata=document.get("metadata", {}) or {},
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        from flashrank import RerankRequest

        passages = [
            {
                "id": index,
                "text": document.get("text", ""),
                "meta": document.get("metadata", {}) or {},
            }
            for index, document in enumerate(documents)
        ]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
        results = []
        for rank, item in enumerate(ranked[:top_k]):
            index = int(item.get("id", rank))
            document = documents[index]
            results.append(
                RerankResult(
                    text=document.get("text", ""),
                    original_score=float(document.get("score", 0.0) or 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=document.get("metadata", {}) or {},
                    rank=rank,
                )
            )
        return results



def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark reranker latency over n_runs calls, in milliseconds."""
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}



if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
