from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_SET_PATH,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _empty_evaluation(questions, answers, contexts, ground_truths) -> dict:
    """Return zero scores while retaining rows for downstream diagnosis."""
    per_question = [
        EvalResult(
            question=question,
            answer=answer,
            contexts=list(context_list),
            ground_truth=ground_truth,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
        for question, answer, context_list, ground_truth in zip(
            questions, answers, contexts, ground_truths
        )
    ]
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": per_question,
    }


def _finite_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate RAG responses with RAGAS and OpenRouter-backed chat/embeddings."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    if not questions:
        return _empty_evaluation(questions, answers, contexts, ground_truths)
    if not OPENROUTER_API_KEY:
        print("  RAGAS skipped: set OPENROUTER_API_KEY in .env to enable evaluation.")
        return _empty_evaluation(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        default_headers = {"X-OpenRouter-Title": OPENROUTER_APP_NAME}
        if OPENROUTER_APP_URL:
            default_headers["HTTP-Referer"] = OPENROUTER_APP_URL
        llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers,
            temperature=0,
        )
        embeddings = OpenAIEmbeddings(
            model=OPENROUTER_EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers,
        )
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
        frame = result.to_pandas()
        metric_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        per_question = []
        for index, (question, answer, context_list, ground_truth) in enumerate(
            zip(questions, answers, contexts, ground_truths)
        ):
            row = frame.iloc[index] if index < len(frame) else {}
            values = {
                name: _finite_score(row.get(name, 0.0))
                for name in metric_names
            }
            per_question.append(
                EvalResult(
                    question=question,
                    answer=answer,
                    contexts=list(context_list),
                    ground_truth=ground_truth,
                    **values,
                )
            )

        aggregate = {
            name: (
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in metric_names
        }
        return {**aggregate, "per_question": per_question}
    except Exception as exc:
        print(f"  RAGAS evaluation failed: {exc}")
        return _empty_evaluation(questions, answers, contexts, ground_truths)



def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Rank the weakest questions and diagnose their lowest-scoring metric."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời có thông tin không được hỗ trợ bởi ngữ cảnh.",
            "Thắt chặt system prompt và đặt temperature về 0.",
        ),
        "context_recall": (
            "Retriever bỏ sót đoạn văn cần thiết để trả lời.",
            "Cải thiện chunking hoặc bổ sung từ khóa cho BM25.",
        ),
        "context_precision": (
            "Các đoạn không liên quan được xếp quá cao.",
            "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata.",
        ),
        "answer_relevancy": (
            "Câu trả lời lệch trọng tâm câu hỏi.",
            "Chỉnh prompt để mô hình trả lời trực tiếp câu hỏi.",
        ),
    }
    metric_names = tuple(diagnostic_tree)
    ranked = []
    for result in eval_results:
        scores = {
            name: _finite_score(getattr(result, name, 0.0))
            for name in metric_names
        }
        average_score = sum(scores.values()) / len(metric_names)
        worst_metric = min(metric_names, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append(
            {
                "question": result.question,
                "score": average_score,
                "average_score": average_score,
                "worst_metric": worst_metric,
                "worst_score": scores[worst_metric],
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    ranked.sort(key=lambda item: item["score"])
    return ranked[:bottom_n]



def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save aggregate scores, per-question evidence, and weakest cases as JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    per_question = []
    for item in results.get("per_question", []):
        if hasattr(item, "__dataclass_fields__"):
            per_question.append(asdict(item))
        elif isinstance(item, dict):
            per_question.append(item)
        else:
            per_question.append(
                {
                    name: getattr(item, name)
                    for name in (
                        "question", "answer", "contexts", "ground_truth",
                        "faithfulness", "answer_relevancy", "context_precision",
                        "context_recall",
                    )
                    if hasattr(item, name)
                }
            )
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": per_question,
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
