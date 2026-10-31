
import json
import time
import statistics
from pathlib import Path

from src.retrieval.reranker import (
    load_models,
    connect_database,
    retrieve_candidates,
    rerank_documents,
)
from src.generation.llm import generate_answer


# ==========================================================
# 1. EVALUATION CONFIGURATION
# ==========================================================

TEST_CASES = [
    {
        "question": "What is the difference between a public and private subnet?",
        "relevant_pages": [
            ("sample.pdf", 33),
            ("sample.pdf", 42),
            ("sample.pdf", 43),
        ],
    },
    {
        "question": "What is the purpose of a NAT Gateway?",
        "relevant_pages": [
            ("sample.pdf", 43),
            ("sample.pdf", 42),
            ("sample.pdf", 33),
        ],
    },
]

TOP_K = 5

OUTPUT_FILE = Path(
    "data/processed/evaluation_results.json"
)


# ==========================================================
# 2. RETRIEVAL METRICS
# ==========================================================

def get_page_reference(item):

    metadata = item.get("metadata", {})

    return (
        metadata.get("filename"),
        metadata.get("page_number"),
    )


def precision_at_k(results, relevant_pages, k=5):

    retrieved = [
        get_page_reference(item)
        for item in results[:k]
    ]

    relevant = set(relevant_pages)

    hits = sum(
        1 for page in retrieved
        if page in relevant
    )

    return hits / k if k else 0.0


def recall_at_k(results, relevant_pages, k=5):

    retrieved = [
        get_page_reference(item)
        for item in results[:k]
    ]

    relevant = set(relevant_pages)

    if not relevant:
        return 0.0

    hits = sum(
        1 for page in retrieved
        if page in relevant
    )

    return hits / len(relevant)


def reciprocal_rank(results, relevant_pages):

    relevant = set(relevant_pages)

    for rank, item in enumerate(results, start=1):

        if get_page_reference(item) in relevant:
            return 1.0 / rank

    return 0.0


# ==========================================================
# 3. CITATION REFERENCE METRIC
# ==========================================================

def citation_reference_accuracy(results, relevant_pages):

    """
    Measures how many retrieved source references match
    the manually verified relevant PDF/page references.

    This is retrieval source-reference accuracy,
    not claim-level citation entailment.
    """

    if not results:
        return 0.0

    relevant = set(relevant_pages)

    correct = sum(
        1 for item in results
        if get_page_reference(item) in relevant
    )

    return correct / len(results)


# ==========================================================
# 4. LATENCY METRIC
# ==========================================================

def percentile_95(values):

    if not values:
        return 0.0

    ordered = sorted(values)

    index = max(
        0,
        int(0.95 * len(ordered) + 0.999999) - 1
    )

    return ordered[
        min(index, len(ordered) - 1)
    ]


# ==========================================================
# 5. MAIN EVALUATION
# ==========================================================

def main():

    print("\n========== RAG EVALUATION ==========\n")

    print("Loading models...")

    embedding_model, reranker = load_models()

    collection = connect_database()

    evaluation_results = []

    total_latencies = []

    for number, test in enumerate(
        TEST_CASES,
        start=1
    ):

        question = test["question"]

        relevant_pages = test["relevant_pages"]

        print("\n" + "=" * 60)

        print(
            f"TEST {number}/{len(TEST_CASES)}"
        )

        print("Question:", question)

        total_start = time.perf_counter()

        # --------------------------------------------------
        # 1. RETRIEVAL
        # --------------------------------------------------

        retrieval_start = time.perf_counter()

        candidates = retrieve_candidates(
            question,
            collection,
            embedding_model
        )

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        # --------------------------------------------------
        # 2. RERANKING
        # --------------------------------------------------

        reranking_start = time.perf_counter()

        results = rerank_documents(
            question,
            candidates,
            reranker,
            top_k=TOP_K
        )

        reranking_time = (
            time.perf_counter() - reranking_start
        )

        # --------------------------------------------------
        # 3. CONTEXT PREPARATION
        # --------------------------------------------------

        context_parts = []

        for item in results:

            metadata = item["metadata"]

            context_parts.append(
                f"PDF: {metadata.get('filename')}\n"
                f"PAGE: {metadata.get('page_number')}\n"
                f"CONTENT: {item['text']}"
            )

        context = "\n\n".join(context_parts)

        # --------------------------------------------------
        # 4. ANSWER GENERATION
        # --------------------------------------------------

        generation_start = time.perf_counter()

        answer = generate_answer(
            question,
            context
        )

        generation_time = (
            time.perf_counter() - generation_start
        )

        total_time = (
            time.perf_counter() - total_start
        )

        total_latencies.append(total_time)

        # --------------------------------------------------
        # 5. EVALUATION METRICS
        # --------------------------------------------------

        precision = precision_at_k(
            results,
            relevant_pages,
            TOP_K
        )

        recall = recall_at_k(
            results,
            relevant_pages,
            TOP_K
        )

        mrr = reciprocal_rank(
            results,
            relevant_pages
        )

        citation_reference = citation_reference_accuracy(
            results,
            relevant_pages
        )

        row = {

            "question": question,

            "answer": answer,

            "retrieved_sources": [
                {
                    "filename": item["metadata"].get(
                        "filename"
                    ),

                    "page_number": item["metadata"].get(
                        "page_number"
                    ),

                    "rerank_score": float(
                        item["rerank_score"]
                    ),

                    "text": item["text"],
                }

                for item in results
            ],

            "precision_at_5": precision,

            "recall_at_5": recall,

            "mrr": mrr,

            "citation_reference_accuracy": citation_reference,

            "retrieval_seconds": retrieval_time,

            "reranking_seconds": reranking_time,

            "generation_seconds": generation_time,

            "total_seconds": total_time,
        }

        evaluation_results.append(row)

        # --------------------------------------------------
        # 6. DISPLAY RESULTS
        # --------------------------------------------------

        print("\nGenerated Answer:")
        print(answer)

        print("\nRetrieval Evaluation:")

        print(f"Precision@5: {precision:.4f}")

        print(f"Recall@5: {recall:.4f}")

        print(f"MRR: {mrr:.4f}")

        print(
            "Citation Reference Accuracy: "
            f"{citation_reference:.4f}"
        )

        print("\nLatency:")

        print(f"Retrieval: {retrieval_time:.2f} sec")

        print(f"Reranking: {reranking_time:.2f} sec")

        print(f"Generation: {generation_time:.2f} sec")

        print(f"Total: {total_time:.2f} sec")

    # ======================================================
    # 7. AGGREGATE EVALUATION
    # ======================================================

    print("\n========== FINAL EVALUATION ==========")

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "citation_reference_accuracy",
        "retrieval_seconds",
        "reranking_seconds",
        "generation_seconds",
        "total_seconds",
    ]

    summary = {}

    for name in metric_names:

        values = [
            row[name]
            for row in evaluation_results
        ]

        average = (
            statistics.mean(values)
            if values else 0.0
        )

        summary[f"average_{name}"] = average

        print(
            f"Average {name}: {average:.4f}"
        )

    p95 = percentile_95(total_latencies)

    summary["p95_latency_seconds"] = p95

    summary["questions_evaluated"] = len(
        evaluation_results
    )

    print(
        f"\np95 Latency: {p95:.4f} seconds"
    )

    # ======================================================
    # 8. SAVE RESULTS
    # ======================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "summary": summary,

        "individual_results": evaluation_results,

        "evaluation_method": (
            "Precision@5, Recall@5 and MRR use manually "
            "verified relevant PDF/page labels. Citation "
            "reference accuracy measures retrieved-source "
            "matching, not claim-level citation entailment."
        ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        f"\nEvaluation report saved: {OUTPUT_FILE}"
    )

    print("\nRAG evaluation completed successfully!")


if __name__ == "__main__":

    main()
