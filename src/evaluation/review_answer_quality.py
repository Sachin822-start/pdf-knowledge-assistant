
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RESULTS_FILE = Path("data/processed/evaluation_results.json")
OUTPUT_FILE = Path("data/processed/answer_quality_results.json")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Initial heuristic threshold. Validate before reporting as a benchmark.
SIMILARITY_THRESHOLD = 0.65


def split_sentences(text):
    """Split generated answers into individual sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def evaluate_answer(answer, sources, model):
    sentences = split_sentences(answer)

    if not sentences or not sources:
        return {
            "faithfulness_proxy": None,
            "hallucination_proxy": None,
            "sentence_results": [],
        }

    source_texts = [source["text"] for source in sources]

    # Generate normalized embeddings
    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    source_embeddings = model.encode(
        source_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sentence_results = []
    supported_count = 0

    for sentence, embedding in zip(sentences, sentence_embeddings):

        # Compare each answer sentence with all retrieved passages
        similarities = np.dot(source_embeddings, embedding)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        supported = best_score >= SIMILARITY_THRESHOLD

        if supported:
            supported_count += 1

        sentence_results.append({
            "sentence": sentence,
            "best_similarity": round(best_score, 4),
            "support_detected": supported,
            "supporting_pdf": sources[best_index].get("filename"),
            "supporting_page": sources[best_index].get("page_number"),
        })

    total = len(sentences)
    unsupported_count = total - supported_count

    return {
        "total_sentences": total,
        "supported_sentences": supported_count,
        "unsupported_sentences": unsupported_count,
        "faithfulness_proxy": supported_count / total,
        "hallucination_proxy": unsupported_count / total,
        "sentence_results": sentence_results,
    }


def check_citations(answer, sources):
    """
    Check explicit inline PDF/page references in the answer.
    A separate source list does not count as claim-level citation.
    """

    citation_pattern = re.compile(
        r"([^,\n\[\]]+\.pdf)\s*,?\s*(?:Page|p\.)\s*(\d+)",
        re.IGNORECASE,
    )

    citations = citation_pattern.findall(answer)

    if not citations:
        return {
            "total_citations": 0,
            "correct_citations": 0,
            "citation_accuracy": None,
            "note": "No inline PDF/page citations found in the answer.",
        }

    valid_sources = {
        (
            source.get("filename", "").lower(),
            str(source.get("page_number", "")),
        )
        for source in sources
    }

    correct = 0

    for filename, page in citations:
        reference = (filename.strip().lower(), page.strip())

        if reference in valid_sources:
            correct += 1

    return {
        "total_citations": len(citations),
        "correct_citations": correct,
        "citation_accuracy": correct / len(citations),
        "note": "Checks whether cited PDF/page references exist among retrieved sources.",
    }


def main():

    print("\n========== AUTOMATED ANSWER QUALITY EVALUATION ==========\n")

    if not RESULTS_FILE.exists():
        print("Evaluation results not found.")
        print("Run evaluate_rag.py first.")
        return

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    reviews = []

    for index, result in enumerate(results, start=1):

        question = result.get("question", "")
        answer = result.get("answer", "")
        sources = result.get("retrieved_sources", [])

        print(f"\nTest {index}: {question}")

        quality = evaluate_answer(answer, sources, model)
        citations = check_citations(answer, sources)

        review = {
            "question": question,
            "faithfulness_proxy": quality["faithfulness_proxy"],
            "hallucination_proxy": quality["hallucination_proxy"],
            "citation_evaluation": citations,
            "sentence_analysis": quality["sentence_results"],
        }

        reviews.append(review)

        print(
            "Faithfulness proxy:",
            f"{quality['faithfulness_proxy']:.2%}"
            if quality["faithfulness_proxy"] is not None
            else "N/A",
        )

        print(
            "Hallucination proxy:",
            f"{quality['hallucination_proxy']:.2%}"
            if quality["hallucination_proxy"] is not None
            else "N/A",
        )

        print(
            "Citation accuracy:",
            f"{citations['citation_accuracy']:.2%}"
            if citations["citation_accuracy"] is not None
            else "N/A",
        )

        for sentence in quality["sentence_results"]:
            status = (
                "SUPPORTED"
                if sentence["support_detected"]
                else "POSSIBLY UNSUPPORTED"
            )

            print(
                f"  [{status}] Similarity: "
                f"{sentence['best_similarity']:.3f}"
            )
            print(f"  {sentence['sentence']}")

    valid_faithfulness = [
        item["faithfulness_proxy"]
        for item in reviews
        if item["faithfulness_proxy"] is not None
    ]

    valid_hallucination = [
        item["hallucination_proxy"]
        for item in reviews
        if item["hallucination_proxy"] is not None
    ]

    valid_citations = [
        item["citation_evaluation"]["citation_accuracy"]
        for item in reviews
        if item["citation_evaluation"]["citation_accuracy"] is not None
    ]

    summary = {
        "questions_evaluated": len(reviews),
        "average_faithfulness_proxy": (
            float(np.mean(valid_faithfulness))
            if valid_faithfulness else None
        ),
        "average_hallucination_proxy": (
            float(np.mean(valid_hallucination))
            if valid_hallucination else None
        ),
        "average_citation_accuracy": (
            float(np.mean(valid_citations))
            if valid_citations else None
        ),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

    output = {
        "summary": summary,
        "individual_reviews": reviews,
        "evaluation_method": (
            "Automated sentence-to-context semantic similarity. "
            "Faithfulness and hallucination values are heuristic proxies, "
            "not verified entailment or factual correctness."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n========== FINAL QUALITY METRICS ==========")

    for name, value in summary.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value}")

    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("\nAutomated evaluation completed!")


if __name__ == "__main__":
    main()
