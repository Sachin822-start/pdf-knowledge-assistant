
import time
import streamlit as st

from src.retrieval.reranker import (
    load_models,
    connect_database,
    retrieve_candidates,
    rerank_documents,
)

from src.generation.llm import generate_answer


# ==========================================================
# 1. CACHE MODELS
# ==========================================================

@st.cache_resource(show_spinner="Loading RAG models...")
def get_rag_models():

    print("\nLoading RAG models for the first time...")

    embedding_model, reranker = load_models()

    print("RAG models loaded and cached successfully!")

    return embedding_model, reranker


# ==========================================================
# 2. CACHE VECTOR DATABASE
# ==========================================================

@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def get_collection():

    print("\nConnecting to ChromaDB...")

    collection = connect_database()

    print("Connected to ChromaDB")
    print(f"Total records: {collection.count()}")

    return collection


# ==========================================================
# 3. PROGRESS CALLBACK
# ==========================================================

def update_progress(progress_callback, message, percentage):

    if progress_callback:
        progress_callback(message, percentage)


# ==========================================================
# 4. MAIN RAG PIPELINE
# ==========================================================

def run_rag_pipeline(question, progress_callback=None):

    pipeline_start = time.perf_counter()

    print("\n========== RAG PIPELINE ==========")

    # ------------------------------------------------------
    # STEP 1: VALIDATE QUESTION
    # ------------------------------------------------------

    if not question or not question.strip():

        return (
            question,
            "Please enter a valid question.",
            [],
            {
                "retrieval_seconds": 0,
                "reranking_seconds": 0,
                "generation_seconds": 0,
                "total_seconds": 0,
                "candidates_retrieved": 0,
                "chunks_after_reranking": 0,
                "sources_used": 0,
                "average_reranker_score": None,
            },
        )

    question = question.strip()

    try:

        # --------------------------------------------------
        # STEP 2: CONNECT TO CHROMADB
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Connecting to vector database...",
            10,
        )

        print("\n[1] Connecting to ChromaDB...")

        collection = get_collection()

        if collection.count() == 0:

            raise RuntimeError("The vector database contains no documents.")

        # --------------------------------------------------
        # STEP 3: LOAD MODELS
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Loading embedding and reranker models...",
            20,
        )

        print("\n[2] Loading or reusing retrieval models...")

        embedding_model, reranker = get_rag_models()

        # --------------------------------------------------
        # STEP 4: RETRIEVE CANDIDATES
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Searching relevant PDF passages...",
            35,
        )

        print("\n[3] Retrieving relevant PDF chunks...")

        retrieval_start = time.perf_counter()

        candidates = retrieve_candidates(
            question,
            collection,
            embedding_model,
        )

        retrieval_time = time.perf_counter() - retrieval_start

        print(f"Retrieval time: {retrieval_time:.3f} seconds")
        print(f"Candidates retrieved: {len(candidates)}")

        if not candidates:

            total_time = time.perf_counter() - pipeline_start

            metrics = {
                "retrieval_seconds": round(retrieval_time, 3),
                "reranking_seconds": 0,
                "generation_seconds": 0,
                "total_seconds": round(total_time, 3),
                "candidates_retrieved": 0,
                "chunks_after_reranking": 0,
                "sources_used": 0,
                "average_reranker_score": None,
            }

            return (
                question,
                "No relevant information was found in the indexed PDFs.",
                [],
                metrics,
            )

        # --------------------------------------------------
        # STEP 5: RERANK DOCUMENTS
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Reranking retrieved passages...",
            55,
        )

        print("\n[4] Reranking documents...")

        reranking_start = time.perf_counter()

        final_results = rerank_documents(
            question,
            candidates,
            reranker,
            top_k=5,
        )

        reranking_time = time.perf_counter() - reranking_start

        print(f"Reranking time: {reranking_time:.3f} seconds")
        print(f"Final chunks selected: {len(final_results)}")

        if not final_results:

            total_time = time.perf_counter() - pipeline_start

            metrics = {
                "retrieval_seconds": round(retrieval_time, 3),
                "reranking_seconds": round(reranking_time, 3),
                "generation_seconds": 0,
                "total_seconds": round(total_time, 3),
                "candidates_retrieved": len(candidates),
                "chunks_after_reranking": 0,
                "sources_used": 0,
                "average_reranker_score": None,
            }

            return (
                question,
                "No relevant passages remained after reranking.",
                [],
                metrics,
            )

        # --------------------------------------------------
        # STEP 6: PREPARE CONTEXT
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Preparing context from PDF documents...",
            70,
        )

        print("\n[5] Preparing context for LLM...")

        context_parts = []

        for index, result in enumerate(final_results, start=1):

            metadata = result["metadata"]

            filename = metadata.get("filename", "Unknown PDF")
            page_number = metadata.get("page_number", "Unknown")
            text = result["text"]

            context_parts.append(
                f"""
PASSAGE {index}
PDF: {filename}
PAGE: {page_number}
CONTENT:
{text}
"""
            )

        context = "\n\n".join(context_parts)

        # --------------------------------------------------
        # STEP 7: GENERATE ANSWER
        # --------------------------------------------------

        update_progress(
            progress_callback,
            "Generating answer using Qwen LLM...",
            85,
        )

        print("\n[6] Generating answer using Qwen...")

        generation_start = time.perf_counter()

        answer = generate_answer(
            question,
            context,
        )

        generation_time = time.perf_counter() - generation_start

        print(f"Generation time: {generation_time:.3f} seconds")

        # --------------------------------------------------
        # STEP 8: PREPARE SOURCES
        # --------------------------------------------------

        sources = []

        for result in final_results:

            metadata = result["metadata"]

            sources.append({
                "filename": metadata.get("filename", "Unknown PDF"),
                "page_number": metadata.get("page_number", "Unknown"),
                "rerank_score": float(result["rerank_score"]),
                "extraction_method": metadata.get(
                    "extraction_method",
                    "Unknown",
                ),
                "text": result["text"],
            })

        # --------------------------------------------------
        # STEP 9: CALCULATE QUERY METRICS
        # --------------------------------------------------

        total_time = time.perf_counter() - pipeline_start

        reranker_scores = [
            result["rerank_score"]
            for result in final_results
        ]

        average_score = (
            sum(reranker_scores) / len(reranker_scores)
            if reranker_scores else None
        )

        unique_pdfs = {
            source["filename"]
            for source in sources
        }

        metrics = {

            "retrieval_seconds": round(retrieval_time, 3),

            "reranking_seconds": round(reranking_time, 3),

            "generation_seconds": round(generation_time, 3),

            "total_seconds": round(total_time, 3),

            "candidates_retrieved": len(candidates),

            "chunks_after_reranking": len(final_results),

            "sources_used": len(unique_pdfs),

            "average_reranker_score": (
                round(float(average_score), 4)
                if average_score is not None else None
            ),
        }

        # --------------------------------------------------
        # STEP 10: PRINT RESULTS
        # --------------------------------------------------

        print("\n========== FINAL ANSWER ==========")
        print(answer)

        print("\n========== QUERY PERFORMANCE ==========")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        print("\n========== SOURCE DOCUMENTS ==========")

        for source in sources:

            print(
                f"{source['filename']} | "
                f"Page {source['page_number']} | "
                f"Score {source['rerank_score']:.4f}"
            )

        update_progress(
            progress_callback,
            "Answer generated successfully!",
            100,
        )

        print("\nRAG pipeline completed successfully!")

        return question, answer, sources, metrics

    except Exception as error:

        print(f"\nRAG pipeline error: {error}")

        raise


# ==========================================================
# 5. STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    question = "What is the purpose of a NAT Gateway?"

    result = run_rag_pipeline(question)

    print("\nQuestion:", result[0])
    print("\nAnswer:", result[1])
    print("\nSources:", result[2])
    print("\nMetrics:", result[3])
