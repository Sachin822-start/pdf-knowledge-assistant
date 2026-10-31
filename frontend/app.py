
import sys
import json
from pathlib import Path

import streamlit as st


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PDF_FOLDER = PROJECT_ROOT / "data" / "pdfs"

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation_results.json"
)


# ==========================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_data" not in st.session_state:
    st.session_state.evaluation_data = None


# ==========================================================
# 4. HELPER FUNCTIONS
# ==========================================================

def display_sources(sources):
    """Display retrieved PDF sources."""

    if not sources:
        st.info("No source documents were found.")
        return

    with st.expander(
        f"📄 Retrieved Sources ({len(sources)})",
        expanded=True
    ):

        for index, source in enumerate(sources, start=1):

            filename = source.get(
                "filename", "Unknown PDF"
            )

            page_number = source.get(
                "page_number", "Unknown"
            )

            score = source.get("rerank_score")

            extraction_method = source.get(
                "extraction_method", "Unknown"
            )

            st.markdown(
                f"**Source {index}: {filename}**"
            )

            st.caption(
                f"Page: {page_number} | "
                f"Extraction: {extraction_method}"
            )

            if score is not None:
                st.caption(
                    f"Reranker Score: {score:.4f}"
                )

            source_text = source.get("text", "")

            if source_text:
                st.write(source_text)

            if index < len(sources):
                st.divider()


def display_query_metrics(metrics):
    """Display metrics for the current query only."""

    if not metrics:
        return

    st.subheader("📊 Query Performance Metrics")

    retrieval_time = metrics.get(
        "retrieval_seconds", 0
    )

    reranking_time = metrics.get(
        "reranking_seconds", 0
    )

    generation_time = metrics.get(
        "generation_seconds", 0
    )

    total_time = metrics.get(
        "total_seconds", 0
    )

    candidates = metrics.get(
        "candidates_retrieved", 0
    )

    final_chunks = metrics.get(
        "chunks_after_reranking", 0
    )

    sources_used = metrics.get(
        "sources_used", 0
    )

    average_score = metrics.get(
        "average_reranker_score", 0
    )

    # Query latency

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retrieval Time",
        f"{retrieval_time:.2f} sec"
    )

    col2.metric(
        "Reranking Time",
        f"{reranking_time:.2f} sec"
    )

    col3.metric(
        "Generation Time",
        f"{generation_time:.2f} sec"
    )

    col4.metric(
        "Total Response Time",
        f"{total_time:.2f} sec"
    )

    # Retrieval statistics

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Candidates Retrieved",
        candidates
    )

    col6.metric(
        "Final Chunks",
        final_chunks
    )

    col7.metric(
        "Source PDFs",
        sources_used
    )

    col8.metric(
        "Average Reranker Score",
        f"{average_score:.4f}"
    )

    # Detailed metrics

    with st.expander("View Detailed Query Metrics"):

        st.write({
            "Retrieval Time (seconds)": retrieval_time,
            "Reranking Time (seconds)": reranking_time,
            "Generation Time (seconds)": generation_time,
            "Total Response Time (seconds)": total_time,
            "Candidates Retrieved": candidates,
            "Chunks After Reranking": final_chunks,
            "Sources Used": sources_used,
            "Average Reranker Score": average_score
        })


def display_assistant_message(message):
    """Display answer, sources and metrics."""

    st.markdown(
        message.get("content", "")
    )

    display_sources(
        message.get("sources", [])
    )

    display_query_metrics(
        message.get("metrics", {})
    )


# ==========================================================
# 5. EVALUATION FUNCTIONS
# ==========================================================

def load_evaluation_results():
    """Load the latest evaluation report."""

    if not EVALUATION_FILE.exists():
        return None

    try:

        with open(
            EVALUATION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:

        st.error(
            f"Could not read evaluation report: {error}"
        )

        return None


def get_evaluation_summary(data):

    if isinstance(data, dict):
        return data.get("summary", {})

    return {}


def get_individual_results(data):

    if isinstance(data, dict):

        return data.get(
            "individual_results",
            data.get("results", [])
        )

    if isinstance(data, list):
        return data

    return []


def get_metric(summary, *possible_names):

    for name in possible_names:

        if name in summary:
            return summary[name]

    return None


def display_evaluation_results(data):
    """Display saved RAG evaluation metrics."""

    if not data:

        st.info(
            "No evaluation report found. "
            "Run the evaluation script first."
        )

        return

    summary = get_evaluation_summary(data)

    individual_results = get_individual_results(data)

    if not summary:

        st.warning(
            "The evaluation report does not contain a summary."
        )

        return

    st.caption(
        f"Evaluation test cases: {len(individual_results)}"
    )

    precision = get_metric(
        summary,
        "average_precision_at_5",
        "precision_at_5"
    )

    recall = get_metric(
        summary,
        "average_recall_at_5",
        "recall_at_5"
    )

    mrr = get_metric(
        summary,
        "average_mrr",
        "mrr"
    )

    citation_accuracy = get_metric(
        summary,
        "average_citation_reference_accuracy",
        "citation_reference_accuracy"
    )

    p95_latency = get_metric(
        summary,
        "p95_latency_seconds",
        "p95_latency",
        "p95_total_seconds"
    )

    st.subheader("📈 RAG Evaluation Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Precision@5",
        f"{precision:.4f}"
        if precision is not None else "N/A"
    )

    col2.metric(
        "Recall@5",
        f"{recall:.4f}"
        if recall is not None else "N/A"
    )

    col3.metric(
        "MRR",
        f"{mrr:.4f}"
        if mrr is not None else "N/A"
    )

    col4.metric(
        "Citation Reference Accuracy",
        f"{citation_accuracy:.4f}"
        if citation_accuracy is not None else "N/A"
    )

    col5.metric(
        "p95 Latency",
        f"{p95_latency:.2f} sec"
        if p95_latency is not None else "N/A"
    )

    st.caption(
        "Citation reference accuracy measures whether retrieved "
        "source references match expected references. "
        "It does not verify claim-level factual support."
    )

    if individual_results:

        with st.expander(
            "View Individual Evaluation Results"
        ):

            for index, result in enumerate(
                individual_results,
                start=1
            ):

                st.markdown(
                    f"**Test {index}: "
                    f"{result.get('question', 'Unknown question')}**"
                )

                metric_columns = st.columns(4)

                metric_columns[0].metric(
                    "Precision@5",
                    f"{result.get('precision_at_5', 0):.4f}"
                )

                metric_columns[1].metric(
                    "Recall@5",
                    f"{result.get('recall_at_5', 0):.4f}"
                )

                metric_columns[2].metric(
                    "MRR",
                    f"{result.get('mrr', 0):.4f}"
                )

                metric_columns[3].metric(
                    "Total Time",
                    f"{result.get('total_seconds', 0):.2f} sec"
                )

                with st.expander("View Generated Answer"):

                    st.write(
                        result.get(
                            "answer",
                            "No answer available."
                        )
                    )

                st.divider()


# ==========================================================
# 6. SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("📚 PDF Knowledge Assistant")

    st.markdown(
        "Ask questions from your indexed PDF documents."
    )

    st.divider()

    # ----------------------------------------------
    # DOCUMENT MANAGEMENT
    # ----------------------------------------------

    st.subheader("📂 Knowledge Base")

    pdf_files = (
        list(PDF_FOLDER.glob("*.pdf"))
        if PDF_FOLDER.exists()
        else []
    )

    st.caption(
        f"PDF files available: {len(pdf_files)}"
    )

    if st.button(
        "📥 Load Documents into ChromaDB",
        width="stretch",
        type="primary"
    ):

        if not pdf_files:

            st.warning(
                "No PDF files found. Add PDFs to data/pdfs first."
            )

        else:

            try:

                from src.ingestion.ingest_manager import (
                    run_ingestion
                )

                with st.spinner(
                    "Processing PDFs, generating embeddings "
                    "and updating ChromaDB..."
                ):

                    ingestion_result = run_ingestion()

                if ingestion_result is None:

                    st.warning(
                        "Ingestion returned no result. "
                        "Check the return statement in ingest_manager.py."
                    )

                else:

                    st.success(
                        "Document ingestion finished!"
                    )

                    st.metric(
                        "Documents Discovered",
                        ingestion_result.get(
                            "total_documents", 0
                        )
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "New / Updated",
                        ingestion_result.get(
                            "completed", 0
                        )
                    )

                    col2.metric(
                        "Skipped",
                        ingestion_result.get(
                            "skipped", 0
                        )
                    )

                    failed = ingestion_result.get(
                        "failed", []
                    )

                    st.metric(
                        "Failed",
                        len(failed)
                    )

                    st.metric(
                        "Total ChromaDB Chunks",
                        ingestion_result.get(
                            "total_chunks", 0
                        )
                    )

                    if failed:

                        st.error(
                            "Some PDFs could not be indexed."
                        )

                        for item in failed:

                            st.write(
                                f"**{item.get('filename')}**: "
                                f"{item.get('error')}"
                            )

                    else:

                        st.success(
                            "All documents were processed successfully."
                        )

            except Exception as error:

                st.error(
                    f"Ingestion failed: {error}"
                )

                st.exception(error)

    st.divider()

    # ----------------------------------------------
    # EVALUATION RESULTS
    # ----------------------------------------------

    st.subheader("📊 RAG Evaluation")

    if st.button(
        "🔄 Refresh Evaluation Results",
        width="stretch"
    ):

        st.session_state.evaluation_data = (
            load_evaluation_results()
        )

        if st.session_state.evaluation_data is not None:

            st.success(
                "Evaluation results loaded."
            )

        else:

            st.warning(
                "Evaluation report not found."
            )

    st.caption(
        "Run the evaluation script separately, then refresh "
        "to display the latest results."
    )

    st.divider()

    # ----------------------------------------------
    # TECHNOLOGY STACK
    # ----------------------------------------------

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **PDF Extraction:** PyMuPDF
        - **OCR:** Tesseract
        - **Chunking:** LangChain
        - **Embeddings:** BGE Small
        - **Vector Database:** ChromaDB
        - **Retrieval:** Semantic Search
        - **Reranking:** BGE Reranker
        - **LLM:** Ollama Qwen 2.5
        - **Frontend:** Streamlit
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat History",
        width="stretch"
    ):

        st.session_state.messages = []

        st.rerun()


# ==========================================================
# 7. MAIN HEADER
# ==========================================================

st.title("📚 PDF Knowledge Assistant")

st.markdown(
    """
    Ask questions about your PDF documents.

    The system retrieves relevant passages, reranks them,
    and generates an answer using a locally running language model.
    """
)

st.divider()


# ==========================================================
# 8. RAG EVALUATION RESULTS
# ==========================================================

st.subheader("📊 RAG Evaluation Results")

if st.session_state.evaluation_data is None:

    st.info(
        "Click Refresh Evaluation Results in the sidebar "
        "to load the latest evaluation report."
    )

else:

    display_evaluation_results(
        st.session_state.evaluation_data
    )

st.divider()


# ==========================================================
# 9. CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    with st.chat_message(role):

        if role == "user":

            st.markdown(
                message.get("content", "")
            )

        else:

            display_assistant_message(message)


# ==========================================================
# 10. USER QUESTION AND RAG PIPELINE
# ==========================================================

question = st.chat_input(
    "Ask a question about your PDF documents..."
)


if question:

    # ----------------------------------------------
    # DISPLAY USER QUESTION
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # ----------------------------------------------
    # START ASSISTANT RESPONSE
    # ----------------------------------------------

    with st.chat_message("assistant"):

        # Create UI elements BEFORE importing RAG modules.
        # This allows Streamlit to display feedback first.

        status_text = st.empty()

        progress_bar = st.progress(0)

        status_text.info(
            "⏳ Initializing your RAG assistant..."
        )

        try:

            # --------------------------------------
            # IMPORT RAG PIPELINE
            # --------------------------------------

            status_text.info(
                "⚙️ Loading RAG pipeline..."
            )

            from src.generation.rag_pipeline import (
                run_rag_pipeline
            )

            # --------------------------------------
            # PROGRESS CALLBACK
            # --------------------------------------

            def update_progress(message, percentage):

                safe_percentage = min(
                    max(int(percentage), 0),
                    100
                )

                progress_bar.progress(
                    safe_percentage
                )

                status_text.info(
                    f"🔄 {message}"
                )

            # --------------------------------------
            # EXECUTE RAG PIPELINE
            # --------------------------------------

            status_text.info(
                "🔍 Searching your indexed PDF documents..."
            )

            with st.spinner(
                "Please wait while your answer is generated..."
            ):

                (
                    processed_question,
                    answer,
                    sources,
                    metrics
                ) = run_rag_pipeline(
                    question,
                    progress_callback=update_progress
                )

            # --------------------------------------
            # COMPLETION
            # --------------------------------------

            progress_bar.progress(100)

            status_text.success(
                "✅ Answer generated successfully!"
            )

            # --------------------------------------
            # DISPLAY ANSWER
            # --------------------------------------

            st.subheader("Answer")

            st.markdown(answer)

            # --------------------------------------
            # DISPLAY SOURCES
            # --------------------------------------

            display_sources(sources)

            # --------------------------------------
            # DISPLAY QUERY METRICS
            # --------------------------------------

            display_query_metrics(metrics)

            # --------------------------------------
            # SAVE ASSISTANT RESPONSE
            # --------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metrics": metrics
            })

        except Exception as error:

            status_text.error(
                "❌ An error occurred while processing your question."
            )

            st.error(str(error))

            st.exception(error)


# ==========================================================
# 11. FOOTER
# ==========================================================

st.divider()

st.caption(
    "PDF Knowledge Assistant | "
    "PyMuPDF | Tesseract | ChromaDB | "
    "BGE Embeddings | BGE Reranker | "
    "Ollama Qwen | Streamlit"
)
