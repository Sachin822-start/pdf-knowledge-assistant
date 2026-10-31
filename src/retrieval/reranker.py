
import time
import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DB_PATH = "vectorstore/chroma_db"

COLLECTION_NAME = "pdf_knowledge_base"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL = "BAAI/bge-reranker-base"

CANDIDATE_COUNT = 8

FINAL_TOP_K = 5

RERANKER_MAX_LENGTH = 256

RERANKER_BATCH_SIZE = 4

DEVICE = "cpu"


# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

def load_models():

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE
    )

    print("\nLoading reranker model...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=RERANKER_MAX_LENGTH,
        device=DEVICE
    )

    print("\nModels loaded successfully!")

    print(f"Device: {DEVICE}")
    print(f"Reranker max length: {RERANKER_MAX_LENGTH}")
    print(f"Reranker batch size: {RERANKER_BATCH_SIZE}")

    return embedding_model, reranker


# --------------------------------------------------
# CONNECT TO CHROMADB
# --------------------------------------------------

def connect_database():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("\nConnected to ChromaDB")
    print("Total records:", collection.count())

    return collection


# --------------------------------------------------
# RETRIEVE CANDIDATE DOCUMENTS
# --------------------------------------------------

def retrieve_candidates(query, collection, embedding_model):

    if not query or not query.strip():
        return []

    start_time = time.perf_counter()

    query_text = (
        "Represent this sentence for searching "
        "relevant passages: " + query
    )

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    total_records = collection.count()

    if total_records == 0:
        return []

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(CANDIDATE_COUNT, total_records),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    candidates = []

    for i, document in enumerate(results["documents"][0]):

        candidates.append({
            "text": document,
            "metadata": results["metadatas"][0][i],
            "vector_distance": results["distances"][0][i]
        })

    elapsed_time = time.perf_counter() - start_time

    print(
        f"Retrieval time: {elapsed_time:.2f} seconds"
    )

    return candidates


# --------------------------------------------------
# RERANK DOCUMENTS
# --------------------------------------------------

def rerank_documents(
    query,
    candidates,
    reranker,
    top_k=FINAL_TOP_K
):

    if not candidates:
        print("\nNo candidates available for reranking.")
        return []

    print("\nReranking retrieved documents...")

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    start_time = time.perf_counter()

    scores = reranker.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False
    )

    rerank_time = time.perf_counter() - start_time

    for candidate, score in zip(candidates, scores):

        candidate["rerank_score"] = float(score)

    ranked_results = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    print(
        f"Reranking time: {rerank_time:.2f} seconds"
    )

    print(
        f"Candidates processed: {len(candidates)}"
    )

    print(
        f"Final documents selected: "
        f"{min(top_k, len(ranked_results))}"
    )

    return ranked_results[:top_k]


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

    print("\n========== RERANKED RESULTS ==========")

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nRank: {rank}")

        print(
            "PDF:",
            metadata.get("filename", "Unknown")
        )

        print(
            "Page:",
            metadata.get("page_number", "Unknown")
        )

        print(
            "Vector distance:",
            round(result["vector_distance"], 4)
        )

        print(
            "Reranker score:",
            round(result["rerank_score"], 4)
        )

        print(
            "Text:",
            result["text"][:300]
        )

        print("-" * 50)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    query = "What is the purpose of a NAT Gateway?"

    print("\n========== RERANKER PERFORMANCE TEST ==========")

    total_start = time.perf_counter()

    # Connect to vector database
    collection = connect_database()

    # Load models
    embedding_model, reranker = load_models()

    # Retrieve candidates
    candidates = retrieve_candidates(
        query,
        collection,
        embedding_model
    )

    print(
        "\nRetrieved candidates:",
        len(candidates)
    )

    # Rerank candidates
    final_results = rerank_documents(
        query,
        candidates,
        reranker,
        top_k=FINAL_TOP_K
    )

    # Display results
    display_results(final_results)

    total_time = time.perf_counter() - total_start

    print("\n========== PERFORMANCE SUMMARY ==========")

    print(f"Total execution time: {total_time:.2f} seconds")

    print(f"Candidates retrieved: {len(candidates)}")

    print(f"Final results: {len(final_results)}")

    print("\nReranking pipeline completed!")

    del embedding_model
    del reranker
