from src.retrieval.reranker import (
    load_models,
    connect_database,
    retrieve_candidates,
    rerank_documents,
    display_results
)


def main():
    print("Loading embedding and reranker models...")
    embedding_model, reranker = load_models()

    print("Connecting to ChromaDB...")
    collection = connect_database()

    print("\nReranker Test")
    print("=" * 50)

    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        print("\nRetrieving top 10 candidates...")
        candidates = retrieve_candidates(
            query,
            collection,
            embedding_model
        )

        print(f"Retrieved {len(candidates)} candidates.")

        print("\nReranking candidates...")
        results = rerank_documents(
            query,
            candidates,
            reranker,
            top_k=5
        )

        print("\nTop 5 Reranked Results")
        print("=" * 50)
        display_results(results)


if __name__ == "__main__":
    main()