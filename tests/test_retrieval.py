
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "vectorstore/chroma_db"
COLLECTION_NAME = "pdf_knowledge_base"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 10

def main():

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("\nSemantic Retrieval Test")
    print("=" * 50)

    while True:

        query = input(
            "\nEnter your question (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        query_embedding = model.encode(
            [
                "Represent this sentence for searching relevant passages: "
                + query
            ],
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        print("\nRetrieved Results")
        print("-" * 50)

        for index in range(
            len(results["ids"][0])
        ):

            metadata = results["metadatas"][0][index]

            print(f"\nRank: {index + 1}")

            print(
                "PDF:",
                metadata.get("filename")
            )

            print(
                "Page:",
                metadata.get("page_number")
            )

            print(
                "Distance:",
                round(
                    results["distances"][0][index],
                    4
                )
            )

            print(
                "Content:",
                results["documents"][0][index][:500]
            )

            print("-" * 50)


if __name__ == "__main__":
    main()
