
import json
import numpy as np
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DB_PATH = "vectorstore/chroma_db"

COLLECTION_NAME = "pdf_knowledge_base"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")


# --------------------------------------------------
# INITIALIZE CHROMADB
# --------------------------------------------------

def initialize_database():

    print("\nInitializing ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "PDF RAG knowledge base",
            "hnsw:space": "cosine"
        }
    )

    print("ChromaDB initialized successfully!")

    print("Collection:", COLLECTION_NAME)

    return client, collection


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():

                chunks.append(json.loads(line))

    return chunks


# --------------------------------------------------
# STORE VECTORS IN CHROMADB
# --------------------------------------------------

def store_embeddings(collection):

    print("\nLoading chunks and embeddings...")

    chunks = load_chunks()

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    if len(chunks) != len(embeddings):

        raise ValueError(
            "Number of chunks and embeddings do not match!"
        )

    print("Total chunks:", len(chunks))

    print("Embedding dimensions:", embeddings.shape[1])

    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        ids.append(metadata["chunk_id"])

        texts.append(chunk["text"])

        metadatas.append({
            "pdf_id": str(metadata["pdf_id"]),
            "filename": str(metadata["filename"]),
            "page_number": int(metadata["page_number"]),
            "chunk_index": int(metadata["chunk_index"]),
            "chunk_size": int(metadata["chunk_size"]),
            "extraction_method": str(
                metadata["extraction_method"]
            )
        })

    # Upsert allows safe re-running without duplicate IDs
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\nEmbeddings stored successfully!")

    print("Total records:", collection.count())


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(query, collection, model, top_k=5):

    query_text = (
        "Represent this sentence for searching "
        "relevant passages: " + query
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    # Initialize vector database
    client, collection = initialize_database()

    # Load embeddings and store them
    store_embeddings(collection)

    # Load embedding model for query processing
    print("\nLoading query embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Test semantic search
    query = "What is the purpose of a NAT Gateway?"

    print("\nUser Query:", query)

    results = semantic_search(
        query,
        collection,
        model,
        top_k=5
    )

    print("\n========== RETRIEVED RESULTS ==========")

    for i, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]

        similarity = 1 - distance

        print(f"\nResult {i + 1}")

        print("PDF:", metadata["filename"])

        print("Page:", metadata["page_number"])

        print("Similarity:", round(similarity, 4))

        print("Text:", document[:300])

    print("\nVector database test completed!")
