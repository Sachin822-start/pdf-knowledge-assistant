
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.ingestion.pdf_loader import extract_pdf
from src.ingestion.chunking import create_chunks


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

PDF_FOLDER = Path("data/pdfs")
OUTPUT_FOLDER = Path("data/processed")

BATCH_SIZE = 16


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():

    print("\nLoading embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully!")

    return model


# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    print("Total chunks:", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("\nEmbedding generation completed!")

    print("Embedding shape:", embeddings.shape)

    return embeddings


# --------------------------------------------------
# SAVE EMBEDDINGS AND CHUNK DATA
# --------------------------------------------------

def save_embeddings(chunks, embeddings):

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save numerical vectors
    np.save(
        OUTPUT_FOLDER / "embeddings.npy",
        embeddings
    )

    # Save corresponding text and metadata
    with open(
        OUTPUT_FOLDER / "chunks.jsonl",
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nSaved files:")
    print("1. data/processed/embeddings.npy")
    print("2. data/processed/chunks.jsonl")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:

        print("No PDF documents found!")

        raise SystemExit(1)

    all_chunks = []

    # Process every PDF
    for pdf_file in pdf_files:

        print("\nProcessing:", pdf_file.name)

        documents = extract_pdf(pdf_file)

        chunks = create_chunks(documents)

        all_chunks.extend(chunks)

    if not all_chunks:

        print("No chunks available for embedding.")

        raise SystemExit(1)

    print("\nTotal chunks:", len(all_chunks))

    # Load model
    embedding_model = load_embedding_model()

    # Generate vectors
    embeddings = generate_embeddings(
        all_chunks,
        embedding_model
    )

    # Save results
    save_embeddings(
        all_chunks,
        embeddings
    )

    print("\n========== PIPELINE COMPLETED ==========")
