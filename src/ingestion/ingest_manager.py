
import gc
import json
import hashlib
import logging

from pathlib import Path

import chromadb
import pymupdf

from sentence_transformers import SentenceTransformer

from src.ingestion.pdf_loader import extract_pdf
from src.ingestion.chunking import create_chunks


# =====================================================
# CONFIGURATION
# =====================================================

PDF_FOLDER = Path("data/pdfs")

DB_PATH = "vectorstore/chroma_db"

COLLECTION_NAME = "pdf_knowledge_base"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

REGISTRY_FILE = Path(
    "data/processed/ingestion_registry.json"
)

# Start with 8 for your 8 GB RAM laptop.
# Benchmark 16 later if memory permits.
EMBEDDING_BATCH_SIZE = 8

MIN_REQUIRED_PAGES = 200


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger("PDF_INGESTION")


# =====================================================
# FILE HASH
# =====================================================

def calculate_file_hash(pdf_path):

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(block)

    return sha256.hexdigest()


# =====================================================
# REGISTRY MANAGEMENT
# =====================================================

def load_registry():

    if not REGISTRY_FILE.exists():
        return {}

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:

        logger.warning(
            "Could not read ingestion registry: %s",
            error
        )

        return {}


def save_registry(registry):

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = REGISTRY_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )

    temporary_file.replace(REGISTRY_FILE)


# =====================================================
# CHROMADB INITIALIZATION
# =====================================================

def initialize_collection():

    logger.info("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Multi-PDF RAG knowledge base",
            "hnsw:space": "cosine"
        }
    )

    logger.info(
        "Connected. Existing records: %s",
        collection.count()
    )

    return collection


# =====================================================
# CHECK WHETHER FILE IS ALREADY INDEXED
# =====================================================

def find_existing_hash(collection, file_hash):

    result = collection.get(
        where={
            "file_hash": file_hash
        },
        include=["metadatas"],
        limit=1
    )

    if result["ids"]:

        metadata = result["metadatas"][0]

        return metadata.get(
            "filename",
            "Unknown document"
        )

    return None


# =====================================================
# REMOVE OLD DOCUMENT CHUNKS
# =====================================================

def remove_old_document(collection, pdf_id):

    existing = collection.get(
        where={
            "pdf_id": pdf_id
        },
        include=[]
    )

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

        logger.info(
            "Removed %s previous chunks for %s",
            len(existing["ids"]),
            pdf_id
        )


# =====================================================
# VALIDATE PDF
# =====================================================

def validate_pdf(pdf_path):

    try:

        with pymupdf.open(pdf_path) as pdf:

            if pdf.is_encrypted:

                raise ValueError(
                    "Encrypted PDFs are not supported."
                )

            page_count = len(pdf)

        if page_count == 0:

            raise ValueError(
                "PDF contains no pages."
            )

        return page_count

    except Exception as error:

        raise ValueError(
            f"Invalid PDF {pdf_path.name}: {error}"
        )


# =====================================================
# INGEST ONE PDF
# =====================================================

def ingest_pdf(
    pdf_path,
    collection,
    model,
    registry
):

    filename = pdf_path.name

    pdf_id = pdf_path.stem

    logger.info("=" * 55)

    logger.info("Processing: %s", filename)

    # -----------------------------------------------
    # STEP 1: Calculate file hash
    # -----------------------------------------------

    file_hash = calculate_file_hash(pdf_path)

    # -----------------------------------------------
    # STEP 2: Check ChromaDB for identical content
    # -----------------------------------------------

    existing_filename = find_existing_hash(
        collection,
        file_hash
    )

    if existing_filename:

        logger.info(
            "Already present in ChromaDB: %s",
            existing_filename
        )

        logger.info(
            "Duplicate document detected: %s. Skipping.",
            filename
        )

        # Save alias information if filenames differ
        if filename != existing_filename:

            registry[filename] = {
                "file_hash": file_hash,
                "duplicate_of": existing_filename
            }

            save_registry(registry)

        return "skipped"

    # -----------------------------------------------
    # STEP 3: Check registry
    # -----------------------------------------------

    previous = registry.get(filename)

    if (
        previous
        and previous.get("file_hash") == file_hash
    ):

        # Confirm the document still exists in ChromaDB.
        # If it does not, re-index it.
        logger.info(
            "Registry entry found, but no matching "
            "ChromaDB record. Re-indexing."
        )

    # -----------------------------------------------
    # STEP 4: Validate PDF
    # -----------------------------------------------

    page_count = validate_pdf(pdf_path)

    logger.info(
        "PDF page count: %s",
        page_count
    )

    if page_count < MIN_REQUIRED_PAGES:

        logger.warning(
            "%s has only %s pages. "
            "Final challenge requires at least 200 pages.",
            filename,
            page_count
        )

    # -----------------------------------------------
    # STEP 5: Extract text and OCR
    # -----------------------------------------------

    logger.info(
        "[1/4] Extracting text and performing OCR..."
    )

    documents = extract_pdf(pdf_path)

    if not documents:

        raise ValueError(
            "No text could be extracted from this PDF."
        )

    logger.info(
        "Successfully extracted %s pages.",
        len(documents)
    )

    # -----------------------------------------------
    # STEP 6: Create chunks
    # -----------------------------------------------

    logger.info(
        "[2/4] Creating chunks..."
    )

    chunks = create_chunks(documents)

    if not chunks:

        raise ValueError(
            "No chunks generated."
        )

    logger.info(
        "Total chunks generated: %s",
        len(chunks)
    )

    # -----------------------------------------------
    # STEP 7: Prepare metadata
    # -----------------------------------------------

    texts = []

    ids = []

    metadatas = []

    for chunk in chunks:

        metadata = chunk["metadata"].copy()

        metadata["file_hash"] = file_hash

        texts.append(
            chunk["text"]
        )

        ids.append(
            metadata["chunk_id"]
        )

        metadatas.append(metadata)

    # -----------------------------------------------
    # STEP 8: Remove old version if file changed
    # -----------------------------------------------

    if previous:

        logger.info(
            "Updated PDF detected. Replacing previous version."
        )

        remove_old_document(
            collection,
            pdf_id
        )

    # -----------------------------------------------
    # STEP 9: Generate embeddings and store in batches
    # -----------------------------------------------

    logger.info(
        "[3/4] Generating embeddings and storing in ChromaDB..."
    )

    total_chunks = len(texts)

    for start in range(
        0,
        total_chunks,
        EMBEDDING_BATCH_SIZE
    ):

        end = min(
            start + EMBEDDING_BATCH_SIZE,
            total_chunks
        )

        batch_texts = texts[start:end]

        batch_ids = ids[start:end]

        batch_metadatas = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas
        )

        logger.info(
            "Stored %s/%s chunks",
            end,
            total_chunks
        )

        del embeddings

    # -----------------------------------------------
    # STEP 10: Save registry after successful ingestion
    # -----------------------------------------------

    registry[filename] = {
        "file_hash": file_hash,
        "page_count": page_count,
        "chunk_count": total_chunks
    }

    save_registry(registry)

    logger.info(
        "[4/4] Successfully indexed: %s",
        filename
    )

    return "completed"


# =====================================================
# MAIN INGESTION MANAGER
# =====================================================

def run_ingestion():

    PDF_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    if not pdf_files:

        logger.warning(
            "No PDF files found in %s",
            PDF_FOLDER
        )

        return

    logger.info("=" * 55)

    logger.info(
        "MULTI-PDF INGESTION STARTED"
    )

    logger.info(
        "PDF documents discovered: %s",
        len(pdf_files)
    )

    # Initialize ChromaDB
    collection = initialize_collection()

    registry = load_registry()

    # Load embedding model once
    logger.info(
        "Loading embedding model: %s",
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    logger.info(
        "Embedding model loaded successfully."
    )

    completed = 0

    skipped = 0

    failed = []

    # Process documents sequentially
    for index, pdf_path in enumerate(
        pdf_files,
        start=1
    ):

        logger.info(
            "Document %s/%s",
            index,
            len(pdf_files)
        )

        try:

            result = ingest_pdf(
                pdf_path,
                collection,
                model,
                registry
            )

            if result == "completed":

                completed += 1

            elif result == "skipped":

                skipped += 1

        except Exception as error:

            logger.exception(
                "Failed to process %s",
                pdf_path.name
            )

            failed.append({
                "filename": pdf_path.name,
                "error": str(error)
            })

    # Release memory
    del model

    gc.collect()

    # Final report
    logger.info("=" * 55)

    logger.info(
        "INGESTION SUMMARY"
    )

    logger.info(
        "New or updated PDFs: %s",
        completed
    )

    logger.info(
        "Skipped PDFs: %s",
        skipped
    )

    logger.info(
        "Failed PDFs: %s",
        len(failed)
    )

    logger.info(
        "Total ChromaDB records: %s",
        collection.count()
    )

    if failed:

        logger.error(
            "Failed documents: %s",
            failed
        )

    logger.info(
        "MULTI-PDF INGESTION FINISHED"
    )
    return {
            "total_documents": len(pdf_files),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "total_chunks": collection.count()
        }


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    run_ingestion()


