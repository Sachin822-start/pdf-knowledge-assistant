
from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# --------------------------------------------------
# CHUNKING FUNCTION
# --------------------------------------------------

def create_chunks(documents):

    all_chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        if not text.strip():
            continue

        # Split each PDF page independently
        text_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(text_chunks):

            chunk_metadata = metadata.copy()

            chunk_metadata.update({
                "chunk_index": index,
                "chunk_size": len(
                    tokenizer.encode(
                        chunk_text,
                        add_special_tokens=False
                    )
                ),
                "chunk_id": (
                    f"{metadata['pdf_id']}_"
                    f"page_{metadata['page_number']}_"
                    f"chunk_{index}"
                )
            })

            all_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return all_chunks


# --------------------------------------------------
# SAVE CHUNKS TO TEXT FILE
# --------------------------------------------------

def save_chunks(chunks, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write("=" * 70 + "\n")

            file.write(
                f"Chunk ID: {chunk['metadata']['chunk_id']}\n"
            )

            file.write(
                f"Filename: {chunk['metadata']['filename']}\n"
            )

            file.write(
                f"Page: {chunk['metadata']['page_number']}\n"
            )

            file.write(
                f"Token count: {chunk['metadata']['chunk_size']}\n"
            )

            file.write("\n")

            file.write(chunk["text"])

            file.write("\n\n")


# --------------------------------------------------
# TEST CHUNKING PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    from src.ingestion.pdf_loader import extract_pdf

    pdf_folder = Path("data/pdfs")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:

        print("No PDF found in data/pdfs")

    else:

        print(f"Processing PDF: {pdf_files[0].name}")

        documents = extract_pdf(pdf_files[0])

        print(f"Extracted pages: {len(documents)}")

        chunks = create_chunks(documents)

        print(f"Total chunks generated: {len(chunks)}")

        save_chunks(
            chunks,
            "data/processed/chunks.txt"
        )

        print("\n========== CHUNKING RESULTS ==========")

        for chunk in chunks[:5]:

            print("\nChunk ID:",
                  chunk["metadata"]["chunk_id"])

            print("Page:",
                  chunk["metadata"]["page_number"])

            print("Token count:",
                  chunk["metadata"]["chunk_size"])

            print("Text:",
                  chunk["text"][:200])

        print("\nChunks saved successfully!")
