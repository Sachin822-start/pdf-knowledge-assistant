
import chromadb
from transformers import AutoTokenizer

DB_PATH = "vectorstore/chroma_db"
COLLECTION_NAME = "pdf_knowledge_base"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

total = collection.count()

print("Total chunks:", total)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

result = collection.get(
    limit=total,
    include=["documents", "metadatas"]
)

documents = result["documents"]
metadatas = result["metadatas"]

max_tokens = 0
oversized_chunks = 0
filenames = set()

for document, metadata in zip(documents, metadatas):

    token_count = len(
        tokenizer.encode(
            document,
            add_special_tokens=True,
            truncation=False
        )
    )

    max_tokens = max(max_tokens, token_count)

    if token_count > 512:
        oversized_chunks += 1

    filenames.add(metadata.get("filename", "Unknown"))

print("\n========== INDEX VERIFICATION ==========")

print("Total chunks:", total)
print("Unique PDF files:", len(filenames))
print("Maximum chunk token length:", max_tokens)
print("Chunks exceeding 512 tokens:", oversized_chunks)

print("\nIndexed PDF files:")

for filename in sorted(filenames):
    print("-", filename)

print("\n========================================")

if oversized_chunks == 0 and len(filenames) == 10:
    print("SUCCESS: Index verification passed!")
else:
    print("WARNING: Please investigate the verification results.")
