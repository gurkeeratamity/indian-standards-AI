import sys
import json
from pathlib import Path
import chromadb


CHUNKS_FILE = Path("data/processed/chunks.jsonl")

DB_PATH = "./dairy_vector_db"
COLLECTION_NAME = "dairy_standards"

REQUIRED_FIELDS = [
    "chunk_id",
    "standard_id",
    "is_number",
    "chunk_text",
    "source_document",
    "page_start",
    "page_end",
]


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def load_chunks(path):
    if not path.exists():
        sys.exit(f"ERROR: Could not find '{path}'. Run chunk_pdfs.py first.")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"ERROR: Invalid JSON on line {line_number} of '{path}': {e}")
            chunks.append(record)

    return chunks


def validate_chunk(chunk):
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in chunk or _is_missing(chunk.get(field)):
            issues.append(f"missing {field}")
    return issues


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def _clean_value(value):
    if value is None:
        return ""
    return str(value)


def build_records(valid_chunks):
    ids = []
    documents = []
    metadatas = []

    for chunk in valid_chunks:
        ids.append(str(chunk["chunk_id"]))
        documents.append(str(chunk["chunk_text"]))

        metadata = {
            "standard_id": _clean_value(chunk.get("standard_id")),
            "is_number": _clean_value(chunk.get("is_number")),
            "section_number": _clean_value(chunk.get("section_number")),
            "section_title": _clean_value(chunk.get("section_title")),
            "page_start": _clean_value(chunk.get("page_start")),
            "page_end": _clean_value(chunk.get("page_end")),
            "chunk_type": _clean_value(chunk.get("chunk_type")),
            "source_document": _clean_value(chunk.get("source_document")),
            "verification_status": _clean_value(chunk.get("verification_status")),
        }
        metadatas.append(metadata)

    return ids, documents, metadatas


def main():
    chunks = load_chunks(CHUNKS_FILE)
    chunks_read = len(chunks)

    valid_chunks = []
    invalid_records = []

    for line_number, chunk in enumerate(chunks, start=1):
        issues = validate_chunk(chunk)
        if issues:
            invalid_records.append((line_number, chunk.get("chunk_id", "UNKNOWN"), issues))
        else:
            valid_chunks.append(chunk)

    if invalid_records:
        print("Invalid/missing records:")
        for line_number, chunk_id, issues in invalid_records:
            print(f"  Line {line_number} (chunk_id={chunk_id}): {', '.join(issues)}")
    else:
        print("No invalid records found.")

    collection = get_collection()
    ids, documents, metadatas = build_records(valid_chunks)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print()
    print(f"Chunks read: {chunks_read}")
    print(f"Valid chunks: {len(valid_chunks)}")
    print(f"Invalid chunks: {len(invalid_records)}")
    print(f"Documents uploaded: {len(ids)}")
    print(f"ChromaDB collection count: {collection.count()}")


if __name__ == "__main__":
    main()
