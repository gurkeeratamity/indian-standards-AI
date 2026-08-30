import sys
import chromadb


DB_PATH = "./dairy_vector_db"
COLLECTION_NAME = "dairy_standards"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        sys.exit("Collection not found. Run python3 ingest.py first.")
    return collection


def search(collection, query_text, n_results=5):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    return results


def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        print("No matching results found.")
        return

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        print(f"--- Result {i} ---")
        print(f"IS Number:            {meta.get('is_number', '')}")
        print(f"Standard ID:          {meta.get('standard_id', '')}")
        print(f"Section:              {meta.get('section_number', '')} {meta.get('section_title', '')}")
        print(f"Pages:                {meta.get('page_start', '')} - {meta.get('page_end', '')}")
        print(f"Source Document:      {meta.get('source_document', '')}")
        print(f"Verification Status:  {meta.get('verification_status', '')}")
        print(f"Matched Text:         {doc}")
        print()


def main():
    collection = get_collection()

    query_text = input("Enter procurement requirement: ").strip()
    if not query_text:
        sys.exit("No requirement entered.")

    results = search(collection, query_text, n_results=5)
    print_results(results)


if __name__ == "__main__":
    main()