import chromadb

DB_PATH = "./dairy_vector_db"
COLLECTION_NAME = "dairy_standards"
OLD_SOURCE = "FAD_Book-3-For-net.pdf"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

total = collection.count()

old_records = collection.get(
    where={"source_document": OLD_SOURCE},
    include=["metadatas"]
)

old_count = len(old_records["ids"])
other_count = total - old_count

print("\n--- ChromaDB Cleanup Preview ---")
print(f"Total records: {total}")
print(f"Old catalogue records: {old_count}")
print(f"Other records: {other_count}")

if old_count == 0:
    print("\nNo old catalogue records found.")
    exit()

print(f"\nReady to delete {old_count} records.")
print(f'Source: "{OLD_SOURCE}"')
print("Nothing has been deleted yet.")

confirmation = input("\nType YES to continue: ").strip()

if confirmation == "YES":
    collection.delete(
        where={"source_document": OLD_SOURCE}
    )

    print(f"\nDeleted: {old_count} records.")
    print(f"Remaining records: {collection.count()}")
else:
    print("\nDeletion cancelled. Database unchanged.")