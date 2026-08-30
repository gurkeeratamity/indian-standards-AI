import chromadb

client = chromadb.PersistentClient(path="./dairy_vector_db")

collection = client.get_collection("dairy_knowledge")

print("Number of documents:", collection.count())

results = collection.query(
    query_texts=["dairy products and milk standards"],
    n_results=3
)

print("\nSearch results:")

for i, document in enumerate(results["documents"][0]):
    print("\nResult", i + 1)
    print(document)