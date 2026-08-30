import chromadb

client = chromadb.PersistentClient(
    path="/Users/gurkeeratsingh/Desktop/Indian_standards_ai/dairy_vector_db"
)

collection = client.get_collection("dairy_standards")

query = input("Enter procurement requirement: ")

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\n🔎 Recommended standards:\n")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print(f"Result {i + 1}")
    print("Standard:", metadata["is_number"])
    print("Section:", metadata["section_title"])
    print("Text:", document)
    print("-" * 60)