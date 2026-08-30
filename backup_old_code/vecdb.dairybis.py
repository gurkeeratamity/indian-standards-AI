import pandas as pd
import chromadb

# Read Excel
df = pd.read_excel("dairy.bis.xlsx")

# Clean data
df = df.dropna(how="all")
df.columns = ["topic", "information"]
df = df.dropna(subset=["information"])

# Convert rows to documents
documents = []

for _, row in df.iterrows():
    document = f"Topic: {row['topic']}\nInformation: {row['information']}"
    documents.append(document)

# Create local vector database
client = chromadb.PersistentClient(path="./dairy_vector_db")

# Create collection
collection = client.get_or_create_collection(
    name="dairy_knowledge"
)

# Add documents
collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

print("✅ Vector database created successfully!")
print("Number of documents:", collection.count())