import pandas as pd
import chromadb

# -----------------------------------
# 1. Read Excel
# -----------------------------------

excel_file = "/Users/gurkeeratsingh/Desktop/Indian_standards_ai/dairy.bis.xlsx"

df = pd.read_excel(
    excel_file,
    sheet_name="Standard_Chunks"
)

print("Rows loaded:", len(df))

# -----------------------------------
# 2. Connect to existing ChromaDB
# -----------------------------------

client = chromadb.PersistentClient(
    path="/Users/gurkeeratsingh/Desktop/Indian_standards_ai/dairy_vector_db"
)

# -----------------------------------
# 3. Create a NEW collection
# -----------------------------------

collection = client.get_or_create_collection(
    name="dairy_standards"
)

# -----------------------------------
# 4. Prepare documents
# -----------------------------------

documents = []
ids = []
metadatas = []

for _, row in df.iterrows():

    document = str(row["chunk_text"])

    documents.append(document)

    ids.append(str(row["chunk_id"]))

    metadatas.append({
        "standard_id": str(row["standard_id"]),
        "is_number": str(row["is_number"]),
        "section_title": str(row["section_title"]),
        "chunk_type": str(row["chunk_type"]),
        "source_document": str(row["source_document"]),
        "verification_status": str(row["verification_status"])
    })

# -----------------------------------
# 5. Upload to ChromaDB
# -----------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

# -----------------------------------
# 6. Confirm
# -----------------------------------

print("✅ Standards uploaded successfully!")
print("Number of documents:", collection.count())
