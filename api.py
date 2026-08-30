import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import get_collection
from search import search

app = FastAPI()

collection = get_collection()


class SearchRequest(BaseModel):
    query: str


class RecommendRequest(BaseModel):
    query: str


NUMBER_PATTERN = re.compile(r'\d{3,6}')


def normalize_is_number(raw: str) -> str:
    """Extract the core numeric IS number from any common input format."""
    match = NUMBER_PATTERN.search(raw)
    if not match:
        return ""
    return match.group(0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(request: SearchRequest):
    results = search(collection, request.query, n_results=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    response = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        response.append({
            "rank": i,
            "is_number": meta.get("is_number", ""),
            "standard_id": meta.get("standard_id", ""),
            "section_number": meta.get("section_number", ""),
            "section_title": meta.get("section_title", ""),
            "page_start": meta.get("page_start", ""),
            "page_end": meta.get("page_end", ""),
            "source_document": meta.get("source_document", ""),
            "verification_status": meta.get("verification_status", ""),
            "matched_text": doc,
        })

    return {"results": response}


@app.get("/standards/{is_number}")
def get_standard(is_number: str):
    target_number = normalize_is_number(is_number)

    if not target_number:
        raise HTTPException(
            status_code=404,
            detail=f"Could not parse an IS number from '{is_number}'",
        )

    all_records = collection.get(include=["documents", "metadatas"])

    documents = all_records["documents"]
    metadatas = all_records["metadatas"]

    chunks = []
    for doc, meta in zip(documents, metadatas):
        stored_number = normalize_is_number(meta.get("is_number", ""))
        if stored_number == target_number:
            chunks.append({
                "is_number": meta.get("is_number", ""),
                "standard_id": meta.get("standard_id", ""),
                "section_number": meta.get("section_number", ""),
                "section_title": meta.get("section_title", ""),
                "page_start": meta.get("page_start", ""),
                "page_end": meta.get("page_end", ""),
                "source_document": meta.get("source_document", ""),
                "verification_status": meta.get("verification_status", ""),
                "matched_text": doc,
            })

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No standard found matching IS number '{target_number}' (parsed from '{is_number}')",
        )

    versions_found = sorted(set(c["is_number"] for c in chunks))

    return {
        "queried_input": is_number,
        "matched_is_number": target_number,
        "versions_found": versions_found,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.post("/recommend")
def recommend_endpoint(request: RecommendRequest):
    results = search(collection, request.query, n_results=10)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return {"query": request.query, "recommendations": []}

    groups = {}

    for doc, meta, distance in zip(documents, metadatas, distances):
        standard_id = meta.get("standard_id", "")
        is_number = meta.get("is_number", "")
        key = standard_id if standard_id else is_number

        # Transparent similarity score derived from vector distance.
        # This is NOT an AI-generated confidence score — it is a plain
        # numeric transform of ChromaDB's returned distance for this chunk.
        similarity = 1 / (1 + distance)

        if key not in groups:
            groups[key] = {
                "standard_id": standard_id,
                "is_number": is_number,
                "supporting_chunks": [],
                "similarities": [],
            }

        groups[key]["supporting_chunks"].append({
            "section_number": meta.get("section_number", ""),
            "section_title": meta.get("section_title", ""),
            "page_start": meta.get("page_start", ""),
            "page_end": meta.get("page_end", ""),
            "source_document": meta.get("source_document", ""),
            "verification_status": meta.get("verification_status", ""),
            "matched_text": doc,
            "distance": distance,
            "similarity": round(similarity, 6),
        })
        groups[key]["similarities"].append(similarity)

    recommendations = []
    for group in groups.values():
        similarities = group.pop("similarities")
        # Standard-level score: average similarity across its retrieved
        # supporting chunks. Transparent aggregation, not a model output.
        score = sum(similarities) / len(similarities)
        group["score"] = round(score, 6)
        recommendations.append(group)

    recommendations.sort(key=lambda r: r["score"], reverse=True)

    return {
        "query": request.query,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
    }