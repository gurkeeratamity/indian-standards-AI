import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from ingest import get_collection
from search import search
from requirement_extractor import extract_requirement, ExtractedRequirement

app = FastAPI()

collection = get_collection()

MAX_QUERY_LENGTH = 10000


def _validate_query_value(v: str) -> str:
    """Shared validation for any request model's 'query' field:
    strip whitespace, reject empty/whitespace-only, reject overly long input.
    Raising ValueError here causes FastAPI/Pydantic to return HTTP 422.
    """
    stripped = v.strip()
    if not stripped:
        raise ValueError("query must not be empty or whitespace-only")
    if len(stripped) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_QUERY_LENGTH} characters")
    return stripped


class SearchRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        return _validate_query_value(v)


class RecommendRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        return _validate_query_value(v)


class ExtractRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        return _validate_query_value(v)


NUMBER_PATTERN = re.compile(r'\d{3,6}')


def normalize_is_number(raw: str) -> str:
    """Extract the core numeric IS number from any common input format."""
    match = NUMBER_PATTERN.search(raw)
    if not match:
        return ""
    return match.group(0)


# ---------------------------------------------------------------------------
# Requirement-aware matching helpers (unchanged from STEP 4J.1).
#
# NOTE: this is still the naive substring matcher. The orientation/insulated/
# capacity conflict-handling improvements discussed in the prior analysis
# have NOT been implemented — they were analysis-only per that step's
# instructions. Nothing here has been "un-fixed"; there was no fix in code
# to preserve. This logic is retained as-is because this pass is scoped to
# validation + response shape only.
# ---------------------------------------------------------------------------

def _build_combined_text(supporting_chunks: list) -> str:
    parts = []
    for chunk in supporting_chunks:
        parts.append(str(chunk.get("section_title", "")))
        parts.append(str(chunk.get("matched_text", "")))
    return " ".join(parts).lower()


def _format_capacity(capacity: float) -> str:
    if float(capacity).is_integer():
        return str(int(capacity))
    return str(capacity)


def compute_requirement_match(extracted: ExtractedRequirement, combined_text: str):
    checklist = []

    if extracted.product:
        value = str(extracted.product).lower()
        checklist.append({"field": "product", "value": extracted.product,
                           "matched": value in combined_text})

    if extracted.category:
        value = str(extracted.category).lower()
        checklist.append({"field": "category", "value": extracted.category,
                           "matched": value in combined_text})

    if extracted.capacity is not None:
        value = _format_capacity(extracted.capacity)
        checklist.append({"field": "capacity", "value": extracted.capacity,
                           "matched": value in combined_text})

    if extracted.capacity_unit:
        value = str(extracted.capacity_unit).lower()
        checklist.append({"field": "capacity_unit", "value": extracted.capacity_unit,
                           "matched": value in combined_text})

    if extracted.material:
        value = str(extracted.material).lower()
        checklist.append({"field": "material", "value": extracted.material,
                           "matched": value in combined_text})

    if extracted.orientation:
        value = str(extracted.orientation).lower()
        checklist.append({"field": "orientation", "value": extracted.orientation,
                           "matched": value in combined_text})

    if extracted.insulated is True:
        checklist.append({"field": "insulated", "value": True,
                           "matched": "insulat" in combined_text})

    if extracted.application:
        value = str(extracted.application).lower()
        checklist.append({"field": "application", "value": extracted.application,
                           "matched": value in combined_text})

    if extracted.other_attributes:
        for key, val in extracted.other_attributes.items():
            if val is None or str(val).strip() == "":
                continue
            value = str(val).lower()
            checklist.append({"field": f"other_attributes.{key}", "value": val,
                               "matched": value in combined_text})

    if not checklist:
        return checklist, None

    matched_count = sum(1 for item in checklist if item["matched"])
    requirement_match_score = matched_count / len(checklist)
    return checklist, requirement_match_score


RETRIEVAL_WEIGHT = 0.6
REQUIREMENT_WEIGHT = 0.4


def compute_combined_score(retrieval_score: float, requirement_match_score):
    if requirement_match_score is None:
        return retrieval_score
    return (RETRIEVAL_WEIGHT * retrieval_score) + (REQUIREMENT_WEIGHT * requirement_match_score)


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
    try:
        extracted = extract_requirement(request.query)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = search(collection, request.query, n_results=10)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return {
            "query": request.query,
            "extracted_requirements": extracted.model_dump(),
            "recommendations": [],
        }

    groups = {}

    for doc, meta, distance in zip(documents, metadatas, distances):
        standard_id = meta.get("standard_id", "")
        is_number = meta.get("is_number", "")
        key = standard_id if standard_id else is_number

        similarity = 1 / (1 + distance)

        if key not in groups:
            groups[key] = {
                "standard_id": standard_id,
                "is_number": is_number,
                "raw_chunks": [],  # internal only, not returned as-is
            }

        groups[key]["raw_chunks"].append({
            "section_number": meta.get("section_number", ""),
            "section_title": meta.get("section_title", ""),
            "page_start": meta.get("page_start", ""),
            "page_end": meta.get("page_end", ""),
            "source_document": meta.get("source_document", ""),
            "verification_status": meta.get("verification_status", ""),
            "matched_text": doc,
            "distance": distance,
            "similarity": similarity,
        })

    recommendations = []
    for group in groups.values():
        raw_chunks = group.pop("raw_chunks")

        similarities = [c["similarity"] for c in raw_chunks]
        retrieval_score = sum(similarities) / len(similarities)

        # Total chunks retrieved for this standard among the top 10,
        # BEFORE capping/deduping the displayed evidence list below.
        total_supporting_chunks = len(raw_chunks)

        # Requirement matching still runs on the full (uncapped) chunk set
        # for this standard, using the existing naive substring matcher.
        combined_text = _build_combined_text(raw_chunks)
        _checklist, requirement_match_score = compute_requirement_match(extracted, combined_text)
        combined_score = compute_combined_score(retrieval_score, requirement_match_score)

        # --- Build cleaned, capped, deduped supporting evidence ---
        # Strongest relevance first (lowest distance = closest match).
        sorted_chunks = sorted(raw_chunks, key=lambda c: c["distance"])

        deduped_chunks = []
        seen_texts = set()
        for chunk in sorted_chunks:
            text_key = chunk["matched_text"]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            deduped_chunks.append(chunk)
            if len(deduped_chunks) == 3:
                break

        supporting_chunks = [
            {
                "section_number": c["section_number"],
                "section_title": c["section_title"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "source_document": c["source_document"],
                "verification_status": c["verification_status"],
                "matched_text": c["matched_text"],
            }
            for c in deduped_chunks
        ]

        recommendations.append({
            "standard_id": group["standard_id"],
            "is_number": group["is_number"],
            "retrieval_score": round(retrieval_score, 6),
            "number_of_supporting_chunks": total_supporting_chunks,
            "supporting_chunks": supporting_chunks,
            "_combined_score_for_sorting": combined_score,  # stripped before return
        })

    recommendations.sort(key=lambda r: r["_combined_score_for_sorting"], reverse=True)
    for r in recommendations:
        del r["_combined_score_for_sorting"]

    return {
        "query": request.query,
        "extracted_requirements": extracted.model_dump(),
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
    }


@app.post("/extract", response_model=ExtractedRequirement)
def extract_endpoint(request: ExtractRequest):
    try:
        return extract_requirement(request.query)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))