import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from ollama import list as ollama_list

from ingest import get_collection
from search import search
from requirement_extractor import extract_requirement, ExtractedRequirement
from rag import ask_question


app = FastAPI()

collection = get_collection()

MAX_QUERY_LENGTH = 10000

OLLAMA_MODEL = "llama3.2:3b"


def _validate_query_value(v: str) -> str:
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


class AskRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        return _validate_query_value(v)


NUMBER_PATTERN = re.compile(r'\d{3,6}')


def normalize_is_number(raw: str) -> str:
    match = NUMBER_PATTERN.search(raw)
    if not match:
        return ""
    return match.group(0)


# =============================================================================
# Requirement-aware matching (deterministic, no LLM).
#
# Every check below classifies a single extracted field into exactly one of:
#   "matched"       - retrieved evidence explicitly supports the requested value
#   "conflict"      - retrieved evidence explicitly supports a DIFFERENT/opposite
#                     value than what was requested
#   "not_verified"  - retrieved evidence says nothing usable either way
#
# A field with value None/null is never checked at all (handled by the
# calling code only adding non-null fields), so absent information can never
# count as a match, per the correctness rules.
# =============================================================================

def _extract_snippet(text: str, keyword: str, window: int = 60, max_length: int = 200):
    """Return a short excerpt of `text` centered on the first case-insensitive
    occurrence of `keyword`, collapsed to a single line. Returns None if not
    found. This is an exact excerpt of retrieved text — never generated or
    paraphrased."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return None
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    snippet = " ".join(text[start:end].split())
    if len(snippet) > max_length:
        snippet = snippet[:max_length].rstrip() + "..."
    return snippet


def _chunk_text(chunk: dict) -> str:
    return f'{chunk.get("section_title", "")} {chunk.get("matched_text", "")}'


def _find_first_match(chunks: list, keyword: str):
    """Search each chunk's own text individually (not one giant concatenated
    blob) so evidence stays tied to a single retrieved passage."""
    keyword_l = keyword.lower()
    for chunk in chunks:
        text = _chunk_text(chunk)
        if keyword_l in text.lower():
            return _extract_snippet(text, keyword_l)
    return None


def _check_substring_field(value: str, chunks: list):
    """Generic matched/not_verified check for fields with no reliable
    opposite-value set (product, category, capacity_unit, material,
    application, other_attributes). No conflict is ever claimed for these,
    since we have no safe way to know a mentioned alternative is truly
    exclusive rather than just contextually referenced."""
    evidence = _find_first_match(chunks, str(value))
    if evidence:
        return "matched", evidence
    return "not_verified", None


# --- Orientation: known binary opposite pair -------------------------------
ORIENTATION_OPPOSITES = {"horizontal": "vertical", "vertical": "horizontal"}


def _check_orientation(value: str, chunks: list):
    value_l = str(value).lower()
    opposite = ORIENTATION_OPPOSITES.get(value_l)

    # Check the opposite value FIRST. If the evidence explicitly states the
    # opposite orientation, that is a conflict regardless of whether the
    # requested word also happens to appear elsewhere (e.g. in a comparison).
    if opposite:
        evidence = _find_first_match(chunks, opposite)
        if evidence:
            return "conflict", evidence

    evidence = _find_first_match(chunks, value_l)
    if evidence:
        return "matched", evidence

    return "not_verified", None


# --- Insulated: boolean field, only the True case is ever checked ----------
INSULATION_NEGATION_PHRASES = [
    "not insulated", "non-insulated", "non insulated",
    "uninsulated", "without insulation", "shall not be insulated",
]


def _check_insulated(value: bool, chunks: list):
    # Only insulated=True reaches here (caller never checks insulated=False —
    # there is no reliable way to confirm "explicitly not insulated" was the
    # user's actual requirement without risking a false conflict claim).
    for phrase in INSULATION_NEGATION_PHRASES:
        evidence = _find_first_match(chunks, phrase)
        if evidence:
            return "conflict", evidence

    evidence = _find_first_match(chunks, "insulat")
    if evidence:
        return "matched", evidence

    return "not_verified", None


# --- Capacity: numeric + unit aware -----------------------------------------
CAPACITY_UNIT_SYNONYMS = {
    "litre": ["litre", "litres", "liter", "liters", "l"],
    "kilogram": ["kilogram", "kilograms", "kg"],
    "kilogramme": ["kilogramme", "kilogrammes", "kg"],
    "gram": ["gram", "grams", "g"],
    "millimetre": ["millimetre", "millimetres", "millimeter", "millimeters", "mm"],
    "metre": ["metre", "metres", "meter", "meters", "m"],
    "tonne": ["tonne", "tonnes", "ton", "tons", "mt"],
}


def _unit_variants(unit):
    if not unit:
        return None
    return CAPACITY_UNIT_SYNONYMS.get(unit.strip().lower(), [unit.strip().lower()])


def _check_capacity(capacity_value: float, capacity_unit, chunks: list):
    """
    Deterministic rule:
    - Search each chunk for "<number> <unit-or-known-synonym>" patterns.
    - If any occurrence's number equals the requested capacity (within a
      small float tolerance) -> "matched".
    - Else, if the standard explicitly states a DIFFERENT number with the
      same unit -> "conflict" (it specifies a different capacity value).
    - Else -> "not_verified" (no capacity+unit figure found at all).

    KNOWN LIMITATION: a standard covering a capacity RANGE that includes the
    requested value, but is only stated as boundary numbers (e.g. "1000 to
    10000 litre"), may be incorrectly flagged as "conflict" here, since this
    check has no range-parsing logic. This is a deterministic, non-LLM
    substring/regex approach and cannot resolve that ambiguity; flagged
    here rather than silently guessed.

    Without a capacity_unit, no search is attempted (a bare number is too
    likely to collide with unrelated figures, page numbers, clause numbers).
    """
    variants = _unit_variants(capacity_unit)
    if not variants:
        return "not_verified", None

    pattern = re.compile(
        r'(\d[\d,]*\.?\d*)\s*(?:' + '|'.join(re.escape(v) for v in variants) + r')\b',
        re.IGNORECASE,
    )

    matched_evidence = None
    conflict_evidence = None

    for chunk in chunks:
        text = _chunk_text(chunk)
        for m in pattern.finditer(text):
            num_str = m.group(1).replace(",", "")
            try:
                num = float(num_str)
            except ValueError:
                continue

            if abs(num - float(capacity_value)) < 0.01:
                if matched_evidence is None:
                    matched_evidence = _extract_snippet(text, m.group(0))
            else:
                if conflict_evidence is None:
                    conflict_evidence = _extract_snippet(text, m.group(0))

    if matched_evidence:
        return "matched", matched_evidence
    if conflict_evidence:
        return "conflict", conflict_evidence
    return "not_verified", None


def compute_requirement_match_details(extracted: ExtractedRequirement, chunks: list):
    """Build the full per-field checklist for one standard's retrieved
    chunks. Only non-null fields are ever added — absence of information
    in the query is never treated as a requirement to check."""
    checklist = []

    def add(field, value, status, evidence):
        checklist.append({"field": field, "value": value, "status": status, "evidence": evidence})

    if extracted.product:
        status, evidence = _check_substring_field(extracted.product, chunks)
        add("product", extracted.product, status, evidence)

    if extracted.category:
        status, evidence = _check_substring_field(extracted.category, chunks)
        add("category", extracted.category, status, evidence)

    if extracted.capacity is not None:
        status, evidence = _check_capacity(extracted.capacity, extracted.capacity_unit, chunks)
        add("capacity", extracted.capacity, status, evidence)

    if extracted.capacity_unit:
        status, evidence = _check_substring_field(extracted.capacity_unit, chunks)
        add("capacity_unit", extracted.capacity_unit, status, evidence)

    if extracted.material:
        status, evidence = _check_substring_field(extracted.material, chunks)
        add("material", extracted.material, status, evidence)

    if extracted.orientation:
        status, evidence = _check_orientation(extracted.orientation, chunks)
        add("orientation", extracted.orientation, status, evidence)

    if extracted.insulated is True:
        status, evidence = _check_insulated(True, chunks)
        add("insulated", True, status, evidence)

    if extracted.application:
        status, evidence = _check_substring_field(extracted.application, chunks)
        add("application", extracted.application, status, evidence)

    if extracted.other_attributes:
        for key, val in extracted.other_attributes.items():
            if val is None or str(val).strip() == "":
                continue
            status, evidence = _check_substring_field(str(val), chunks)
            add(f"other_attributes.{key}", val, status, evidence)

    return checklist


RETRIEVAL_WEIGHT = 0.6
REQUIREMENT_WEIGHT = 0.4


def compute_combined_score(retrieval_score: float, requirement_match_score):
    """Unchanged formula: weighted average of retrieval_score and
    requirement_match_score, falling back to pure retrieval_score when no
    fields were available to check. Not an AI confidence value — a
    documented arithmetic combination of two transparent numbers."""
    if requirement_match_score is None:
        return retrieval_score
    return (RETRIEVAL_WEIGHT * retrieval_score) + (REQUIREMENT_WEIGHT * requirement_match_score)


# =============================================================================
# Standard classification (deterministic, no LLM).
#
# Rules (checked in this order):
#   1. Any conflicting field at all           -> "weak_candidate"
#   2. Weak retrieval relevance (< 0.3)       -> "weak_candidate"
#   3. Strong retrieval (>= 0.5) AND at least
#      2 matched requirement fields           -> "primary"
#   4. Everything else                        -> "related"
#
# This never claims legal applicability — "primary" means "strongest
# candidate given retrieval + matched requirements", not "the correct
# mandatory standard". Thresholds are fixed constants, not learned.
# =============================================================================

RETRIEVAL_STRONG_THRESHOLD = 0.5
RETRIEVAL_WEAK_THRESHOLD = 0.3
MIN_MATCHED_FOR_PRIMARY = 2


def classify_recommendation(retrieval_score: float, matched_count: int, conflict_count: int) -> str:
    if conflict_count > 0:
        return "weak_candidate"
    if retrieval_score < RETRIEVAL_WEAK_THRESHOLD:
        return "weak_candidate"
    if retrieval_score >= RETRIEVAL_STRONG_THRESHOLD and matched_count >= MIN_MATCHED_FOR_PRIMARY:
        return "primary"
    return "related"


@app.get("/health")
def health():
    chromadb_status = "ok"
    ollama_status = "ok"

    # Check that the existing ChromaDB collection object is accessible.
    # This does NOT create another ChromaDB connection.
    try:
        collection.count()
    except Exception:
        chromadb_status = "error"

    # Check that Ollama is reachable and that the configured model exists.
    try:
        response = ollama_list()

        models = getattr(response, "models", [])

        model_available = any(
            getattr(model, "model", "") == OLLAMA_MODEL
            for model in models
        )

        if not model_available:
            ollama_status = "error"

    except Exception:
        ollama_status = "error"

    if chromadb_status == "ok" and ollama_status == "ok":
        return {
            "status": "ok",
            "chromadb": "ok",
            "ollama": "ok"
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "chromadb": chromadb_status,
            "ollama": ollama_status
        }
    )


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    try:
        return ask_question(request.query)
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG generation failed: {e}"
        )


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
                "raw_chunks": [],
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

        # Requirement matching runs against the FULL retrieved chunk set for
        # this standard (before capping/deduping the displayed evidence).
        requirement_match_details = compute_requirement_match_details(extracted, raw_chunks)

        matched_fields = [c["field"] for c in requirement_match_details if c["status"] == "matched"]
        conflicting_fields = [c["field"] for c in requirement_match_details if c["status"] == "conflict"]
        unverified_fields = [c["field"] for c in requirement_match_details if c["status"] == "not_verified"]

        total_checked = len(requirement_match_details)
        requirement_match_score = (
            len(matched_fields) / total_checked if total_checked else None
        )

        combined_score = compute_combined_score(retrieval_score, requirement_match_score)

        recommendation_type = classify_recommendation(
            retrieval_score=retrieval_score,
            matched_count=len(matched_fields),
            conflict_count=len(conflicting_fields),
        )

        # --- Build cleaned, capped, deduped supporting evidence ---
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
            "recommendation_type": recommendation_type,
            "retrieval_score": round(retrieval_score, 6),
            "requirement_match_score": (
                round(requirement_match_score, 6) if requirement_match_score is not None else None
            ),
            "combined_score": round(combined_score, 6),
            "score_explanation": {
                "matched_requirements": len(matched_fields),
                "conflicting_requirements": len(conflicting_fields),
                "unverified_requirements": len(unverified_fields),
                "matched_fields": matched_fields,
                "conflicting_fields": conflicting_fields,
            },
            "requirement_match_details": requirement_match_details,
            "supporting_chunks": supporting_chunks,
            "_combined_score_for_sorting": combined_score,
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