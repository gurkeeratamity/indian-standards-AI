from ollama import chat

from ingest import get_collection
from search import search


collection = get_collection()

MODEL_NAME = "llama3.2:3b"

# Simple transparent retrieval threshold.
# Uses the same distance -> similarity conversion already used
# by the existing /recommend endpoint.
RETRIEVAL_MIN_SIMILARITY = 0.30

NO_EVIDENCE_MESSAGE = (
    "I could not find sufficient information in the available BIS standards "
    "to answer this question."
)


SYSTEM_PROMPT = """
You are an AI assistant for Indian dairy standards.

Answer the user's question using ONLY retrieved BIS standard evidence
that directly supports the question.

Rules:
1. Use ONLY retrieved BIS evidence that directly supports the user's question.
2. Do NOT combine unrelated retrieved chunks merely because they came from the retrieval results.
3. Do NOT assume that every retrieved source is relevant.
4. Do NOT infer missing requirements, values, limits, dates, specifications, or standards from general knowledge.
5. If the retrieved evidence does not provide enough information to answer the question confidently, respond:
   "I could not find sufficient information in the available BIS standards to answer this question."
6. Never invent a citation, IS number, section, page number, requirement, numerical value, or regulatory claim.
7. When answering, prefer the most directly relevant standard/section over loosely related retrieved material.
8. Treat only directly relevant retrieved context as authoritative evidence.
9. Prefer precise answers over lengthy explanations.
10. When supported by the retrieved evidence, cite the relevant IS number and section.
11. Do NOT cite a source unless the retrieved evidence actually supports the statement.
"""


def _distance_to_similarity(distance):
    """
    Convert Chroma distance into the same simple similarity representation
    already used by the existing /recommend endpoint.
    """
    return 1 / (1 + distance)


def _get_metadata(meta):
    """
    Preserve the BIS source metadata stored in ChromaDB.
    """
    return {
        "is_number": meta.get("is_number", ""),
        "standard_id": meta.get("standard_id", ""),
        "section_number": meta.get("section_number", ""),
        "section_title": meta.get("section_title", ""),
        "page_start": meta.get("page_start", ""),
        "page_end": meta.get("page_end", ""),
        "source_document": meta.get("source_document", ""),
        "verification_status": meta.get("verification_status", ""),
    }


def _filter_and_deduplicate(documents, metadatas, distances):
    """
    Keep only sufficiently relevant retrieved evidence and remove
    duplicate chunks.

    The returned chunks are exactly the chunks that may be supplied
    to Ollama as usable context.
    """
    filtered = []
    seen_texts = set()

    for doc, meta, distance in zip(documents, metadatas, distances):
        if not doc:
            continue

        similarity = _distance_to_similarity(distance)

        if similarity < RETRIEVAL_MIN_SIMILARITY:
            continue

        text_key = doc.strip()

        if text_key in seen_texts:
            continue

        seen_texts.add(text_key)

        filtered.append({
            "matched_text": doc,
            "metadata": _get_metadata(meta),
        })

    return filtered


def ask_question(query: str, n_results: int = 5):
    """
    Retrieve BIS evidence, filter and deduplicate it, then answer using
    only the evidence actually supplied to Ollama.
    """
    try:
        results = search(collection, query, n_results=n_results)
    except Exception as e:
        raise RuntimeError(f"BIS retrieval failed: {e}") from e

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": []
        }

    filtered_chunks = _filter_and_deduplicate(
        documents,
        metadatas,
        distances
    )

    # Nothing usable remains, so do not call Ollama.
    if not filtered_chunks:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "sources": []
        }

    context_parts = []
    sources = []

    # Build BOTH the LLM context and API sources from the exact same
    # filtered/deduplicated evidence.
    for chunk in filtered_chunks:
        metadata = chunk["metadata"]
        doc = chunk["matched_text"]

        context_parts.append(
            f"""
SOURCE:
IS Number: {metadata["is_number"]}
Standard ID: {metadata["standard_id"]}
Section: {metadata["section_number"]} {metadata["section_title"]}
Pages: {metadata["page_start"]} - {metadata["page_end"]}
Document: {metadata["source_document"]}
Verification Status: {metadata["verification_status"]}

TEXT:
{doc}
"""
        )

        sources.append({
            "is_number": metadata["is_number"],
            "standard_id": metadata["standard_id"],
            "section_number": metadata["section_number"],
            "section_title": metadata["section_title"],
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            "source_document": metadata["source_document"],
            "verification_status": metadata["verification_status"],
        })

    context = "\n".join(context_parts)

    user_prompt = f"""
Answer this question:

{query}

Use the following BIS standard evidence:

{context}
"""

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama generation failed using model '{MODEL_NAME}': {e}"
        ) from e

    try:
        answer = response["message"]["content"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "Ollama returned an unexpected response format."
        )

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    return {
        "answer": answer.strip(),
        "sources": sources
    }


if __name__ == "__main__":
    question = input("Enter question: ").strip()

    if question:
        result = ask_question(question)

        print("\n--- ANSWER ---")
        print(result["answer"])

        print("\n--- SOURCES ---")
        for source in result["sources"]:
            print(
                f'{source["is_number"]} | '
                f'{source["section_number"]} {source["section_title"]} | '
                f'Pages {source["page_start"]}-{source["page_end"]}'
            )