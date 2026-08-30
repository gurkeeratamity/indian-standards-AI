"""
requirement_extractor.py

Converts a natural-language procurement requirement into structured
JSON using an LLM (Google Gemini). This module is intentionally
independent of FastAPI and ChromaDB — it only does one thing: text ->
structured fields.
"""

import os
import json
from typing import Optional

from pydantic import BaseModel
from google import genai
from google.genai import types


class ExtractedRequirement(BaseModel):
    product: Optional[str] = None
    category: Optional[str] = None
    capacity: Optional[float] = None
    capacity_unit: Optional[str] = None
    material: Optional[str] = None
    orientation: Optional[str] = None
    insulated: Optional[bool] = None
    application: Optional[str] = None
    other_attributes: Optional[dict] = None


SYSTEM_PROMPT = """You extract structured procurement information from a single \
natural-language requirement describing a product to be procured.

Return ONLY a JSON object with exactly these fields:
- product (string or null)
- category (string or null)
- capacity (number or null)
- capacity_unit (string or null)
- material (string or null)
- orientation (string or null)
- insulated (boolean or null)
- application (string or null)
- other_attributes (object or null) — any additional descriptive attributes \
mentioned in the text that don't fit the fields above, as key-value pairs

Strict rules:
- Only extract information that is explicitly present or directly stated in \
the input text.
- Do NOT infer, guess, assume, or add information that is not present in the \
text, even if it seems technically likely.
- If a field is not mentioned in the text, its value MUST be null (or {} for \
other_attributes if nothing else applies — use null if truly nothing extra).
- Do not include any explanation, commentary, or text outside the JSON object.
- Do not wrap the JSON in markdown code fences.
"""

MODEL_NAME = "gemini-3.6-flash"


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before calling extract_requirement()."
        )
    return genai.Client(api_key=api_key)


def extract_requirement(text: str) -> ExtractedRequirement:
    """
    Convert a natural-language procurement requirement string into an
    ExtractedRequirement object using an LLM.

    Does NOT search ChromaDB or Indian Standards. Pure text -> structured JSON.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=text.strip(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )

    raw_text = (response.text or "").strip()

    # Defensive cleanup in case the model wraps output in code fences
    # despite instructions not to.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_text!r}"
        ) from e

    return ExtractedRequirement(**parsed)


if __name__ == "__main__":
    sample_input = "5000 litre horizontal insulated stainless steel milk storage tank"
    result = extract_requirement(sample_input)
    print(json.dumps(result.model_dump(), indent=2))