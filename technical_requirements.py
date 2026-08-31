"""
technical_requirements.py

Structured technical-requirement model and deterministic (non-LLM)
matching helpers for the Indian Standards recommendation engine.

This module is intentionally independent of FastAPI and ChromaDB. It only
defines:
  - a Pydantic model for a single technical requirement
    (e.g. "Moisture <= 60 %", "Capacity = 5000 litre")
  - helper functions to normalize units (safe/trivial synonyms only,
    no unit conversion between different scales)
  - helper functions to compare a requirement against evidence text and
    classify the result as "matched", "conflict", or "not_verified"

No LLM calls. No guessing of missing values. No unit conversion beyond
recognizing that "litre"/"litres"/"l" (etc.) refer to the same unit.
"""

import re
from typing import Optional, Union, List
from pydantic import BaseModel, field_validator, model_validator


# =============================================================================
# Model
# =============================================================================

SUPPORTED_OPERATORS = ("=", "<=", ">=", "<", ">")


class TechnicalRequirement(BaseModel):
    """
    A single technical requirement extracted from a procurement query.

    Examples:
      TechnicalRequirement(parameter="moisture", operator="<=", value=60, unit="%")
      TechnicalRequirement(parameter="capacity", operator="=", value=5000, unit="litre")
      TechnicalRequirement(parameter="material", operator="=", value="stainless steel")
    """
    parameter: str
    operator: str
    # Union tries float first: numeric-looking values (including numeric
    # strings like "60") become float; anything non-numeric stays a string.
    value: Union[float, str]
    unit: Optional[str] = None

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v):
        if v not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported operator '{v}'. Must be one of {SUPPORTED_OPERATORS}"
            )
        return v

    @model_validator(mode="after")
    def validate_operator_matches_value_type(self):
        # Comparison operators other than "=" only make sense for numeric
        # values. A textual requirement (e.g. material) can only be an
        # equality check — we don't guess what "material >= steel" would mean.
        if self.operator != "=" and not isinstance(self.value, (int, float)):
            raise ValueError(
                f"Operator '{self.operator}' requires a numeric value; "
                f"got textual value '{self.value}'. Only '=' is valid for text."
            )
        return self


# =============================================================================
# Unit normalization — SAFE/TRIVIAL SYNONYMS ONLY.
#
# This maps different spellings/abbreviations of the SAME unit to one
# canonical name. It does NOT convert between different units (e.g. it will
# never turn millimetres into centimetres) — that would require a
# conversion factor and risks silently changing the meaning of a requirement.
# =============================================================================

UNIT_SYNONYMS = {
    "litre": ["litre", "litres", "liter", "liters", "l"],
    "kilogram": ["kilogram", "kilograms", "kilogramme", "kilogrammes", "kg"],
    "gram": ["gram", "grams", "g"],
    "millimetre": ["millimetre", "millimetres", "millimeter", "millimeters", "mm"],
    "centimetre": ["centimetre", "centimetres", "centimeter", "centimeters", "cm"],
    "metre": ["metre", "metres", "meter", "meters", "m"],
    "tonne": ["tonne", "tonnes", "ton", "tons", "mt"],
    "percent": ["%", "percent", "per cent", "pct"],
    "celsius": ["°c", "degc", "deg c", "celsius", "centigrade"],
}

_VARIANT_TO_CANONICAL = {
    variant.lower(): canonical
    for canonical, variants in UNIT_SYNONYMS.items()
    for variant in variants
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """
    Map a unit string to its canonical form if it's a known safe synonym.
    Unknown units are returned lowercased/stripped, unchanged otherwise.
    Returns None if input is None/empty.

    Examples:
      normalize_unit("litres") -> "litre"
      normalize_unit("L")      -> "litre"
      normalize_unit("°C")     -> "celsius"
      normalize_unit("furlong") -> "furlong"   (unknown, passed through)
    """
    if not unit:
        return None
    cleaned = unit.strip().lower()
    return _VARIANT_TO_CANONICAL.get(cleaned, cleaned)


def _unit_variants(unit: Optional[str]) -> Optional[List[str]]:
    """Return all known textual variants for a unit (for regex searching),
    or a single-item list of the raw unit if it's not a recognized synonym."""
    if not unit:
        return None
    canonical = normalize_unit(unit)
    return UNIT_SYNONYMS.get(canonical, [unit.strip().lower()])


# =============================================================================
# Numeric comparison
# =============================================================================

_OPERATOR_FUNCS = {
    "=": lambda observed, target: abs(observed - target) < 1e-9,
    "<=": lambda observed, target: observed <= target,
    ">=": lambda observed, target: observed >= target,
    "<": lambda observed, target: observed < target,
    ">": lambda observed, target: observed > target,
}


def compare_numeric(observed_value: float, operator: str, target_value: float) -> bool:
    """
    Evaluate whether `observed_value <operator> target_value` holds.

    Example: compare_numeric(65, "<=", 60) -> False
             compare_numeric(45, "<=", 60) -> True
    """
    if operator not in _OPERATOR_FUNCS:
        raise ValueError(f"Unsupported operator '{operator}'")
    return _OPERATOR_FUNCS[operator](observed_value, target_value)


# =============================================================================
# Evidence extraction (no invention — only reads numbers/text actually
# present in the supplied evidence text).
# =============================================================================

def _find_numeric_observations(evidence_text: str, unit: Optional[str]):
    """
    Find every (number, matched_substring) occurrence in evidence_text of a
    number immediately followed by a variant of `unit`. Returns a list of
    (float_value, snippet) tuples, in order of appearance.

    Without a unit, no search is attempted — a bare number is too likely to
    collide with unrelated figures (page numbers, clause numbers, etc.).
    """
    variants = _unit_variants(unit)
    if not variants:
        return []

    unit_pattern = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    pattern = re.compile(
        r'(\d[\d,]*\.?\d*)\s*(?:' + unit_pattern + r')\b',
        re.IGNORECASE,
    )

    observations = []
    for m in pattern.finditer(evidence_text):
        num_str = m.group(1).replace(",", "")
        try:
            num = float(num_str)
        except ValueError:
            continue
        observations.append((num, m.group(0)))
    return observations


def _snippet_around(evidence_text: str, needle: str, window: int = 60, max_length: int = 200) -> str:
    """Exact excerpt of evidence_text around the first occurrence of needle.
    Never generated or paraphrased — only trimmed/whitespace-collapsed."""
    idx = evidence_text.lower().find(needle.lower())
    if idx == -1:
        return evidence_text[:max_length]
    start = max(0, idx - window)
    end = min(len(evidence_text), idx + len(needle) + window)
    snippet = " ".join(evidence_text[start:end].split())
    if len(snippet) > max_length:
        snippet = snippet[:max_length].rstrip() + "..."
    return snippet


def _find_textual_observation(evidence_text: str, value: str):
    """Return an exact excerpt if `value` appears as a substring of
    evidence_text (case-insensitive), else None."""
    if value.lower() in evidence_text.lower():
        return _snippet_around(evidence_text, value)
    return None


# =============================================================================
# Classification: matched / conflict / not_verified
# =============================================================================

def evaluate_requirement(
    requirement: TechnicalRequirement,
    evidence_text: str,
    conflicting_values: Optional[List[str]] = None,
):
    """
    Classify a single TechnicalRequirement against evidence_text into
    exactly one of: "matched", "conflict", "not_verified".

    NUMERIC requirements (operator in <=, >=, <, >, or = with a numeric value):
      - Search evidence_text for numbers followed by a matching unit.
      - If no such number is found at all -> "not_verified".
      - If any found number SATISFIES the requirement's condition
        (compare_numeric(found, operator, requirement.value) is True)
        -> "matched", evidence = that number's excerpt.
      - Otherwise (a number for this parameter/unit was found, but it does
        NOT satisfy the condition) -> "conflict", evidence = that number's
        excerpt.
      KNOWN LIMITATION: this does not understand ranges (e.g. "1000 to
      10000 litre") or which of several found numbers is the one that
      actually applies — it is a deterministic substring/regex check, not
      semantic understanding. Ambiguous cases favor citing the first
      relevant number found rather than guessing which is authoritative.

    TEXTUAL requirements (operator "=" with a non-numeric value):
      - If requirement.value appears as a substring of evidence_text
        -> "matched".
      - Else, if `conflicting_values` is provided (known mutually-exclusive
        alternatives, e.g. ["vertical"] for a "horizontal" requirement) and
        one of them appears in evidence_text -> "conflict".
      - Else -> "not_verified".
      No conflict is ever claimed for textual values unless a known
      alternative was explicitly supplied by the caller — this module does
      not guess what counts as "the opposite" of an arbitrary text value.

    Returns:
      {"status": "matched" | "conflict" | "not_verified", "evidence": str | None}
    """
    is_numeric = isinstance(requirement.value, (int, float))

    if is_numeric:
        observations = _find_numeric_observations(evidence_text, requirement.unit)
        if not observations:
            return {"status": "not_verified", "evidence": None}

        matched_evidence = None
        conflict_evidence = None
        for observed_value, snippet_source in observations:
            satisfies = compare_numeric(observed_value, requirement.operator, requirement.value)
            excerpt = _snippet_around(evidence_text, snippet_source)
            if satisfies and matched_evidence is None:
                matched_evidence = excerpt
            elif not satisfies and conflict_evidence is None:
                conflict_evidence = excerpt

        if matched_evidence:
            return {"status": "matched", "evidence": matched_evidence}
        return {"status": "conflict", "evidence": conflict_evidence}

    # Textual requirement
    evidence = _find_textual_observation(evidence_text, requirement.value)
    if evidence:
        return {"status": "matched", "evidence": evidence}

    if conflicting_values:
        for alt in conflicting_values:
            evidence = _find_textual_observation(evidence_text, alt)
            if evidence:
                return {"status": "conflict", "evidence": evidence}

    return {"status": "not_verified", "evidence": None}


if __name__ == "__main__":
    print("--- Numeric: matched ---")
    req = TechnicalRequirement(parameter="moisture", operator="<=", value=60, unit="%")
    evidence = "The moisture content of paneer shall not exceed 55 percent by mass."
    print(req.model_dump())
    print(evaluate_requirement(req, evidence))

    print("\n--- Numeric: conflict ---")
    req = TechnicalRequirement(parameter="moisture", operator="<=", value=60, unit="%")
    evidence = "The moisture content of paneer shall not exceed 65 percent by mass."
    print(evaluate_requirement(req, evidence))

    print("\n--- Numeric: not_verified ---")
    req = TechnicalRequirement(parameter="thickness", operator=">=", value=2, unit="mm")
    evidence = "This standard covers general requirements for milk cans."
    print(evaluate_requirement(req, evidence))

    print("\n--- Textual: matched ---")
    req = TechnicalRequirement(parameter="material", operator="=", value="stainless steel")
    evidence = "The tank shall be constructed of stainless steel conforming to grade 304."
    print(evaluate_requirement(req, evidence))

    print("\n--- Textual: conflict (with known alternative) ---")
    req = TechnicalRequirement(parameter="orientation", operator="=", value="horizontal")
    evidence = "This standard applies to vertical cylindrical milk storage tanks."
    print(evaluate_requirement(req, evidence, conflicting_values=["vertical"]))

    print("\n--- Textual: not_verified ---")
    req = TechnicalRequirement(parameter="orientation", operator="=", value="horizontal")
    evidence = "This standard specifies material and welding requirements only."
    print(evaluate_requirement(req, evidence, conflicting_values=["vertical"]))

    print("\n--- Unit normalization examples ---")
    for u in ["litres", "L", "°C", "kg", "furlong"]:
        print(f"{u!r} -> {normalize_unit(u)!r}")