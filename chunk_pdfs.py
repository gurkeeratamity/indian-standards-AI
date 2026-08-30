import sys
import json
import re
import hashlib
from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

HEADING_PATTERN = re.compile(r'^(\d{1,2}(?:\.\d{1,3}){0,3})\s+([A-Z][A-Z0-9 ,\-/&\(\)]{2,})\s*$')
ANNEX_PATTERN = re.compile(r'^(ANNEX\s+[A-Z0-9]+)\b\s*(.*)$', re.IGNORECASE)
TABLE_PATTERN = re.compile(r'^(TABLE\s+\d+[A-Z]?)\b\s*(.*)$', re.IGNORECASE)
FILENAME_PATTERN = re.compile(r'(\d{3,6})[_\-](\d{4})')
TEXT_IS_PATTERN = re.compile(r'IS\s+(\d{3,6})\s*[:\-]?\s*(\d{4})?')


def detect_standard_info(source_document, full_text):
    match = FILENAME_PATTERN.search(Path(source_document).stem)
    if match:
        number, year = match.group(1), match.group(2)
        return f"IS {number} : {year}", f"{number}_{year}"

    match = TEXT_IS_PATTERN.search(full_text)
    if match:
        number = match.group(1)
        year = match.group(2) or ""
        is_number = f"IS {number} : {year}".strip(" :")
        standard_id = f"{number}_{year}" if year else number
        return is_number, standard_id

    return "", ""


def classify_chunk_type(section_title, chunk_text, is_table):
    title = (section_title or "").upper()
    text = chunk_text.upper()

    if is_table or "TABLE" in title:
        return "table"
    if "ANNEX" in title:
        return "annex"
    if "SCOPE" in title:
        return "scope"
    if "DEFINITION" in title or "TERMINOLOGY" in title:
        return "definition"
    if "METHOD OF TEST" in title or "TEST METHOD" in title or "TESTING" in title:
        return "test_method"
    if "SHALL" in text:
        return "requirement"
    if section_title:
        return "paragraph"
    return "other"


def build_lines(pages):
    lines = []
    for page in pages:
        page_number = page.get("page")
        text = page.get("text", "")
        for line in text.split("\n"):
            lines.append((page_number, line))
    return lines


def detect_heading(line):
    stripped = line.strip()
    if not stripped:
        return None

    table_match = TABLE_PATTERN.match(stripped)
    if table_match:
        number = table_match.group(1).strip()
        rest = table_match.group(2).strip()
        title = f"{number} {rest}".strip()
        return number, title, True

    annex_match = ANNEX_PATTERN.match(stripped)
    if annex_match:
        number = annex_match.group(1).strip()
        rest = annex_match.group(2).strip()
        title = f"{number} {rest}".strip()
        return number, title, False

    heading_match = HEADING_PATTERN.match(stripped)
    if heading_match:
        number = heading_match.group(1).strip()
        title = heading_match.group(2).strip()
        return number, title, False

    return None


def build_chunks(pages, source_document, is_number, standard_id):
    lines = build_lines(pages)

    raw_chunks = []
    current_section_number = ""
    current_section_title = ""
    current_is_table = False
    current_lines = []
    current_pages = []

    def flush_chunk():
        if not current_lines:
            return
        chunk_text = "\n".join(current_lines).strip("\n")
        if not chunk_text.strip():
            return
        page_start = min(current_pages)
        page_end = max(current_pages)
        chunk_type = classify_chunk_type(current_section_title, chunk_text, current_is_table)
        raw_chunks.append({
            "section_number": current_section_number,
            "section_title": current_section_title,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_text": chunk_text,
            "chunk_type": chunk_type,
        })

    for page_number, line in lines:
        heading = detect_heading(line)
        if heading:
            flush_chunk()
            current_section_number, current_section_title, current_is_table = heading
            current_lines = [line]
            current_pages = [page_number]
        else:
            current_lines.append(line)
            current_pages.append(page_number)

    flush_chunk()

    final_chunks = []
    for i, chunk in enumerate(raw_chunks, start=1):
        raw_id = f"{source_document}|{chunk['section_number']}|{chunk['page_start']}|{i}"
        chunk_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:16]
        final_chunks.append({
            "chunk_id": chunk_id,
            "standard_id": standard_id,
            "is_number": is_number,
            "section_number": chunk["section_number"],
            "section_title": chunk["section_title"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_text": chunk["chunk_text"],
            "chunk_type": chunk["chunk_type"],
            "source_document": source_document,
        })

    return final_chunks


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"ERROR: Input folder '{INPUT_DIR}' does not exist. Run python3 extract_pdfs.py first.")

    json_files = sorted(f for f in INPUT_DIR.glob("*.json"))
    if not json_files:
        sys.exit(f"ERROR: No JSON files found in '{INPUT_DIR}'. Run python3 extract_pdfs.py first.")

    all_chunks = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_document = data.get("source_document", json_path.stem)
        pages = data.get("pages", [])

        full_text = "\n".join(p.get("text", "") for p in pages)
        is_number, standard_id = detect_standard_info(source_document, full_text)

        chunks = build_chunks(pages, source_document, is_number, standard_id)
        all_chunks.extend(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("----- Chunking summary -----")
    print(f"JSON files processed: {len(json_files)}")
    print(f"Chunks created:       {len(all_chunks)}")
    print(f"Output written to:    {OUTPUT_FILE}")


if __name__ == "__main__":
    main()