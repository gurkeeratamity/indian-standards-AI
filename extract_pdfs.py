import sys
import json
from pathlib import Path
import pymupdf


INPUT_DIR = Path("data/standards")
OUTPUT_DIR = Path("data/processed")
LOW_TEXT_THRESHOLD = 20


def extract_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    low_text_pages = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text,
        })

        if len(text.strip()) < LOW_TEXT_THRESHOLD:
            low_text_pages.append(page_number + 1)

    doc.close()
    return pages, low_text_pages


def write_output(source_filename, pages, output_dir):
    output_data = {
        "source_document": source_filename,
        "pages": pages,
    }

    output_path = output_dir / (Path(source_filename).stem + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"ERROR: Input folder '{INPUT_DIR}' does not exist.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"ERROR: No PDF files found in '{INPUT_DIR}'.")

    total_pages = 0
    low_text_report = []

    for pdf_path in pdf_files:
        pages, low_text_pages = extract_pdf(pdf_path)
        write_output(pdf_path.name, pages, OUTPUT_DIR)

        total_pages += len(pages)

        if low_text_pages:
            low_text_report.append((pdf_path.name, low_text_pages))

    print("----- Extraction summary -----")
    print(f"PDFs processed:  {len(pdf_files)}")
    print(f"Pages processed: {total_pages}")

    if low_text_report:
        print("\nPages with little or no extracted text (possibly scanned images):")
        for filename, pages in low_text_report:
            print(f"  {filename}: pages {pages}")
    else:
        print("\nNo low-text pages detected.")


if __name__ == "__main__":
    main()