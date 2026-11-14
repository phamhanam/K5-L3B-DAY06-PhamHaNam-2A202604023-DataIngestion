import json
from datetime import date
from pathlib import Path
import fitz  # PyMuPDF

PDF_PATH = Path("data/corpus.pdf")
MANIFEST_PATH = Path("data/manifest.json")

# Update these to match your chosen corpus
METADATA = {
    "source_id": "openstax_econ_3e",
    "title": "Principles of Economics 3e",
    "url_or_path": "https://openstax.org/details/books/principles-economics-3e",
    "license": "CC BY-NC-SA 4.0",
    "access_date": str(date.today()),
}

MIN_CHARS_PER_PAGE = 250  # Filters out blank pages, dividers, and sparse covers
MAX_PAGES_TO_KEEP = 50    # Keep 30-50 content pages for fast iteration

def is_toc_or_frontmatter(text: str, page_num: int) -> bool:
    lower = text.lower()
    if page_num <= 5 and ("table of contents" in lower or "contents" in lower[:200]):
        return True
    if "access for free at openstax.org" in lower and len(text.strip()) < 400:
        return True
    return False

def main():
    doc = fitz.open(PDF_PATH)
    page_ids = []
    char_counts = {}

    for idx in range(len(doc)):
        page_num = idx + 1  # 1-indexed page ID
        text = doc[idx].get_text("text").strip()
        char_len = len(text)

        if char_len < MIN_CHARS_PER_PAGE:
            continue
        if is_toc_or_frontmatter(text, page_num):
            continue

        page_ids.append(page_num)
        char_counts[str(page_num)] = char_len

        if len(page_ids) >= MAX_PAGES_TO_KEEP:
            break

    assert len(page_ids) >= 30, f"Only found {len(page_ids)} content pages; need >= 30."

    manifest = {
        **METADATA,
        "local_path": str(PDF_PATH),
        "total_selected_pages": len(page_ids),
        "total_chars": sum(char_counts.values()),
        "page_ids": page_ids,
        "char_counts": char_counts,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest with {len(page_ids)} pages ({manifest['total_chars']:,} chars) -> {MANIFEST_PATH}")

if __name__ == "__main__":
    main()