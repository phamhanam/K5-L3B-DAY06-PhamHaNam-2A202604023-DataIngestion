import json
import re
from pathlib import Path
import fitz  # PyMuPDF

MANIFEST_PATH = Path("data/manifest.json")
BENCHMARK_PATH = Path("data/benchmark.json")
PDF_PATH = Path("data/corpus.pdf")

REQUIRED_FIELDS = {
    "id", "split", "question", "expected_answer",
    "gold_source_id", "gold_page", "evidence_quote"
}

def normalize(text: str) -> str:
    # Fix PDF line-break hyphenation, collapse whitespace, lowercase
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    doc = fitz.open(PDF_PATH)

    valid_pages = set(manifest["page_ids"])
    dev_items = [x for x in items if x.get("split") == "dev"]
    test_items = [x for x in items if x.get("split") == "test"]
    filter_items = [x for x in items if x.get("filter")]
    test_filter_items = [x for x in test_items if x.get("filter")]

    errors = []

    if len(items) < 20:
        errors.append(f"Need >= 20 questions, found {len(items)}.")
    if len(dev_items) < 15:
        errors.append(f"Need >= 15 dev questions, found {len(dev_items)}.")
    if len(test_items) < 5:
        errors.append(f"Need >= 5 test questions, found {len(test_items)}.")
    if len(filter_items) < 3:
        errors.append(f"Need >= 3 metadata-filter questions, found {len(filter_items)}.")
    if len(test_filter_items) < 1:
        errors.append("Need >= 1 metadata-filter question in the test split.")

    seen_ids = set()
    for item in items:
        qid = item.get("id", "<missing>")
        if qid in seen_ids:
            errors.append(f"[{qid}] Duplicate ID.")
        seen_ids.add(qid)

        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            errors.append(f"[{qid}] Missing fields: {missing}")
            continue

        pages = item["gold_page"] if isinstance(item["gold_page"], list) else [item["gold_page"]]
        quote_norm = normalize(item["evidence_quote"])

        matched_any_page = False
        for p in pages:
            if p not in valid_pages:
                errors.append(f"[{qid}] gold_page {p} is not in manifest page_ids.")
                continue
            page_text_norm = normalize(doc[p - 1].get_text("text"))
            if quote_norm in page_text_norm:
                matched_any_page = True

        if not matched_any_page:
            errors.append(f"[{qid}] evidence_quote NOT found on gold_page(s) {pages}.")

        if item.get("filter"):
            f_str = json.dumps(item["filter"]).lower()
            if "gold_page" in f_str or str(item["expected_answer"]).lower() in f_str:
                errors.append(f"[{qid}] Filter appears to leak gold_page or expected_answer.")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    print(f"VALIDATION PASSED: {len(items)} questions ({len(dev_items)} dev / {len(test_items)} test), "
          f"{len(filter_items)} with filters ({len(test_filter_items)} in test).")

if __name__ == "__main__":
    main()
