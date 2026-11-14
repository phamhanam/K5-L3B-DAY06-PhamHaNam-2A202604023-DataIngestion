import re
import unicodedata
from typing import Any


def normalize_for_match(text: str) -> str:
    """Normalize text for reliable substring matching of evidence_quote."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _as_list(val: Any) -> list[Any]:
    return val if isinstance(val, list) else [val]


def is_evidence_hit(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
    """True iff chunk matches gold_source_id + gold_page and contains full evidence_quote."""
    gold_sources = set(str(s) for s in _as_list(item["gold_source_id"]))
    gold_pages = set(int(p) for p in _as_list(item["gold_page"]))

    chunk_source = str(chunk.get("document_id") or chunk.get("source") or "")
    chunk_page = chunk.get("page")

    if chunk_source not in gold_sources or chunk_page not in gold_pages:
        return False

    quote_norm = normalize_for_match(item["evidence_quote"])
    chunk_norm = normalize_for_match(chunk.get("text", ""))
    return quote_norm in chunk_norm


def is_split_evidence(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
    """Detect partial quote overlap on the gold page (evidence split across chunks)."""
    gold_sources = set(str(s) for s in _as_list(item["gold_source_id"]))
    gold_pages = set(int(p) for p in _as_list(item["gold_page"]))

    chunk_source = str(chunk.get("document_id") or chunk.get("source") or "")
    if chunk_source not in gold_sources or chunk.get("page") not in gold_pages:
        return False

    quote_norm = normalize_for_match(item["evidence_quote"])
    chunk_norm = normalize_for_match(chunk.get("text", ""))
    if quote_norm in chunk_norm:
        return False

    # Check if at least 40% of the quote (or a 6-word span) appears in this chunk
    words = quote_norm.split()
    if len(words) < 4:
        return False
    half = max(3, len(words) // 2)
    return (" ".join(words[:half]) in chunk_norm) or (" ".join(words[-half:]) in chunk_norm)


def score_question(item: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Hit@1, Hit@3, and split-evidence error flag for one question."""
    hits = [is_evidence_hit(c, item) for c in retrieved]
    hit_at_1 = bool(hits[0]) if hits else False
    hit_at_3 = any(hits[:3])
    split_error = (not hit_at_3) and any(is_split_evidence(c, item) for c in retrieved[:3])

    return {
        "id": item["id"],
        "split": item["split"],
        "question": item["question"],
        "filter": item.get("filter"),
        "gold_source_id": item["gold_source_id"],
        "gold_page": item["gold_page"],
        "evidence_quote": item["evidence_quote"],
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "split_evidence_error": split_error,
        "retrieved": retrieved[:3],
    }


def summarize_split(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Report n/N counts and percentages for Hit@1, Hit@3, and split-evidence errors."""
    n = len(traces)
    h1 = sum(1 for t in traces if t["hit_at_1"])
    h3 = sum(1 for t in traces if t["hit_at_3"])
    splits = [t["id"] for t in traces if t["split_evidence_error"]]
    return {
        "total": n,
        "hit_at_1": f"{h1}/{n} ({(100 * h1 / n):.1f}%)" if n else "0/0",
        "hit_at_3": f"{h3}/{n} ({(100 * h3 / n):.1f}%)" if n else "0/0",
        "hit_at_1_count": h1,
        "hit_at_3_count": h3,
        "split_evidence_errors": splits,
    }


def evaluate_strategy(traces: list[dict[str, Any]]) -> dict[str, Any]:
    dev_traces = [t for t in traces if t["split"] == "dev"]
    test_traces = [t for t in traces if t["split"] == "test"]
    return {
        "dev": summarize_split(dev_traces),
        "test": summarize_split(test_traces),
    }