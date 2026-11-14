"""Read and normalize source data. TODO: implement functions."""
from pathlib import Path
from .models import DocumentPage
import logging
import re
import unicodedata
from typing import Dict, Optional, List, Tuple
from pypdf import PdfReader

logger = logging.getLogger(__name__)

def load_pdf(
    path: str | Path,
    document_id: str,
    source: str,
    allowed_pages: Optional[set[int]] = None,
) -> Tuple[List[DocumentPage], List[Dict]]:
    """
    Extract each 1-indexed physical page with pypdf.
    Returns (pages, issue_log) where issue_log records empty or failed pages for OCR review.
    """
    reader = PdfReader(str(path))
    pages: List[DocumentPage] = []
    issue_log: List[Dict] = []
    current_chapter: Optional[str] = None

    for page_num, pdf_page in enumerate(reader.pages, start=1):
        if allowed_pages is not None and page_num not in allowed_pages:
            continue
        try:
            raw_text = pdf_page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", page_num, exc)
            issue_log.append({
                "page": page_num,
                "status": "error",
                "note": f"Extraction error: {exc}. Needs OCR or manual inspection.",
            })
            continue

        norm_text = normalize_text(raw_text)
        if not norm_text:
            logger.info("Empty text on page %d (candidate for OCR).", page_num)
            issue_log.append({
                "page": page_num,
                "status": "empty",
                "note": "No extractable text; may be scanned image/blank (OCR step).",
            })
            continue

        meta = extract_page_metadata(norm_text, current_chapter)
        current_chapter = meta.get("chapter", current_chapter)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=norm_text,
                metadata=meta,
            )
        )

    return pages, issue_log


def normalize_text(text: str) -> str:
    """TODO: normalize Unicode and whitespace without discarding useful structure."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printable characters except for newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse horizontal whitespace per line without destroying line breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Preserve paragraph breaks, collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_page_metadata(text: str, current_chapter: Optional[str] = None) -> Dict[str, str]:
    """
        Extract real filterable metadata from page text (e.g., Chapter / Section).
    """

    meta: Dict[str, str] = {}
    chap_match = re.search(r"(?im)^chapter\s+(\d+)\b", text)
    if chap_match:
        current_chapter = chap_match.group(1)
    if current_chapter:
        meta["chapter"] = current_chapter

    sec_match = re.search(r"(?m)^(\d+\.\d+)\s+[A-Z]", text)
    if sec_match:
        meta["section"] = sec_match.group(1)
    return meta