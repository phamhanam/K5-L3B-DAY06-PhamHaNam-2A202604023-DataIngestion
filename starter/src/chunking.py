"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage
import re
from typing import List


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """TODO: split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if not (0 <= overlap < size):
        raise ValueError("overlap must be >= 0 and < size")

    text = page.text
    chunks: List[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0

    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}_p{page.page}_fixed_{idx}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=piece,
                    metadata={**page.metadata, "strategy": "fixed_size", "chunk_index": idx},
                )
            )
            idx += 1
        if start + size >= len(text):
            break
        start += step

    return chunks

def _split_oversized(paragraph: str, max_size: int) -> List[str]:
    """Split a paragraph > max_size on sentence or word boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: List[str] = []
    buf = ""

    for sent in sentences:
        if len(sent) > max_size:
            # Hard wrap very long sentences on word boundaries
            if buf:
                parts.append(buf.strip())
                buf = ""
            words = sent.split()
            for w in words:
                if len(buf) + len(w) + 1 <= max_size:
                    buf = f"{buf} {w}".strip()
                else:
                    if buf:
                        parts.append(buf)
                    buf = w
        elif len(buf) + len(sent) + 1 <= max_size:
            buf = f"{buf} {sent}".strip()
        else:
            if buf:
                parts.append(buf.strip())
            buf = sent

    if buf:
        parts.append(buf.strip())
    return parts

def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """TODO: group paragraphs, split oversized paragraphs and retain metadata."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
    merged: List[str] = []
    buf = ""

    for para in paragraphs:
        if len(para) > max_size:
            if buf:
                merged.append(buf)
                buf = ""
            merged.extend(_split_oversized(para, max_size))
        elif not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= max_size:
            buf = f"{buf}\n\n{para}"
        else:
            merged.append(buf)
            buf = para

    if buf:
        merged.append(buf)

    return [
        Chunk(
            chunk_id=f"{page.document_id}_p{page.page}_para_{idx}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=piece,
            metadata={**page.metadata, "strategy": "by_paragraph", "chunk_index": idx},
        )
        for idx, piece in enumerate(merged)
    ]
