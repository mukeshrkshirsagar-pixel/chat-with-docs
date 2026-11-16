"""Turn uploaded bytes into text *sections* (text + page + heading), nothing smarter.

Keeping page/heading metadata is what makes citations useful later.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".docx"}


class IngestError(Exception):
    """Raised for user-fixable problems (unsupported type, empty file, ...)."""


@dataclass
class Section:
    text: str
    page: int | None = None
    heading: str | None = None


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _load_markdown_or_text(data: bytes) -> list[Section]:
    """Split on markdown headings so each section knows which heading it lives under."""
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text=text, heading=heading))
        buf.clear()

    for line in _decode(data).splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            heading = m.group(2)
        else:
            buf.append(line)
    flush()
    return sections


def _load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestError("PDF is password-protected.")
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    except PyPdfError as e:
        raise IngestError(f"Could not parse PDF: {e}") from e
    return [Section(text=t, page=n) for n, t in pages if t]


def _load_docx(data: bytes) -> list[Section]:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:  # python-docx raises assorted zip/xml errors
        raise IngestError(f"Could not parse DOCX: {e}") from e

    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        text = "\n\n".join(buf).strip()
        if text:
            sections.append(Section(text=text, heading=heading))
        buf.clear()

    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        if p.style is not None and p.style.name.lower().startswith("heading"):
            flush()
            heading = p.text.strip()
        else:
            buf.append(p.text.strip())
    flush()
    return sections


def load_document(filename: str, data: bytes) -> list[Section]:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise IngestError(
            f"Unsupported file type '{ext or filename}'. Supported: {', '.join(sorted(SUPPORTED))}"
        )
    if ext == ".pdf":
        sections = _load_pdf(data)
    elif ext == ".docx":
        sections = _load_docx(data)
    else:
        sections = _load_markdown_or_text(data)
    if not sections:
        raise IngestError(
            "No extractable text found (scanned/image-only PDFs need OCR, which is not supported yet)."
        )
    return sections
