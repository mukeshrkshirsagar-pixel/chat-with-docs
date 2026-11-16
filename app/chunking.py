"""Sentence-aware chunking with overlap.

Strategy: split each section into sentence-ish units, greedily pack units up to
`size` characters, and start each new chunk with the last few units of the previous
one (<= `overlap` chars) so facts straddling a boundary are still retrievable.
Chunks never cross section boundaries, so page / heading metadata stays truthful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class ChunkDraft:
    text: str
    page: int | None
    heading: str | None


def _units(text: str, size: int) -> list[str]:
    """Paragraphs -> sentences -> hard splits, so no unit exceeds `size`."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())  # collapse whitespace / PDF line wraps
        if not para:
            continue
        pieces = [para] if len(para) <= size else _SENTENCE_END.split(para)
        for piece in pieces:
            while len(piece) > size:  # pathological: no sentence boundaries
                cut = piece.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                out.append(piece[:cut])
                piece = piece[cut:].lstrip()
            if piece:
                out.append(piece)
    return out


def chunk_sections(sections: list[Section], size: int = 900, overlap: int = 150) -> list[ChunkDraft]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[ChunkDraft] = []
    for sec in sections:
        current: list[str] = []
        length = 0
        for unit in _units(sec.text, size):
            if current and length + len(unit) + 1 > size:
                chunks.append(ChunkDraft(" ".join(current), sec.page, sec.heading))
                # seed the next chunk with trailing units up to `overlap` chars
                carry: list[str] = []
                carried = 0
                for u in reversed(current):
                    if carried + len(u) > overlap:
                        break
                    carry.insert(0, u)
                    carried += len(u) + 1
                current, length = carry, carried
            current.append(unit)
            length += len(unit) + 1
        if current:  # always holds at least one unit not yet emitted (the carry is seeded before append)
            chunks.append(ChunkDraft(" ".join(current), sec.page, sec.heading))
    return chunks
