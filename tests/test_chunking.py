import pytest

from app.chunking import chunk_sections
from app.loaders import IngestError, Section, load_document


def test_chunks_respect_size_and_keep_metadata():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_sections([Section(text, page=3, heading="Intro")], size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.page == 3 and c.heading == "Intro" for c in chunks)


def test_overlap_repeats_content_across_boundary():
    text = " ".join(f"Fact {i} stands alone." for i in range(60))
    chunks = chunk_sections([Section(text)], size=200, overlap=60)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text  # last unit of chunk 0 reappears in chunk 1


def test_chunks_never_cross_sections():
    secs = [Section("Alpha content here.", heading="A"), Section("Beta content here.", heading="B")]
    chunks = chunk_sections(secs, size=900, overlap=100)
    assert [c.heading for c in chunks] == ["A", "B"]
    assert "Beta" not in chunks[0].text


def test_unbroken_text_is_hard_split():
    chunks = chunk_sections([Section("x" * 2500)], size=500, overlap=50)
    assert all(len(c.text) <= 500 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 2500


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_sections([Section("hi")], size=100, overlap=100)


def test_markdown_headings_become_section_metadata():
    secs = load_document("a.md", b"# Title\n\nintro\n\n## Pay\n\nSalary is paid monthly.\n")
    assert [(s.heading, s.text) for s in secs] == [("Title", "intro"), ("Pay", "Salary is paid monthly.")]


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(IngestError):
        load_document("evil.exe", b"MZ")
    with pytest.raises(IngestError):
        load_document("empty.txt", b"   \n  ")
