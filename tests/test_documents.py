"""Attachment extraction, including the binary-decode regression."""

from __future__ import annotations

import zipfile

import pytest

from aichatlab import documents


def test_reads_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("pipe spec: 6 inch\n", encoding="utf-8")
    assert "pipe spec" in documents.extract(path)


def test_reads_utf16_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes("café pressure".encode("utf-16"))
    assert "café pressure" in documents.extract(path)


def test_binary_file_is_refused_not_decoded(tmp_path):
    """The original bug: binary bytes silently became 200k chars of garbage."""
    path = tmp_path / "photo.bin"
    path.write_bytes(bytes(range(256)) * 40)

    with pytest.raises(documents.BinaryFileError):
        documents.extract(path)


def test_pdf_shaped_garbage_raises_instead_of_returning_junk(tmp_path):
    """A .pdf that pypdf cannot parse must fail loudly, never return bytes."""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 20)

    with pytest.raises((documents.ExtractionError, documents.BinaryFileError)):
        documents.extract(path)


def test_extracts_docx_paragraphs(tmp_path):
    path = tmp_path / "spec.docx"
    document_xml = (
        '<?xml version="1.0"?><w:document xmlns:w="x"><w:body>'
        '<w:p><w:r><w:t>Pipe specification</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Material: </w:t></w:r><w:r><w:t>steel</w:t></w:r></w:p>'
        '</w:body></w:document>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    text = documents.extract(path)
    assert "Pipe specification" in text
    assert "Material: steel" in text          # runs are joined, not split


def test_extracts_pptx_slides_in_order(tmp_path):
    path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide2.xml", '<p><a:t>second</a:t></p>')
        archive.writestr("ppt/slides/slide1.xml", '<p><a:t>first</a:t></p>')
        archive.writestr("ppt/slides/slide10.xml", '<p><a:t>tenth</a:t></p>')

    text = documents.extract(path)
    assert text.index("first") < text.index("second") < text.index("tenth")


def test_empty_docx_reports_no_text(tmp_path):
    path = tmp_path / "blank.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml",
                         '<w:document xmlns:w="x"><w:body></w:body></w:document>')

    with pytest.raises(documents.ExtractionError, match="No text"):
        documents.extract(path)


def test_long_files_are_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "MAX_CHARS", 100)
    path = tmp_path / "big.txt"
    path.write_text("x" * 5000, encoding="utf-8")

    text = documents.extract(path)
    assert len(text) < 200
    assert "truncated" in text
