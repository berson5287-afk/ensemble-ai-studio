"""Turn attached files into plain text.

Every extractor returns readable text or raises.  Binary files that we have no
extractor for are rejected loudly rather than decoded into garbage — an earlier
version of this app happily fed 200k characters of raw PDF bytes to a model,
which is exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Callable

MAX_BYTES = 2_000_000     # cap on raw bytes read from a plain-text file
MAX_CHARS = 200_000       # cap on extracted characters


class BinaryFileError(Exception):
    """The file has no text we can read."""


class ExtractionError(Exception):
    """We recognised the format but could not get text out of it."""


def _inner_text(xml_text: str, tag: str) -> list[str]:
    """Inner text of every <tag>...</tag>, with nested markup stripped."""
    chunks = re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", xml_text, re.DOTALL)
    return [re.sub(r"<[^>]+>", "", chunk) for chunk in chunks]


def extract_pdf(path: Path) -> str:
    reader = None
    last_error = None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        try:
            reader = module.PdfReader(str(path))
            break
        except Exception as exc:  # encrypted, malformed, ...
            last_error = exc
    if reader is None:
        if last_error is not None:
            raise ExtractionError(f"Could not open the PDF: {last_error}")
        raise ExtractionError(
            "Reading PDFs needs the 'pypdf' package.\n\n"
            "Install it once with:\n    py -m pip install pypdf")

    pages = []
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"--- Page {number} ---\n{text.strip()}")
    if not pages:
        raise ExtractionError(
            "No text found in this PDF — it is probably a scanned image, "
            "which needs OCR to read.")
    return "\n\n".join(pages)


def extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_text = archive.read("word/document.xml").decode("utf-8", "ignore")
    paragraphs = []
    for paragraph in re.findall(r"<w:p[ >].*?</w:p>", xml_text, re.DOTALL):
        line = "".join(_inner_text(paragraph, "w:t")).strip()
        if line:
            paragraphs.append(line)
    if not paragraphs:
        raise ExtractionError("No text found in this Word document.")
    return "\n".join(paragraphs)


def extract_pptx(path: Path) -> str:
    slides = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            (n for n in archive.namelist()
             if re.match(r"ppt/slides/slide\d+\.xml$", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)))
        for number, name in enumerate(names, 1):
            xml_text = archive.read(name).decode("utf-8", "ignore")
            lines = [t.strip() for t in _inner_text(xml_text, "a:t") if t.strip()]
            if lines:
                slides.append(f"--- Slide {number} ---\n" + "\n".join(lines))
    if not slides:
        raise ExtractionError("No text found in this presentation.")
    return "\n\n".join(slides)


def extract_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise ExtractionError(
            "Reading Excel files needs the 'openpyxl' package.\n\n"
            "Install it once with:\n    py -m pip install openpyxl") from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if value is None else str(value) for value in row]
                if any(cell.strip() for cell in cells):
                    rows.append("\t".join(cells).rstrip())
            if rows:
                sheets.append(f"--- Sheet: {sheet.title} ---\n" + "\n".join(rows))
    finally:
        workbook.close()
    if not sheets:
        raise ExtractionError("No data found in this spreadsheet.")
    return "\n\n".join(sheets)


def read_text(path: Path) -> str:
    """Decode a plain-text file, refusing anything that is really binary."""
    raw = path.read_bytes()[:MAX_BYTES]

    # UTF-16 text is full of null bytes, so check for its BOM before the
    # binary heuristic rejects a perfectly good text file.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except (UnicodeDecodeError, UnicodeError):
            pass

    if b"\x00" in raw[:8192]:
        raise BinaryFileError(path.name)
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        sample = text[:4000]
        if not sample:
            return text
        readable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
        if readable / len(sample) >= 0.90:
            return text
        raise BinaryFileError(path.name)
    raise BinaryFileError(path.name)


EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".pptx": extract_pptx,
    ".xlsx": extract_xlsx,
    ".xlsm": extract_xlsx,
}

SUPPORTED_HINT = ("text files, PDFs, Word documents, PowerPoint decks "
                  "and Excel workbooks")


def extract(path) -> str:
    """Extract text from any supported file, truncating very long results."""
    path = Path(path)
    extractor = EXTRACTORS.get(path.suffix.lower())
    text = extractor(path) if extractor else read_text(path)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n…[truncated — file was longer]…"
    return text
