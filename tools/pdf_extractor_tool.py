from pathlib import Path
from threading import Lock
from typing import Sequence

import pymupdf
from langchain_core.documents import Document

from helper.ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    InvalidDocumentError,
    ProcessingError,
)


_PDF_LOCK = Lock()


def _document_metadata(pdf) -> dict:
    result = {}
    for key in ("title", "author", "subject", "keywords", "creator", "producer", "creationDate", "modDate"):
        value = (pdf.metadata or {}).get(key)
        if isinstance(value, str) and value.strip():
            result[f"document_{key}"] = value.strip()[:512]
    return result


def _page_images(page) -> tuple[int, float]:
    images = page.get_image_info()
    area = sum(pymupdf.Rect(image["bbox"]).get_area() for image in images)
    return len(images), min(1.0, area / max(1.0, page.rect.get_area()))


def _page_text(page) -> tuple[str, int]:
    text = page.get_text("text", sort=True).strip()
    # Only ruled tables are detected; unstructured layouts retain the existing
    # text extraction. Table cell text replaces its page lines exactly once.
    finder = page.find_tables() if text and hasattr(page, "find_tables") else None
    tables = list(finder.tables) if finder is not None else []
    if not tables:
        return text, 0
    areas = [pymupdf.Rect(table.bbox) for table in tables]
    lines = []
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    for block in page.get_text("dict", sort=True, flags=flags)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            bounds = pymupdf.Rect(line["bbox"])
            center = (bounds.tl + bounds.br) / 2
            if any(center in area for area in areas):
                continue
            value = "".join(span["text"] for span in line.get("spans", [])).strip()
            if value:
                lines.append((bounds.y0, bounds.x0, value))
    for table in tables:
        rows = ["\t".join(str(cell or "").replace("\n", " / ") for cell in row) for row in table.extract()]
        value = "\n".join(row for row in rows if row.strip()).strip()
        if value:
            lines.append((table.bbox[1], table.bbox[0], value))
    lines.sort(key=lambda item: (round(item[0], 1), item[1]))
    return "\n".join(item[2] for item in lines).strip(), len(tables)


def inspect_pdf_pages(pdf_path: str) -> dict:
    path = Path(pdf_path)
    if not path.is_file():
        raise AssetNotFoundError("The PDF file was not found.")
    try:
        with _PDF_LOCK, pymupdf.open(path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise InvalidDocumentError("The PDF is invalid or password protected.")
            pages = []
            for page in pdf:
                text = page.get_text("text", sort=True).strip()
                image_count, coverage = _page_images(page)
                pages.append({
                    "page": page.number + 1, "has_text": bool(text),
                    "image_count": image_count,
                    "scanned": image_count > 0 and (not text or coverage >= 0.4 and len(text) < 40),
                })
            return {
                "page_count": pdf.page_count, "pages": pages,
                "scanned_pages": [page["page"] for page in pages if page["scanned"]],
                "empty_pages": [page["page"] for page in pages if not page["has_text"]],
                "metadata": _document_metadata(pdf),
            }
    except (AssetNotFoundError, InvalidDocumentError):
        raise
    except Exception as exc:
        raise InvalidDocumentError("The PDF could not be inspected.") from exc


def extract_text_from_pdf(
    pdf_path: str,
    asset_id: str | None = None,
    source_name: str | None = None,
    *,
    ocr: bool = False,
    ocr_language: str = "eng",
    max_ocr_pages: int = 20,
) -> list[Document]:
    """Extract readable pages with their original, one-based page numbers."""
    if not isinstance(ocr, bool) or type(max_ocr_pages) is not int or max_ocr_pages < 1:
        raise ValueError("OCR requires an explicit boolean and positive page limit.")
    if not isinstance(ocr_language, str) or not ocr_language or len(ocr_language) > 64:
        raise ValueError("A valid OCR language is required.")
    path = Path(pdf_path)
    source = Path(source_name or path.name).name
    documents = []
    try:
        if not path.is_file():
            raise AssetNotFoundError("The PDF file was not found.")
        with _PDF_LOCK, pymupdf.open(path) as pdf:
            if not pdf.is_pdf:
                raise InvalidDocumentError("The file is not a valid PDF.")
            if pdf.needs_pass:
                raise InvalidDocumentError("Password-protected PDFs are not supported.")
            page_count = pdf.page_count
            common = _document_metadata(pdf)
            scanned_count = empty_count = ocr_count = 0
            for page in pdf:
                text, table_count = _page_text(page)
                image_count, coverage = _page_images(page)
                scanned = image_count > 0 and (not text or coverage >= 0.4 and len(text) < 40)
                scanned_count += int(scanned)
                used_ocr = False
                if ocr and scanned:
                    if ocr_count >= max_ocr_pages:
                        raise ProcessingError("The PDF exceeds the configured OCR page limit.")
                    try:
                        text_page = page.get_textpage_ocr(language=ocr_language, dpi=150, full=True)
                        text = page.get_text("text", textpage=text_page, sort=True).strip()
                    except Exception as exc:
                        raise ProcessingError("PDF OCR is unavailable or failed; check the configured Tesseract installation.") from exc
                    ocr_count += 1
                    used_ocr = True
                if not text:
                    empty_count += 1
                    continue
                metadata = {
                    **common,
                    "source": source,
                    "source_type": "pdf",
                    "modality": "pdf",
                    "content_type": "page_text",
                    "page": page.number + 1,
                    "page_count": page_count,
                    "image_count": image_count,
                    "table_count": table_count,
                    "scanned": scanned,
                    "ocr": used_ocr,
                    "record_index": len(documents),
                }
                if asset_id is not None:
                    metadata["asset_id"] = asset_id
                documents.append(Document(page_content=text, metadata=metadata))
            for document in documents:
                document.metadata.update(scanned_page_count=scanned_count, empty_page_count=empty_count)
    except (AssetNotFoundError, InvalidDocumentError, ProcessingError):
        raise
    except Exception as exc:
        raise InvalidDocumentError("The PDF could not be read.") from exc

    if not documents:
        raise EmptyContentError("The PDF contains no readable text; scanned pages require explicit OCR.")
    return documents


def save_extracted_text(documents: Sequence[Document], output_path: str) -> None:
    """Save the compatibility text without joining unrelated source records."""
    text = "\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )
    if not text:
        raise EmptyContentError("There is no extracted text to save.")
    try:
        Path(output_path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ProcessingError("The extracted text could not be saved.") from exc
