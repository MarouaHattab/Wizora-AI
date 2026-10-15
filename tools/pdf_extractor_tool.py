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


def extract_text_from_pdf(
    pdf_path: str,
    asset_id: str | None = None,
    source_name: str | None = None,
) -> list[Document]:
    """Extract readable pages with their original, one-based page numbers."""
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
            for page in pdf:
                text = page.get_text("text", sort=True).strip()
                if not text:
                    continue
                metadata = {
                    "source": source,
                    "source_type": "pdf",
                    "page": page.number + 1,
                    "page_count": page_count,
                    "record_index": len(documents),
                }
                if asset_id is not None:
                    metadata["asset_id"] = asset_id
                documents.append(Document(page_content=text, metadata=metadata))
    except (AssetNotFoundError, InvalidDocumentError):
        raise
    except Exception as exc:
        raise InvalidDocumentError("The PDF could not be read.") from exc

    if not documents:
        raise EmptyContentError("The PDF contains no readable text.")
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
