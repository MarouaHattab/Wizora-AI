import codecs
from pathlib import Path

from charset_normalizer import from_bytes
from langchain_core.documents import Document

from helper.ingestion_errors import AssetNotFoundError, EmptyContentError, InvalidDocumentError


def _decode(payload: bytes) -> tuple[str, str]:
    if not payload:
        raise EmptyContentError("The text file is empty.")
    for prefix, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if payload.startswith(prefix):
            try:
                return payload.decode(encoding), encoding
            except UnicodeError as exc:
                raise InvalidDocumentError("The text file has invalid Unicode encoding.") from exc
    if b"\x00" in payload:
        raise InvalidDocumentError("The upload contains binary data instead of plain text.")
    try:
        return payload.decode("utf-8"), "utf-8"
    except UnicodeError:
        match = from_bytes(payload).best()
        if match is None or match.chaos > 0.2:
            raise InvalidDocumentError(
                "The text encoding could not be determined; save the file as UTF-8."
            )
        # Legacy single-byte encodings are ambiguous. Prefer the common Windows
        # Western encoding only when its language plausibility is comparable;
        # a detector can otherwise rank DOS encodings above ordinary French.
        western = from_bytes(payload, cp_isolation=["cp1252"]).best()
        if (
            western is not None
            and western.chaos <= match.chaos
            and western.coherence >= match.coherence - 0.05
            and (western.coherence > 0.1 or len(payload) < 64)
        ):
            match = western
        return str(match), match.encoding


def extract_text_documents(
    text_path: str, asset_id: str | None = None, source_name: str | None = None
) -> list[Document]:
    path = Path(text_path)
    try:
        text, encoding = _decode(path.read_bytes())
    except FileNotFoundError as exc:
        raise AssetNotFoundError("The text file was not found.") from exc
    except OSError as exc:
        raise InvalidDocumentError("The text file could not be read.") from exc
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        raise InvalidDocumentError(
            "The upload contains binary control characters instead of plain text."
        )
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    common = {
        "source": Path(source_name or path.name).name,
        "source_type": "text",
        "modality": "text",
        "content_type": "text",
        "encoding": encoding,
    }
    if asset_id is not None:
        common["asset_id"] = asset_id
    documents, paragraph, first_line = [], [], 1

    def append(last_line):
        if paragraph:
            documents.append(
                Document(
                    page_content="\n".join(paragraph).strip(),
                    metadata={
                        **common,
                        "record_index": len(documents),
                        "line_start": first_line,
                        "line_end": last_line,
                    },
                )
            )
            paragraph.clear()

    lines = text.split("\n")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            append(number - 1)
        else:
            if not paragraph:
                first_line = number
            paragraph.append(line)
    append(len(lines))
    if not documents:
        raise EmptyContentError("The text file contains no readable text.")
    return documents
