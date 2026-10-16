import hashlib
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document as WordDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document

from helper.ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    InvalidDocumentError,
    ProcessingError,
)


_MAX_MEMBERS = 4096
_MAX_EXPANDED_BYTES = 256 * 1024 * 1024
_IMAGE_EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff",
}


def _open_document(path: Path):
    if not path.is_file():
        raise AssetNotFoundError("The Word document was not found.")
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) > _MAX_MEMBERS
                or sum(member.file_size for member in members) > _MAX_EXPANDED_BYTES
                or any(member.flag_bits & 1 for member in members)
                or "word/document.xml" not in archive.namelist()
            ):
                raise InvalidDocumentError("The Word document exceeds safe package limits or is encrypted.")
        return WordDocument(path)
    except InvalidDocumentError:
        raise
    except (BadZipFile, OSError, ValueError, KeyError) as exc:
        raise InvalidDocumentError("The file is not a readable Word document.") from exc
    except Exception as exc:
        raise InvalidDocumentError("The Word document could not be read.") from exc


def _blocks(document):
    for position, element in enumerate(document.element.body.iterchildren(), 1):
        if element.tag == qn("w:p"):
            yield f"body-{position:06d}", Paragraph(element, document)
        elif element.tag == qn("w:tbl"):
            yield f"body-{position:06d}", Table(element, document)


def _heading_level(paragraph: Paragraph) -> int | None:
    style, seen = paragraph.style, set()
    properties = paragraph._p.pPr
    while True:
        level = properties.find(qn("w:outlineLvl")) if properties is not None else None
        if level is not None:
            value = level.get(qn("w:val"))
            if value is not None and value.isdecimal() and 0 <= int(value) <= 9:
                return int(value) + 1 if int(value) < 9 else None
        if style is None or style.style_id in seen:
            break
        seen.add(style.style_id)
        match = re.fullmatch(r"heading\s*([1-9])", style.name or "", re.IGNORECASE)
        if match:
            return int(match.group(1))
        properties = style.element.pPr
        style = style.base_style
    return None


def _table_text(table: Table) -> str:
    rows, seen = [], set()
    for row in table.rows:
        values = []
        for cell in row.cells:
            if cell._tc in seen:
                values.append("")
                continue
            seen.add(cell._tc)
            parts = []
            for element in cell._tc.iterchildren():
                if element.tag == qn("w:p"):
                    text = Paragraph(element, cell).text.strip()
                elif element.tag == qn("w:tbl"):
                    text = _table_text(Table(element, cell))
                else:
                    continue
                if text:
                    parts.append(text)
            values.append(" / ".join(parts).replace("\t", " "))
        if any(values):
            rows.append("\t".join(values))
    return "\n".join(rows).strip()


def _document_metadata(document) -> dict:
    properties = document.core_properties
    metadata = {}
    for name in ("title", "author", "subject", "keywords", "category", "language"):
        value = getattr(properties, name, None)
        if isinstance(value, str) and value.strip():
            metadata[f"document_{name}"] = value.strip()[:512]
    for name in ("created", "modified"):
        value = getattr(properties, name, None)
        if value is not None:
            metadata[f"document_{name}"] = value.isoformat()
    return metadata


def _image_relationships(element):
    identifiers = []
    for node in element.iter():
        if node.tag == qn("a:blip"):
            identifier = node.get(qn("r:embed"))
        elif node.tag.endswith("}imagedata"):
            identifier = node.get(qn("r:id"))
        else:
            continue
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)
    return identifiers


def _extract_images(document, path, image_dir, max_images, max_image_bytes):
    if type(max_images) is not int or not 0 <= max_images <= 128:
        raise ValueError("max_images must be between zero and 128.")
    if type(max_image_bytes) is not int or not 1 <= max_image_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_image_bytes must be between one byte and 64 MiB.")
    destination = Path(image_dir) if image_dir is not None else path.parent / "images"
    # References stay relative to the asset directory even when it is staged.
    try:
        relative_dir = destination.resolve().relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError("The image directory must be inside the source asset directory.") from exc
    records, written = [], set()
    for element_id, block in _blocks(document):
        for identifier in _image_relationships(block._element):
            relationship = document.part.rels.get(identifier)
            if relationship is None or relationship.is_external:
                continue
            part = relationship.target_part
            mime = part.content_type
            if not mime.startswith("image/"):
                continue
            blob = part.blob
            digest = hashlib.sha256(blob).hexdigest()
            record = {
                "element_id": element_id, "image_id": digest,
                "content_type": mime, "byte_count": len(blob),
            }
            extension = _IMAGE_EXTENSIONS.get(mime)
            if not extension:
                record["status"] = "unsupported_format"
            elif len(blob) > max_image_bytes:
                record["status"] = "too_large"
            elif digest not in written and len(written) >= max_images:
                record["status"] = "limit_reached"
            else:
                filename = digest + extension
                if digest not in written:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / filename).write_bytes(blob)
                written.add(digest)
                record.update(status="available", reference=(relative_dir / filename).as_posix())
            records.append(record)
    return records


def extract_docx_images(
    docx_path: str,
    *,
    image_dir: str | Path | None = None,
    max_images: int = 12,
    max_image_bytes: int = 10 * 1024 * 1024,
) -> list[dict]:
    path = Path(docx_path)
    document = _open_document(path)
    try:
        return _extract_images(document, path, image_dir, max_images, max_image_bytes)
    except OSError as exc:
        raise ProcessingError("The embedded Word images could not be stored.") from exc


def extract_docx_documents(
    docx_path: str,
    asset_id: str | None = None,
    source_name: str | None = None,
    *,
    image_dir: str | Path | None = None,
    max_images: int = 12,
    max_image_bytes: int = 10 * 1024 * 1024,
) -> list[Document]:
    path = Path(docx_path)
    document = _open_document(path)
    source = Path(source_name or path.name).name
    try:
        images = _extract_images(document, path, image_dir, max_images, max_image_bytes)
        common = {
            "source": source, "source_type": "docx", "modality": "docx",
            **_document_metadata(document),
        }
        if asset_id is not None:
            common["asset_id"] = asset_id
        documents, headings, images_by_element = [], {}, {}
        for image in images:
            images_by_element.setdefault(image["element_id"], []).append(image)
        paragraph_index = table_index = 0
        for element_id, block in _blocks(document):
            metadata = {**common, "element_id": element_id, "record_index": len(documents)}
            if isinstance(block, Paragraph):
                paragraph_index += 1
                text = block.text.strip()
                level = _heading_level(block)
                metadata.update(content_type="heading" if level else "paragraph", paragraph=paragraph_index)
                if level and text:
                    headings = {depth: title for depth, title in headings.items() if depth < level}
                    headings[level] = text
                    metadata["heading_level"] = level
            else:
                table_index += 1
                text = _table_text(block)
                metadata.update(content_type="table", table_index=table_index)
            path_names = [title for _, title in sorted(headings.items())]
            if path_names:
                metadata.update(section=path_names[-1], section_path=json.dumps(path_names, ensure_ascii=False))
            owned_images = images_by_element.get(element_id, [])
            if owned_images:
                metadata.update(
                    image_count=len(owned_images),
                    image_references=json.dumps(owned_images, ensure_ascii=False, sort_keys=True),
                )
            if text:
                documents.append(Document(page_content=text, metadata=metadata))
    except (OSError, ValueError, TypeError) as exc:
        raise InvalidDocumentError("The Word document content could not be extracted.") from exc
    if not documents:
        raise EmptyContentError("The Word document contains no readable paragraphs or tables.")
    return documents
