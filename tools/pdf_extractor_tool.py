import hashlib
import json
import re
import unicodedata
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
_EXTRACTION_CONTRACT = "document-text-v3-raw-cell-fusion"
_OCR_UNCERTAINTY = "OCR text is unverified and may omit or misrecognize content; recognition confidence is unavailable. Full-page OCR has no line-level positions or verified fusion order."


def _scanned_page(text, image_count, coverage):
    # Spatial text formatting can contain many padding spaces between a few
    # native labels; those spaces do not demonstrate usable page coverage.
    return image_count > 0 and (not text or coverage >= 0.4 and len("".join(text.split())) < 40)


def _ocr_identity(enabled, language, backend, binary, deadline):
    if not enabled:
        return None
    from tools.ocr_support import ocr_cache_identity

    return ocr_cache_identity(language=language, backend=backend, binary=binary, deadline=deadline)


def _cached_page(checkpoint, page, *, scanned=None, ocr, language, backend, binary, deadline):
    cached = checkpoint.load(page.number)
    # The content/configuration/contract key already binds these records to the
    # source page. Their metadata permits OCR preflight without decoding again.
    eligible = bool(scanned) or any(
        record.metadata.get("scanned") or record.metadata.get("ocr") for record in cached or ()
    )
    if cached is not None and eligible and ocr:
        from tools.ocr_support import validate_ocr_configuration

        try:
            validate_ocr_configuration(
                language=language, backend=backend, binary=binary, deadline=deadline
            )
        except ProcessingError:
            return None
    return cached


def _table_cell_lines(tables):
    # Preserve actual cell boundaries before compatibility text replaces native
    # newlines with " / ". Literal division symbols are never split.
    return [
        line
        for table in tables
        for row in table.extract()
        for cell in row
        for line in str(cell or "").splitlines()
        if line.strip()
    ]


def _complementary_ocr(native, recognized, *, native_fragments=()):
    from tools.ocr_support import normalize_ocr_text

    def identity(line):
        return " ".join(normalize_ocr_text(line).split())

    seen = {identity(line) for line in native.splitlines() if line.strip()}
    seen.update(identity(line) for line in native_fragments if line.strip())
    for line in native.splitlines():
        if "\t" in line:
            # OCR commonly wraps table cells onto separate lines. Compare exact
            # cell text as well as rows without normalizing away math identity.
            seen.update(identity(cell) for cell in line.split("\t") if cell.strip())
    lines = []
    for line in normalize_ocr_text(recognized).splitlines():
        key = identity(line)
        if key and key not in seen:
            lines.append(line)
            seen.add(key)
    return "\n".join(lines)


def _ocr_record(text, metadata, rect):
    return Document(
        page_content=text,
        metadata={
            **metadata,
            "ocr": True,
            "content_type": "page_text",
            "bbox": json.dumps(list(rect)),
            "extraction_method": "ocr",
            "uncertainty": _OCR_UNCERTAINTY,
        },
    )


def _document_metadata(pdf) -> dict:
    result = {}
    for key in (
        "title",
        "author",
        "subject",
        "keywords",
        "creator",
        "producer",
        "creationDate",
        "modDate",
    ):
        value = (pdf.metadata or {}).get(key)
        if isinstance(value, str) and value.strip():
            result[f"document_{key}"] = value.strip()[:512]
    return result


def _page_images(page) -> tuple[int, float]:
    images = page.get_image_info()
    area = sum(pymupdf.Rect(image["bbox"]).get_area() for image in images)
    return len(images), min(1.0, area / max(1.0, page.rect.get_area()))


def _native_text(page):
    from tools.ocr_support import normalize_ocr_text

    text = page.get_text("text", sort=True)
    if any(unicodedata.bidirectional(char) in {"R", "AL"} for char in text):
        text = page.get_text("text", sort=False)
    return normalize_ocr_text(text)


def _page_text(page) -> tuple[str, int]:
    text = _native_text(page)
    # Only ruled tables are detected; unstructured layouts retain the existing
    # text extraction. Table cell text replaces its page lines exactly once.
    finder = page.find_tables() if text and hasattr(page, "find_tables") else None
    tables = list(finder.tables) if finder is not None else []
    if not tables:
        return text, 0
    records = _hybrid_text_records(page, tables, {})
    return "\n".join(record.page_content for record in records).strip(), len(tables)


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
                text = _native_text(page)
                image_count, coverage = _page_images(page)
                pages.append(
                    {
                        "page": page.number + 1,
                        "has_text": bool(text),
                        "image_count": image_count,
                        "scanned": _scanned_page(text, image_count, coverage),
                    }
                )
            return {
                "page_count": pdf.page_count,
                "pages": pages,
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
    structured: bool = False,
    progress=None,
    ocr_backend="auto",
    ocr_binary=None,
    deadline=None,
) -> list[Document]:
    if not isinstance(ocr, bool) or type(max_ocr_pages) is not int or max_ocr_pages < 1:
        raise ValueError("OCR requires an explicit boolean and positive page limit.")
    if not isinstance(ocr_language, str) or not ocr_language or len(ocr_language) > 64:
        raise ValueError("A valid OCR language is required.")
    path = Path(pdf_path)
    source = Path(source_name or path.name).name
    documents = []
    ocr_warnings = []
    from helper.extraction_checkpoint import ChunkCheckpoint

    checkpoint = (
        ChunkCheckpoint(
            path,
            "pdf-text",
            {
                "extraction_contract": _EXTRACTION_CONTRACT,
                "language_data": _ocr_identity(
                    ocr, ocr_language, ocr_backend, ocr_binary, deadline
                ),
                "ocr": ocr,
                "language": ocr_language,
                "backend": ocr_backend,
                "binary": ocr_binary,
                "max_pages": max_ocr_pages,
                "structured": structured,
                "asset_id": asset_id,
                "source_name": source,
            },
        )
        if path.is_file()
        else None
    )
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
                if deadline is not None:
                    from tools.document_visuals import check_deadline

                    check_deadline(deadline)
                cached = _cached_page(
                    checkpoint,
                    page,
                    ocr=ocr,
                    language=ocr_language,
                    backend=ocr_backend,
                    binary=ocr_binary,
                    deadline=deadline,
                )
                if cached is not None:
                    documents.extend(cached)
                    scanned_count += int(bool(cached and cached[0].metadata.get("scanned")))
                    ocr_count += int(any(record.metadata.get("ocr") for record in cached))
                    empty_count += int(not cached)
                    if callable(progress):
                        progress("reusing_extraction_checkpoint", {"page": page.number + 1})
                    continue
                first_record = len(documents)
                text, table_count = _page_text(page)
                image_count, coverage = _page_images(page)
                scanned = _scanned_page(text, image_count, coverage)
                scanned_count += int(scanned)
                used_ocr = False
                ocr_extra = ""
                ocr_page_failed = False
                if ocr and scanned:
                    if ocr_count >= max_ocr_pages:
                        ocr_warnings.append(
                            "The PDF OCR page limit was reached. Remaining readable text was retained."
                        )
                        ocr_page_failed = True
                    else:
                        try:
                            if callable(progress):
                                progress("extracting_document_ocr", {"page": page.number + 1})
                            from tools.ocr_support import recognize_page

                            recognized = recognize_page(
                                page,
                                language=ocr_language,
                                backend=ocr_backend,
                                binary=ocr_binary,
                                deadline=deadline,
                            )
                            if recognized.strip():
                                native_fragments = (
                                    _table_cell_lines(page.find_tables().tables)
                                    if table_count
                                    else ()
                                )
                                ocr_extra = _complementary_ocr(
                                    text, recognized, native_fragments=native_fragments
                                )
                                used_ocr = bool(ocr_extra)
                            else:
                                ocr_warnings.append(
                                    f"OCR returned empty text on page {page.number + 1}; native text was retained and scanned content may be missing."
                                )
                                ocr_page_failed = True
                        except ProcessingError as exc:
                            ocr_warnings.append(f"Page {page.number + 1}: {exc}")
                            ocr_page_failed = True
                        ocr_count += 1
                if not text and not ocr_extra:
                    empty_count += 1
                    if not ocr_page_failed:
                        checkpoint.save(page.number, [])
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
                    "extraction_method": "pymupdf+ocr"
                    if text and used_ocr
                    else "ocr"
                    if used_ocr
                    else "pymupdf",
                    "record_index": len(documents),
                }
                if asset_id is not None:
                    metadata["asset_id"] = asset_id
                if used_ocr:
                    metadata["uncertainty"] = _OCR_UNCERTAINTY
                if structured:
                    tables = _hybrid_tables(page)
                    metadata["table_count"] = len(tables)
                    documents.extend(_hybrid_text_records(page, tables, {**metadata, "ocr": False}))
                    if ocr_extra:
                        documents.append(_ocr_record(ocr_extra, metadata, page.rect))
                else:
                    documents.append(
                        Document(
                            page_content="\n".join(value for value in (text, ocr_extra) if value),
                            metadata=metadata,
                        )
                    )
                if not ocr_page_failed:
                    checkpoint.save(page.number, documents[first_record:])
            for document in documents:
                document.metadata.update(
                    scanned_page_count=scanned_count, empty_page_count=empty_count
                )
                if ocr_warnings:
                    document.metadata["ocr_warning"] = " ".join(dict.fromkeys(ocr_warnings))[:1500]
    except (AssetNotFoundError, InvalidDocumentError, ProcessingError):
        raise
    except Exception as exc:
        raise InvalidDocumentError("The PDF could not be read.") from exc

    if not documents:
        raise EmptyContentError(
            " ".join(dict.fromkeys(ocr_warnings))
            or "The PDF contains no readable text; scanned pages require explicit OCR."
        )
    return documents


def save_extracted_text(documents: Sequence[Document], output_path: str) -> None:
    text = "\n\n".join(
        document.page_content.strip() for document in documents if document.page_content.strip()
    )
    if not text:
        raise EmptyContentError("There is no extracted text to save.")
    try:
        Path(output_path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ProcessingError("The extracted text could not be saved.") from exc


def _hybrid_tables(page):
    finder = page.find_tables() if hasattr(page, "find_tables") else None
    tables = list(finder.tables) if finder is not None else []
    # Flowchart boxes and connectors can resemble a sparse table to PyMuPDF.
    # A purported multi-row table with only one populated row is not reliable
    # structured evidence; keep its labels as text and inspect its vector region.
    return [
        table
        for table in tables
        if sum(any(str(cell or "").strip() for cell in row) for row in table.extract())
        >= min(2, table.row_count)
    ]


def _hybrid_text_records(page, tables, common):
    from tools.ocr_support import normalize_ocr_text

    records, table_areas = [], [pymupdf.Rect(table.bbox) for table in tables]
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    for block in page.get_text("dict", sort=True, flags=flags)["blocks"]:
        if block.get("type") != 0:
            continue
        lines, positions = [], []

        def append_run():
            if lines:
                area = pymupdf.Rect(positions[0])
                for position in positions[1:]:
                    area.include_rect(position)
                records.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            **common,
                            "content_type": "page_text",
                            "bbox": json.dumps(list(area)),
                            "extraction_method": "pymupdf",
                        },
                    )
                )
                lines.clear()
                positions.clear()

        for line in block.get("lines", []):
            rect = pymupdf.Rect(line["bbox"])
            if any((rect.tl + rect.br) / 2 in area for area in table_areas):
                append_run()
                continue
            if positions and any(
                positions[-1].y1 <= area.y0
                and rect.y0 >= area.y1
                and min(rect.x1, area.x1) > max(rect.x0, area.x0)
                for area in table_areas
            ):
                append_run()
            value = normalize_ocr_text("".join(span["text"] for span in line.get("spans", [])))
            if value:
                lines.append(value)
                positions.append(rect)
        append_run()
    for index, table in enumerate(tables, 1):
        rows = [
            [normalize_ocr_text(str(cell or "")).replace("\n", " / ") for cell in row]
            for row in table.extract()
        ]
        text = "\n".join("\t".join(row) for row in rows).strip()
        if text:
            records.append(
                Document(
                    page_content=text,
                    metadata={
                        **common,
                        "content_type": "table",
                        "table_index": index,
                        "bbox": json.dumps(list(table.bbox)),
                        "table_data": json.dumps(rows, ensure_ascii=False),
                        "extraction_method": "pymupdf_table",
                    },
                )
            )
    # Keep the established native block order (including columns). Insert tables
    # among those blocks using their geometry rather than appending after prose.
    text_records = [record for record in records if record.metadata["content_type"] != "table"]
    table_records = [record for record in records if record.metadata["content_type"] == "table"]
    from tools.document_visuals import bounds

    for table in sorted(
        table_records, key=lambda record: (bounds(record.metadata)[1], bounds(record.metadata)[0])
    ):
        area = pymupdf.Rect(bounds(table.metadata))
        insertion = len(text_records)
        for index, record in enumerate(text_records):
            other = pymupdf.Rect(bounds(record.metadata))
            if other.y0 >= area.y0 and min(other.x1, area.x1) > max(other.x0, area.x0):
                insertion = index
                break
        text_records.insert(insertion, table)
    return text_records


def _hybrid_regions(page, tables, *, scanned):
    if scanned:
        return [(page.rect, "scanned_page", None)]
    regions = []
    for info in page.get_image_info(hashes=True):
        area = pymupdf.Rect(info["bbox"]) & page.rect
        if min(area.width, area.height) >= 32 and area.get_area() >= page.rect.get_area() * 0.005:
            # PyMuPDF's decoded-pixel digest ignores page placement/caption and
            # catches repeated logos before region rendering or safety limits.
            regions.append((area, "image", info.get("digest")))
    drawings = page.get_drawings()
    if drawings:
        # Neighboring vector objects form a chart/flowchart region. Single rules,
        # page borders, and standard table outlines do not require a model.
        clusters = page.cluster_drawings(drawings=drawings, x_tolerance=40, y_tolerance=40)
        for cluster in clusters:
            area = pymupdf.Rect(cluster) & page.rect
            owned = [
                drawing for drawing in drawings if pymupdf.Rect(drawing["rect"]).intersects(area)
            ]
            if (
                len(owned) < 2 and sum(len(drawing.get("items", [])) for drawing in owned) < 3
            ) or min(area.width, area.height) < 30:
                continue
            if any(
                (area & pymupdf.Rect(table.bbox)).get_area() >= area.get_area() * 0.7
                for table in tables
            ):
                continue
            if any((area & other).get_area() >= area.get_area() * 0.7 for other, _, _ in regions):
                continue
            regions.append((area, "diagram", None))
    return regions


def extract_pdf_hybrid(
    source, *, asset_id, source_name, settings, vlm_mode, deadline, progress=None
):
    from tools.document_visuals import (
        bounds,
        check_deadline,
        finish_records,
        merge_metadata,
        notify,
        setting,
        store_png,
        visual_metadata,
    )

    if vlm_mode not in {"disabled", "automatic", "enhanced"}:
        raise ValueError("Unsupported VLM processing mode.")
    path = Path(source)
    result = visual_metadata(vlm_mode)
    check_deadline(deadline)
    notify(progress, "extracting_document_text", source_type="pdf")
    if vlm_mode == "disabled":
        documents = extract_text_from_pdf(
            str(path),
            asset_id=asset_id,
            source_name=source_name,
            ocr=bool(setting(settings, "PDF_OCR_ENABLED", False)),
            ocr_language=setting(settings, "PDF_OCR_LANGUAGE", "eng"),
            max_ocr_pages=setting(settings, "PDF_MAX_OCR_PAGES", 20),
            structured=True,
            progress=progress,
            ocr_backend=setting(settings, "PDF_OCR_BACKEND", "auto"),
            ocr_binary=setting(settings, "OCR_BINARY"),
            deadline=deadline,
        )
        for document in documents:
            document.metadata.setdefault(
                "extraction_method", "ocr" if document.metadata.get("ocr") else "pymupdf"
            )
        if documents[0].metadata.get("scanned_page_count") and not setting(
            settings, "PDF_OCR_ENABLED", False
        ):
            result["warnings"].append(
                "Scanned PDF pages were detected with OCR disabled; text inside scanned images was not extracted."
            )
        if documents[0].metadata.get("ocr_warning"):
            result["warnings"].append(documents[0].metadata["ocr_warning"])
        notify(
            progress,
            "text_extraction_completed",
            records=[
                record.model_dump()
                for record in documents
                if record.metadata.get("content_type") != "table"
            ],
        )
        notify(
            progress,
            "table_extraction_completed",
            records=[
                record.model_dump()
                for record in documents
                if record.metadata.get("content_type") == "table"
            ],
        )
        return finish_records(documents, vlm_mode=vlm_mode), result
    if not path.is_file():
        raise AssetNotFoundError("The PDF file was not found.")
    documents, artifacts, omissions, page_metadata = [], [], [], {}
    from helper.extraction_checkpoint import ChunkCheckpoint

    checkpoint = ChunkCheckpoint(
        path,
        "pdf-hybrid-text",
        {
            "extraction_contract": _EXTRACTION_CONTRACT,
            "language_data": _ocr_identity(
                setting(settings, "PDF_OCR_ENABLED", False),
                setting(settings, "PDF_OCR_LANGUAGE", "eng"),
                setting(settings, "PDF_OCR_BACKEND", "auto"),
                setting(settings, "OCR_BINARY"),
                deadline,
            ),
            "asset_id": asset_id,
            "source_name": source_name,
            **{
                key: setting(settings, key)
                for key in (
                    "PDF_OCR_ENABLED",
                    "PDF_OCR_LANGUAGE",
                    "PDF_OCR_BACKEND",
                    "OCR_BINARY",
                    "PDF_MAX_OCR_PAGES",
                )
            },
        },
    )
    scanned_count = empty_count = ocr_count = omitted = detected = duplicate_count = 0
    seen_artifacts = set()
    limit = min(128, max(1, int(setting(settings, "VLM_MAX_VISUALS", 32))))
    try:
        with _PDF_LOCK, pymupdf.open(path) as document:
            if not document.is_pdf or document.needs_pass:
                raise InvalidDocumentError("The PDF is invalid or password protected.")
            common = {
                **_document_metadata(document),
                "asset_id": asset_id,
                "source": Path(source_name or path.name).name,
                "source_type": "pdf",
                "modality": "pdf",
                "page_count": document.page_count,
            }
            for page in document:
                check_deadline(deadline)
                warning_start = len(result["warnings"])
                # Text/image/drawing coordinates use the unrotated page space.
                # Normalize this in-memory copy so clip geometry and rendering
                # share that space; the uploaded PDF is never saved or modified.
                if page.rotation:
                    page.set_rotation(0)
                text = _native_text(page)
                image_count, coverage = _page_images(page)
                scanned = _scanned_page(text, image_count, coverage)
                scanned_count += int(scanned)
                tables = _hybrid_tables(page) if text else []
                metadata = {
                    **common,
                    "page": page.number + 1,
                    "table_count": len(tables),
                    "image_count": image_count,
                    "scanned": scanned,
                    "ocr": False,
                }
                cached = _cached_page(
                    checkpoint,
                    page,
                    scanned=scanned,
                    ocr=setting(settings, "PDF_OCR_ENABLED", False),
                    language=setting(settings, "PDF_OCR_LANGUAGE", "eng"),
                    backend=setting(settings, "PDF_OCR_BACKEND", "auto"),
                    binary=setting(settings, "OCR_BINARY"),
                    deadline=deadline,
                )
                page_records = (
                    cached if cached is not None else _hybrid_text_records(page, tables, metadata)
                )
                if cached is not None:
                    metadata["ocr"] = any(record.metadata.get("ocr") for record in cached)
                    ocr_count += int(metadata["ocr"])
                    notify(progress, "reusing_extraction_checkpoint", page=page.number + 1)
                elif scanned and setting(settings, "PDF_OCR_ENABLED", False):
                    if ocr_count >= int(setting(settings, "PDF_MAX_OCR_PAGES", 20)):
                        result["warnings"].append(
                            "The OCR page limit was reached; remaining scanned pages use visual fallback."
                        )
                    else:
                        notify(progress, "extracting_document_ocr", page=page.number + 1)
                        try:
                            from tools.ocr_support import recognize_page

                            ocr_text = recognize_page(
                                page,
                                language=setting(settings, "PDF_OCR_LANGUAGE", "eng"),
                                backend=setting(settings, "PDF_OCR_BACKEND", "auto"),
                                binary=setting(settings, "OCR_BINARY"),
                                deadline=deadline,
                            )
                            if ocr_text:
                                metadata["ocr"] = True
                                extra = _complementary_ocr(
                                    "\n".join(record.page_content for record in page_records),
                                    ocr_text,
                                    native_fragments=_table_cell_lines(tables),
                                )
                                if extra:
                                    page_records.append(_ocr_record(extra, metadata, page.rect))
                            else:
                                result["warnings"].append(
                                    f"OCR returned empty text on page {page.number + 1}; native text and tables were retained and scanned content may be missing."
                                )
                        except Exception as error:
                            result["warnings"].append(
                                f"Page {page.number + 1}: PDF OCR is unavailable or failed; scanned pages use visual fallback when available. "
                                + str(error)[:350]
                            )
                        ocr_count += 1
                elif scanned:
                    result["warnings"].append(
                        "Scanned PDF pages were detected with OCR disabled; visual understanding is used as a fallback when available."
                    )
                page_warnings = result["warnings"][warning_start:]
                if page_warnings:
                    for record in page_records:
                        record.metadata["ocr_warning"] = " ".join(page_warnings)[:1500]
                empty_count += int(not page_records)
                if not (
                    scanned and setting(settings, "PDF_OCR_ENABLED", False) and not metadata["ocr"]
                ):
                    checkpoint.save(page.number, page_records)
                documents.extend(page_records)
                page_metadata[page.number + 1] = metadata
                if image_count or page.get_drawings():
                    notify(progress, "detecting_visual_content", page=page.number + 1)
                for rect, kind, image_digest in _hybrid_regions(page, tables, scanned=scanned):
                    check_deadline(deadline)
                    detected += 1
                    fingerprint = ("image", image_digest) if image_digest else None
                    if fingerprint is not None:
                        if fingerprint in seen_artifacts:
                            duplicate_count += 1
                            omissions.append(
                                {
                                    "reference": "",
                                    "artifact_type": kind,
                                    "page": page.number + 1,
                                    "bbox": list(rect),
                                    "processing_status": "duplicate",
                                    "reason": "identical_image",
                                }
                            )
                            continue
                        # Deadline and a bounded digest set prevent document size
                        # from producing an unbounded artifact-selection cache.
                        if len(seen_artifacts) < 8192:
                            seen_artifacts.add(fingerprint)
                    nearby = page.get_textbox(
                        pymupdf.Rect(rect.x0 - 12, rect.y0 - 65, rect.x1 + 12, rect.y1 + 65)
                        & page.rect
                    ).strip()[:2000]
                    caption_lines = [
                        line.strip()
                        for line in nearby.splitlines()
                        if re.match(
                            r"^(fig(?:ure)?\b|diagram\b|chart\b|flowchart\b)", line.strip(), re.I
                        )
                    ]
                    caption = " ".join(caption_lines)[:512]
                    priority = 1.0 if kind in {"diagram", "scanned_page"} or caption else 0.5
                    weakest = (
                        min(range(len(artifacts)), key=lambda index: artifacts[index]["priority"])
                        if artifacts
                        else None
                    )
                    if len(artifacts) >= limit and priority <= artifacts[weakest]["priority"]:
                        omitted += 1
                        omissions.append(
                            {
                                "reference": "",
                                "artifact_type": kind,
                                "page": page.number + 1,
                                "bbox": list(rect),
                                "caption": caption,
                                "processing_status": "skipped_budget",
                                "reason": "render_budget",
                            }
                        )
                        continue
                    dimension = min(
                        2048, max(256, int(setting(settings, "VLM_IMAGE_MAX_DIMENSION", 1280)))
                    )
                    expanded = (
                        pymupdf.Rect(rect.x0 - 5, rect.y0 - 5, rect.x1 + 5, rect.y1 + 5) & page.rect
                    )
                    scale = min(2.0, dimension / max(expanded.width, expanded.height))
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), clip=expanded, alpha=False
                    )
                    png = pixmap.tobytes("png")
                    if fingerprint is None:
                        fingerprint = ("rendered", hashlib.sha256(png).digest())
                        if fingerprint in seen_artifacts:
                            duplicate_count += 1
                            continue
                        if len(seen_artifacts) < 8192:
                            seen_artifacts.add(fingerprint)
                    reference = store_png(path, png)
                    artifact = {
                        "reference": reference,
                        "artifact_type": kind,
                        "page": page.number + 1,
                        "bbox": list(rect),
                        "nearby_text": nearby,
                        "caption": caption,
                        "priority": priority,
                    }
                    if kind != "image" or caption:
                        artifact["meaningful"] = True
                    if len(artifacts) >= limit:
                        omissions.append(
                            {
                                **artifacts[weakest],
                                "processing_status": "skipped_budget",
                                "reason": "render_budget",
                            }
                        )
                        artifacts[weakest] = artifact
                        omitted += 1
                    else:
                        artifacts.append(artifact)
    except (AssetNotFoundError, InvalidDocumentError, ProcessingError):
        raise
    except Exception as exc:
        raise InvalidDocumentError("The PDF could not be extracted.") from exc
    notify(
        progress,
        "text_extraction_completed",
        records=[
            record.model_dump()
            for record in documents
            if record.metadata.get("content_type") != "table"
        ],
    )
    notify(
        progress,
        "table_extraction_completed",
        records=[
            record.model_dump()
            for record in documents
            if record.metadata.get("content_type") == "table"
        ],
    )
    artifacts.sort(
        key=lambda artifact: (artifact["page"], artifact["bbox"][1], artifact["bbox"][0])
    )
    result["files"] = [artifact["reference"] for artifact in artifacts]
    if omitted:
        result["warnings"].append(
            f"After duplicate filtering and priority selection, the visual artifact limit omitted {omitted} additional PDF regions."
        )
    if artifacts:
        from tools.visual_understanding import analyze_artifacts

        visual_docs, visual_result = analyze_artifacts(
            path,
            artifacts + omissions,
            asset_id=asset_id,
            source_name=source_name,
            source_type="pdf",
            vlm_mode=vlm_mode,
            settings=settings,
            deadline=deadline,
            progress=progress,
        )
        for record in visual_docs:
            original = page_metadata.get(record.metadata.get("page"), {})
            record.metadata = {**original, **record.metadata}
        documents.extend(visual_docs)
        merge_metadata(result, visual_result)
        if visual_docs:
            notify(progress, "merging_multimodal", source_type="pdf")
    result["detected_visual_count"] = detected
    result["duplicate_visual_count"] += duplicate_count
    for record in documents:
        record.metadata.update(scanned_page_count=scanned_count, empty_page_count=empty_count)
    documents.sort(
        key=lambda record: (
            record.metadata["page"],
            round(bounds(record.metadata)[1], 1),
            bounds(record.metadata)[0],
            int(record.metadata.get("content_type") == "visual_description"),
        )
    )
    if not documents:
        raise EmptyContentError(
            "The PDF contains no readable text or visual evidence. Enable OCR or configure visual understanding for scanned pages."
        )
    return finish_records(documents, vlm_mode=vlm_mode), result
