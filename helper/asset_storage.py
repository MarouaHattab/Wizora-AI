import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath

from langchain_core.documents import Document

from . import asset_paths
from .asset_paths import asset_file, validate_asset_id
from .ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    InvalidDocumentError,
    StorageError,
)

_SOURCE_TYPES = {"pdf", "docx", "audio", "video", "text"}
SOURCE_METADATA_KEYS = (
    "page",
    "page_count",
    "start",
    "end",
    "segment_id",
    "language",
    "modality",
    "content_type",
    "element_id",
    "paragraph",
    "section",
    "section_path",
    "heading_level",
    "table_index",
    "table_count",
    "image_count",
    "image_references",
    "document_title",
    "document_author",
    "document_subject",
    "document_created",
    "document_modified",
    "document_keywords",
    "document_category",
    "document_language",
    "document_creator",
    "document_producer",
    "document_creationDate",
    "document_modDate",
    "scanned",
    "ocr",
    "ocr_warning",
    "scanned_page_count",
    "empty_page_count",
    "timestamp_precision",
    "encoding",
    "line_start",
    "line_end",
    "frame_index",
    "frame_reference",
    "vision_model",
    "vision_provider",
    "artifact_reference",
    "artifact_type",
    "visual_artifact_type",
    "extraction_method",
    "processing_status",
    "confidence",
    "uncertainty",
    "visual_structure",
    "bbox",
    "caption",
    "nearby_text",
    "table_data",
    "vlm_mode",
    "requested_vlm_mode",
    "source_file_id",
    "origin_record_id",
    "origin_locations",
    "semantic_duplicate_count",
)
_BOOLEAN_METADATA = {"scanned", "ocr"}
_INTEGER_METADATA = {
    "page",
    "page_count",
    "paragraph",
    "heading_level",
    "table_index",
    "table_count",
    "image_count",
    "scanned_page_count",
    "empty_page_count",
    "line_start",
    "line_end",
    "frame_index",
    "semantic_duplicate_count",
}
_TIMESTAMP_METADATA = {"start", "end"}


def _image_reference(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise InvalidDocumentError("Invalid source image reference")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ":" in value
        or len(path.parts) < 2
        or path.parts[0] != "images"
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(ord(char) < 32 for char in value)
    ):
        raise InvalidDocumentError("Source image references must stay inside the asset")


def validate_source_metadata(metadata, source_type=None):
    if not isinstance(metadata, dict):
        raise InvalidDocumentError("Invalid source metadata")
    source_type = source_type or metadata.get("source_type")
    if source_type is not None and (
        not isinstance(source_type, str) or source_type not in _SOURCE_TYPES
    ):
        raise InvalidDocumentError("Invalid source type")
    for key in SOURCE_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if key in _BOOLEAN_METADATA:
            if type(value) is not bool:
                raise InvalidDocumentError("Invalid source processing flag")
        elif key in _INTEGER_METADATA:
            minimum = (
                0
                if key in {"table_count", "image_count", "scanned_page_count", "empty_page_count"}
                else 1
            )
            if type(value) is not int or value < minimum:
                raise InvalidDocumentError("Invalid source element number")
            if key == "heading_level" and value > 9:
                raise InvalidDocumentError("Invalid heading hierarchy")
        elif key in _TIMESTAMP_METADATA:
            if (
                type(value) not in (int, float)
                or isinstance(value, float)
                and not math.isfinite(value)
                or value < 0
            ):
                raise InvalidDocumentError("Invalid source timestamp")
        elif key == "confidence":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise InvalidDocumentError("Invalid visual confidence")
        elif key == "segment_id":
            if not (
                (type(value) is int and value >= 0) or (isinstance(value, str) and value.strip())
            ):
                raise InvalidDocumentError("Invalid source segment identifier")
        elif not isinstance(value, str) or not value.strip() or len(value) > 64000:
            raise InvalidDocumentError("Invalid source metadata text")
        if key in {"element_id", "segment_id"} and isinstance(value, str):
            if len(value) > 256 or any(ord(char) < 32 for char in value):
                raise InvalidDocumentError("Invalid source element identifier")
    modality = metadata.get("modality")
    if modality is not None and (
        not isinstance(modality, str)
        or modality not in _SOURCE_TYPES
        or (source_type is not None and modality != source_type)
    ):
        raise InvalidDocumentError("Source modality does not match its asset")
    if source_type == "pdf" and (type(metadata.get("page")) is not int or metadata["page"] < 1):
        raise InvalidDocumentError("PDF records require an original page number")
    if source_type == "docx" and not isinstance(metadata.get("element_id"), str):
        raise InvalidDocumentError("DOCX records require a stable element identifier")
    if source_type == "docx" and metadata.get("page") is not None:
        raise InvalidDocumentError(
            "DOCX records must use stable element references instead of pages"
        )
    if "page_count" in metadata and metadata["page_count"] < metadata.get("page", 1):
        raise InvalidDocumentError("Invalid PDF page count")
    start, end = metadata.get("start"), metadata.get("end")
    if (start is None) != (end is None) or (start is not None and end < start):
        raise InvalidDocumentError("Invalid transcription timestamps")
    allowed_content = (
        {None, "transcript", "visual_description"}
        if source_type == "video"
        else {None, "transcript"}
    )
    if source_type in {"audio", "video"} and metadata.get("content_type") not in allowed_content:
        raise InvalidDocumentError("Invalid media evidence content type")
    frame_fields = {"frame_index", "frame_reference"}
    visual_fields = frame_fields | {"vision_model", "vision_provider", "artifact_reference"}
    visual = metadata.get("content_type") == "visual_description"
    if visual:
        if source_type not in {"pdf", "docx", "video"} or not {
            "vision_provider",
            "vision_model",
        }.issubset(metadata):
            raise InvalidDocumentError("Visual observations require model and source provenance")
        if source_type == "video" and (
            not frame_fields.issubset(metadata)
            or metadata.get("timestamp_precision") != "frame"
            or start is None
        ):
            raise InvalidDocumentError(
                "Video visual observations require complete frame provenance"
            )
        if source_type in {"pdf", "docx"} and (
            "artifact_reference" not in metadata or frame_fields.intersection(metadata)
        ):
            raise InvalidDocumentError(
                "Document visual observations require an original artifact reference"
            )
        if metadata.get("vision_provider") not in {"OLLAMA", "GOOGLE_GENAI"}:
            raise InvalidDocumentError("Unsupported vision provider")
    if not visual and visual_fields.intersection(metadata):
        raise InvalidDocumentError("Only visual observations may contain frame and vision metadata")
    if metadata.get("artifact_reference") is not None:
        _image_reference(metadata["artifact_reference"])
    for key in ("vlm_mode", "requested_vlm_mode"):
        if key in metadata and metadata[key] not in {"disabled", "automatic", "enhanced"}:
            raise InvalidDocumentError("Invalid source visual processing mode")
    if (metadata.get("source_file_id") is None) != (metadata.get("origin_record_id") is None):
        raise InvalidDocumentError("Combined evidence requires complete original ownership")
    if metadata.get("source_file_id") is not None:
        try:
            validate_asset_id(metadata["source_file_id"])
        except Exception as exc:
            raise InvalidDocumentError("Invalid original source file identifier") from exc
        if len(metadata["origin_record_id"]) > 128:
            raise InvalidDocumentError("Invalid original source record identifier")
    if "origin_locations" in metadata:
        try:
            origins = json.loads(metadata["origin_locations"])
        except (TypeError, ValueError) as exc:
            raise InvalidDocumentError("Invalid original source locations") from exc
        if not isinstance(origins, list) or not origins or len(origins) > 1024:
            raise InvalidDocumentError("Invalid original source locations")
        for origin in origins:
            if not isinstance(origin, dict) or set(origin) - {
                "source_file_id",
                "origin_record_id",
                "source",
                "source_type",
                "page",
                "page_count",
                "element_id",
                "paragraph",
                "section",
                "table_index",
                "start",
                "end",
                "timestamp_precision",
                "segment_id",
                "artifact_reference",
                "frame_reference",
                "frame_index",
                "line_start",
                "line_end",
                "caption",
            }:
                raise InvalidDocumentError("Invalid original source locator")
            try:
                validate_asset_id(origin.get("source_file_id"))
            except Exception as exc:
                raise InvalidDocumentError("Invalid original source file identifier") from exc
            if (
                origin.get("source_type") not in _SOURCE_TYPES
                or not isinstance(origin.get("origin_record_id"), str)
                or not origin["origin_record_id"]
                or len(origin["origin_record_id"]) > 128
                or not isinstance(origin.get("source"), str)
                or not origin["source"].strip()
                or len(origin["source"]) > 512
                or any(ord(char) < 32 for char in origin["source"])
            ):
                raise InvalidDocumentError("Invalid original source locator")
            for key in (
                "page",
                "page_count",
                "paragraph",
                "table_index",
                "frame_index",
                "line_start",
                "line_end",
            ):
                if key in origin and (type(origin[key]) is not int or origin[key] < 1):
                    raise InvalidDocumentError("Invalid original source element number")
            if "artifact_reference" in origin:
                _image_reference(origin["artifact_reference"])
            if "frame_reference" in origin:
                reference = origin["frame_reference"]
                if (
                    not isinstance(reference, str)
                    or len(PurePosixPath(reference).parts) != 2
                    or not reference.startswith("frames/")
                    or ".." in reference
                    or "\\" in reference
                    or ":" in reference
                ):
                    raise InvalidDocumentError("Invalid original frame reference")
            start, end = origin.get("start"), origin.get("end")
            if (
                (start is None) != (end is None)
                or start is not None
                and (
                    type(start) not in (float, int)
                    or type(end) not in (float, int)
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end < start
                )
            ):
                raise InvalidDocumentError("Invalid original source timestamp")
    for key in ("bbox", "visual_structure", "table_data"):
        if key not in metadata:
            continue
        try:
            value = json.loads(metadata[key])
        except (TypeError, ValueError) as exc:
            raise InvalidDocumentError("Invalid structured extraction metadata") from exc
        if key == "bbox":
            if (
                not isinstance(value, list)
                or len(value) != 4
                or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
                or value[2] < value[0]
                or value[3] < value[1]
            ):
                raise InvalidDocumentError("Invalid source bounding box")
        elif not isinstance(value, (dict, list)):
            raise InvalidDocumentError("Structured extraction metadata must be an object or array")
    precision = metadata.get("timestamp_precision")
    if precision is not None and (
        precision not in {"segment", "audio_window", "frame"}
        or start is None
        or source_type not in {"audio", "video"}
    ):
        raise InvalidDocumentError("Invalid transcription timestamp precision")
    if precision == "frame" and (
        source_type != "video"
        or metadata.get("content_type") != "visual_description"
        or start != end
    ):
        raise InvalidDocumentError("Frame observations require a single video timestamp")
    if metadata.get("line_end", 1) < metadata.get("line_start", 1):
        raise InvalidDocumentError("Invalid text line range")
    reference = metadata.get("frame_reference")
    if reference is not None:
        path = PurePosixPath(reference)
        if (
            source_type != "video"
            or path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] != "frames"
            or any(part in {"", ".", ".."} for part in reference.split("/"))
            or "\\" in reference
            or ":" in reference
            or any(ord(char) < 32 for char in reference)
        ):
            raise InvalidDocumentError("Frame references must stay inside the asset")
    for key in ("section_path", "image_references"):
        if key not in metadata:
            continue
        try:
            values = json.loads(metadata[key])
        except (TypeError, ValueError) as exc:
            raise InvalidDocumentError("Source reference collections must be JSON strings") from exc
        if not isinstance(values, list) or len(values) > 10000:
            raise InvalidDocumentError("Invalid source reference collection")
        for value in values:
            if key == "image_references":
                if isinstance(value, str):
                    _image_reference(value)
                    continue
                if not isinstance(value, dict) or set(value) - {
                    "element_id",
                    "image_id",
                    "content_type",
                    "byte_count",
                    "status",
                    "reference",
                }:
                    raise InvalidDocumentError("Invalid embedded image metadata")
                if (
                    any(
                        not isinstance(value.get(field), str)
                        or not value[field].strip()
                        or len(value[field]) > 256
                        for field in ("element_id", "image_id", "content_type")
                    )
                    or not value["content_type"].startswith("image/")
                    or type(value.get("byte_count")) is not int
                    or value["byte_count"] < 0
                    or value.get("status")
                    not in {"available", "unsupported_format", "too_large", "limit_reached"}
                    or value["status"] == "available"
                    and "reference" not in value
                ):
                    raise InvalidDocumentError("Invalid embedded image metadata")
                if "reference" in value:
                    _image_reference(value["reference"])
            elif not isinstance(value, str) or not value.strip():
                raise InvalidDocumentError("Invalid section hierarchy")


def flatten_documents(documents):
    text = "\n\n".join(
        document.page_content.strip() for document in documents if document.page_content.strip()
    )
    if not text:
        raise EmptyContentError("The source contains no text")
    return text


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def source_records(documents, asset_id, source_type, source_name, asset_version):
    validate_asset_id(asset_id)
    if (
        source_type not in _SOURCE_TYPES | {"collection"}
        or not isinstance(source_name, str)
        or not source_name.strip()
        or not isinstance(asset_version, str)
        or not asset_version
    ):
        raise InvalidDocumentError("Invalid source configuration")
    records = []
    for document in documents:
        if not isinstance(document, Document) or not isinstance(document.page_content, str):
            raise InvalidDocumentError("Invalid source document")
        if document.metadata.get("asset_id", asset_id) != asset_id:
            raise InvalidDocumentError("Source record ownership does not match its asset")
        text = document.page_content.strip()
        if not text:
            continue
        record_type = (
            document.metadata.get("source_type") if source_type == "collection" else source_type
        )
        record_source = (
            document.metadata.get("source") if source_type == "collection" else source_name
        )
        if (
            record_type not in _SOURCE_TYPES
            or not isinstance(record_source, str)
            or not record_source.strip()
        ):
            raise InvalidDocumentError("Combined evidence requires original file metadata")
        metadata = {
            "asset_id": asset_id,
            "source_type": record_type,
            "source": record_source,
            "asset_version": asset_version,
            "record_index": len(records),
        }
        for key in SOURCE_METADATA_KEYS:
            value = document.metadata.get(key)
            if value is not None:
                if not isinstance(value, (str, int, float, bool)):
                    raise InvalidDocumentError("Invalid source metadata")
                if isinstance(value, float) and not math.isfinite(value):
                    raise InvalidDocumentError("Invalid source metadata")
                metadata[key] = value
        validate_source_metadata(metadata, record_type)
        metadata["record_id"] = _digest({"metadata": metadata, "text": text})
        records.append({"page_content": text, "metadata": metadata})
    if not records:
        raise EmptyContentError("The source contains no text")
    return records


def write_source_records(documents, destination, asset_id, source_type, source_name, asset_version):
    records = source_records(documents, asset_id, source_type, source_name, asset_version)
    payload = {
        "schema_version": 1,
        "status": "ready",
        "asset_id": asset_id,
        "asset_version": asset_version,
        "records_hash": _digest(records),
        "documents": records,
    }
    try:
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise StorageError("Unable to store source records") from exc
    return [Document(**record) for record in records]


def _require_ready(asset_id):
    validate_asset_id(asset_id)
    root = asset_paths.ASSETS_ROOT.resolve()
    directory = (root / ".jobs").resolve()
    path = (directory / f"{asset_id}.json").resolve()
    if not directory.is_relative_to(root) or not path.is_relative_to(directory):
        raise StorageError("Invalid processing status directory")
    try:
        # Windows replacement and open read handles must share the catalog lock.
        from .asset_catalog import _lock as status_lock
        from .asset_catalog import _status_io

        with status_lock:
            payload = _status_io(lambda: path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except UnicodeError as exc:
        raise AssetNotFoundError("The asset is not ready") from exc
    except OSError as exc:
        raise StorageError("Unable to read asset processing status") from exc
    try:
        status = json.loads(payload)
    except ValueError as exc:
        raise AssetNotFoundError("The asset is not ready") from exc
    if (
        not isinstance(status, dict)
        or status.get("id") != asset_id
        or status.get("status") != "ready"
    ):
        raise AssetNotFoundError("The asset is not ready")


def _read_text(asset_id):
    _require_ready(asset_id)
    path = asset_file(asset_id, "extracted_text.txt")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AssetNotFoundError("Asset text not found") from exc
    except (OSError, UnicodeError) as exc:
        raise StorageError("Unable to read asset text") from exc
    if not text.strip():
        raise EmptyContentError("Asset contains no text")
    return text


def load_asset_documents(asset_id, context=None):
    text = _read_text(asset_id)
    if context is not None and text != context:
        raise InvalidDocumentError("Asset content changed; start a new session")
    return _validated_asset_documents(asset_id, text)


def _validated_asset_documents(asset_id, text):
    path = asset_file(asset_id, "documents.json")
    if not path.exists():
        version = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records = source_records(
            [Document(page_content=text)], asset_id, "text", "extracted_text.txt", version
        )
        return [Document(**record) for record in records]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("status") != "ready"
        ):
            raise InvalidDocumentError("Asset records are not ready")
        records = payload["documents"]
        if payload["asset_id"] != asset_id or not isinstance(records, list) or not records:
            raise InvalidDocumentError("Invalid asset records")
        if payload["records_hash"] != _digest(records):
            raise InvalidDocumentError("Asset records are incomplete or changed")
        documents = []
        for number, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
                raise InvalidDocumentError("Invalid source record metadata")
            metadata = record["metadata"]
            if (
                metadata.get("asset_id") != asset_id
                or metadata.get("asset_version") != payload["asset_version"]
            ):
                raise InvalidDocumentError("Asset record ownership does not match")
            expected = source_records(
                [Document(page_content=record["page_content"], metadata=metadata)],
                asset_id,
                metadata["source_type"],
                metadata["source"],
                metadata["asset_version"],
            )[0]
            expected["metadata"]["record_index"] = number
            expected["metadata"].pop("record_id")
            expected["metadata"]["record_id"] = _digest(
                {"metadata": expected["metadata"], "text": expected["page_content"]}
            )
            if expected != record:
                raise InvalidDocumentError("Invalid source record metadata")
            documents.append(Document(**record))
        if flatten_documents(documents) != text:
            raise InvalidDocumentError("Structured records do not match extracted text")
        return documents
    except (KeyError, TypeError, ValueError, OSError, UnicodeError) as exc:
        raise InvalidDocumentError("Unable to read valid asset records") from exc


def load_asset_source(asset_id):
    text = _read_text(asset_id)
    return text, load_asset_documents(asset_id, context=text)


def load_asset_text(asset_id):
    return load_asset_source(asset_id)[0]
