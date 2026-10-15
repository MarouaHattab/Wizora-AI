import hashlib
import json
import math

from langchain_core.documents import Document

from .asset_paths import asset_file, validate_asset_id
from .ingestion_errors import AssetNotFoundError, EmptyContentError, InvalidDocumentError, StorageError


_SOURCE_TYPES = {"pdf", "audio", "video", "text"}


def flatten_documents(documents):
    text = "\n\n".join(document.page_content.strip() for document in documents if document.page_content.strip())
    if not text:
        raise EmptyContentError("The source contains no text")
    return text


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_records(documents, asset_id, source_type, source_name, asset_version):
    validate_asset_id(asset_id)
    if (
        source_type not in _SOURCE_TYPES or not isinstance(source_name, str) or not source_name.strip()
        or not isinstance(asset_version, str) or not asset_version
    ):
        raise InvalidDocumentError("Invalid source configuration")
    records = []
    for document in documents:
        if not isinstance(document, Document) or not isinstance(document.page_content, str):
            raise InvalidDocumentError("Invalid source document")
        text = document.page_content.strip()
        if not text:
            continue
        metadata = {
            "asset_id": asset_id, "source_type": source_type, "source": source_name,
            "asset_version": asset_version, "record_index": len(records),
        }
        for key in ("page", "page_count", "start", "end", "segment_id", "language"):
            value = document.metadata.get(key)
            if value is not None:
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise InvalidDocumentError("Invalid source metadata")
                if isinstance(value, float) and not math.isfinite(value):
                    raise InvalidDocumentError("Invalid source metadata")
                metadata[key] = value
        if source_type == "pdf" and (
            type(metadata.get("page")) is not int or metadata["page"] < 1
        ):
            raise InvalidDocumentError("PDF records require an original page number")
        if "page_count" in metadata and (
            type(metadata["page_count"]) is not int or metadata["page_count"] < metadata.get("page", 1)
        ):
            raise InvalidDocumentError("Invalid PDF page count")
        if "start" in metadata or "end" in metadata:
            start, end = metadata.get("start"), metadata.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start <= end:
                raise InvalidDocumentError("Invalid transcription timestamps")
        metadata["record_id"] = _digest({"metadata": metadata, "text": text})
        records.append({"page_content": text, "metadata": metadata})
    if not records:
        raise EmptyContentError("The source contains no text")
    return records


def write_source_records(documents, destination, asset_id, source_type, source_name, asset_version):
    records = source_records(documents, asset_id, source_type, source_name, asset_version)
    payload = {
        "schema_version": 1, "status": "ready", "asset_id": asset_id,
        "asset_version": asset_version, "records_hash": _digest(records), "documents": records,
    }
    try:
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise StorageError("Unable to store source records") from exc
    return [Document(**record) for record in records]


def _read_text(asset_id):
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
    path = asset_file(asset_id, "documents.json")
    if not path.exists():
        version = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records = source_records([Document(page_content=text)], asset_id, "text", "extracted_text.txt", version)
        return [Document(**record) for record in records]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("status") != "ready":
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
            if metadata.get("asset_id") != asset_id or metadata.get("asset_version") != payload["asset_version"]:
                raise InvalidDocumentError("Asset record ownership does not match")
            expected = source_records(
                [Document(page_content=record["page_content"], metadata=metadata)], asset_id,
                metadata["source_type"], metadata["source"], metadata["asset_version"],
            )[0]
            expected["metadata"]["record_index"] = number
            expected["metadata"].pop("record_id")
            expected["metadata"]["record_id"] = _digest({"metadata": expected["metadata"], "text": expected["page_content"]})
            if expected != record:
                raise InvalidDocumentError("Invalid source record metadata")
            documents.append(Document(**record))
        if flatten_documents(documents) != text:
            raise InvalidDocumentError("Structured records do not match extracted text")
        return documents
    except (KeyError, TypeError, ValueError, OSError, UnicodeError) as exc:
        raise InvalidDocumentError("Unable to read valid asset records") from exc


def load_asset_text(asset_id):
    text = _read_text(asset_id)
    load_asset_documents(asset_id, context=text)
    return text
