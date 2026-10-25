import hashlib
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import BoundedSemaphore, RLock
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.documents import Document

from helper import asset_paths, get_settings
from helper.asset_catalog import _now, _status_io, public_job, read_job, write_job
from helper.asset_http import processing_http_error
from helper.asset_paths import asset_directory, source_filename, validate_asset_id
from helper.asset_storage import flatten_documents, load_asset_documents, write_source_records
from helper.ingestion import (
    cleanup_staging,
    prepare_retry,
    prepare_upload,
    process_prepared,
    source_kind,
)
from helper.ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    StorageError,
)
from helper.vlm_policy import normalize_vlm_mode

_lock = RLock()
_slots = BoundedSemaphore(2)
_ORIGIN_KEYS = (
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
)


@dataclass(frozen=True)
class PreparedBatch:
    id: str
    prepared: tuple
    job_id: str


def _path(identifier):
    validate_asset_id(identifier)
    root = asset_paths.ASSETS_ROOT.resolve()
    folder = (root / ".batches").resolve()
    if not folder.is_relative_to(root):
        raise StorageError("Invalid batch storage directory")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (identifier + ".json")


def _write(identifier, **updates):
    with _lock:
        path = _path(identifier)
        temporary = path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            value = (
                json.loads(_status_io(lambda: path.read_text(encoding="utf-8")))
                if path.is_file()
                else {"id": identifier, "created_at": _now(), "warnings": []}
            )
            if not isinstance(value, dict) or value.get("id") != identifier:
                raise StorageError("Invalid upload batch")
            value.update(updates, updated_at=_now())
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            _status_io(lambda: temporary.replace(path))
            return value
        except (OSError, ValueError) as exc:
            raise StorageError("Unable to store upload batch") from exc
        finally:
            temporary.unlink(missing_ok=True)


def read_batch(identifier):
    try:
        with _lock:
            value = json.loads(_status_io(lambda: _path(identifier).read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise AssetNotFoundError("Upload batch not found") from exc
    except (OSError, ValueError) as exc:
        raise StorageError("Unable to read upload batch") from exc
    if not isinstance(value, dict) or value.get("id") != identifier:
        raise StorageError("Invalid upload batch")
    value["files"] = [public_job(read_job(file_id)) for file_id in value["file_ids"]]
    combined = read_job(value["combined_asset_id"]) if value.get("combined_asset_id") else None
    value["combined_status"] = combined.get("status") if combined else None
    if value.get("status") in {"queued", "processing"} and all(
        row.get("status") in {"ready", "failed"} for row in value["files"]
    ):
        if combined is None or combined.get("status") in {"ready", "failed"}:
            ready = sum(row["status"] == "ready" for row in value["files"])
            value["status"] = (
                "failed"
                if not ready
                else "partial"
                if ready < len(value["files"]) or combined and combined["status"] == "failed"
                else "ready"
            )
    return {key: item for key, item in value.items() if key != "file_ids"}


def _file_size(upload):
    if type(getattr(upload, "size", None)) is int:
        return max(0, upload.size)
    try:
        original = upload.file.tell()
        upload.file.seek(0, 2)
        size = upload.file.tell()
        upload.file.seek(original)
        return size
    except (OSError, AttributeError):
        return 0


def _failure(file_id, name, kind, size, mode, error):
    public = error if isinstance(error, HTTPException) else processing_http_error(error)
    return write_job(
        file_id,
        source_name=name,
        source_type=kind,
        size_bytes=size,
        status="failed",
        requested_vlm_mode=mode,
        vlm_mode="disabled",
        index_status="deferred",
        stages=[{"name": "validation", "status": "failed", "at": _now()}],
        error={"category": type(error).__name__, "detail": str(public.detail)},
    )


def prepare_batch(files, organization_mode="combined", vlm_mode="disabled", file_configs=None):
    from helper.job_results import fail_job, start_job

    # Retain the ignored argument for older Python callers. New and legacy
    # upload requests always produce a batch-scoped combined context.
    organization_mode = "combined"
    mode = normalize_vlm_mode(vlm_mode)
    settings = get_settings()
    maximum = int(getattr(settings, "MAX_BATCH_FILES", 32))
    if not files or len(files) > maximum:
        raise HTTPException(status_code=422, detail=f"Select between one and {maximum} files")
    sizes = [_file_size(upload) for upload in files]
    if sum(sizes) > int(getattr(settings, "MAX_BATCH_BYTES", settings.MAX_UPLOAD_BYTES * 4)):
        raise HTTPException(
            status_code=413, detail="The batch exceeds the configured total upload size limit"
        )
    overrides = {}
    for config in file_configs or []:
        if (
            not isinstance(config, dict)
            or set(config) != {"index", "vlm_mode"}
            or type(config["index"]) is not int
            or not 0 <= config["index"] < len(files)
            or config["index"] in overrides
        ):
            raise HTTPException(
                status_code=422,
                detail="Each file configuration requires a unique valid index and vlm_mode",
            )
        overrides[config["index"]] = normalize_vlm_mode(config["vlm_mode"])
    batch_id, identifiers = str(uuid4()), [str(uuid4()) for _ in files]
    combined_id = str(uuid4())
    job_id = start_job(
        batch_id,
        "Upload batch",
        "batch",
        "batch_ingestion",
        processing_config={"organization_mode": organization_mode, "vlm_mode": mode},
        source_file_ids=identifiers,
    )
    _write(
        batch_id,
        status="queued",
        organization_mode=organization_mode,
        file_ids=identifiers,
        combined_asset_id=combined_id,
        job_id=job_id,
    )
    prepared = []
    try:
        for index, (upload, file_id, size) in enumerate(zip(files, identifiers, sizes)):
            requested = overrides.get(index, mode)
            name, kind, validated = f"file-{index + 1}", "unsupported", False
            try:
                name = source_filename(upload.filename)
                kind, suffix = source_kind(name, upload.content_type)
                validated = True
                item = prepare_upload(upload.file, file_id, name, kind, suffix, requested)
                write_job(file_id, batch_id=batch_id, size_bytes=size)
                prepared.append(item)
            except Exception as exc:
                _failure(file_id, name, kind, size, requested, exc)
                if not validated:
                    failed = start_job(
                        file_id,
                        name,
                        kind,
                        "ingestion_validation",
                        processing_config={"vlm_mode": requested},
                        source_file_ids=[file_id],
                    )
                    fail_job(
                        failed,
                        {
                            "category": type(exc).__name__,
                            "detail": "The uploaded file failed validation.",
                        },
                    )
        if combined_id:
            write_job(
                combined_id,
                status="queued",
                source_name="Combined learning context",
                source_type="collection",
                batch_id=batch_id,
                source_file_ids=identifiers,
                organization_mode="combined",
                files=[],
                stages=[],
            )
        return PreparedBatch(batch_id, tuple(prepared), job_id)
    except BaseException as exc:
        for item in prepared:
            cleanup_staging(item.staging)
        _write(batch_id, status="failed")
        fail_job(job_id, {"category": type(exc).__name__, "detail": "Batch preparation failed."})
        raise


def _process_file(prepared):
    with _slots:
        try:
            return process_prepared(prepared)
        except Exception as exc:
            public = exc if isinstance(exc, HTTPException) else processing_http_error(exc)
            write_job(
                prepared.asset_id,
                status="failed",
                error={"category": type(exc).__name__, "detail": str(public.detail)},
            )
            return None


def prepare_batch_resume(identifier):
    from helper.job_results import start_job

    with _lock:
        snapshot = read_batch(identifier)
        if snapshot.get("status") in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="This batch is already processing.")
        prepared, warnings = [], []
        for row in snapshot["files"]:
            if row.get("status") != "failed":
                continue
            try:
                prepared.append(prepare_retry(row["id"]))
            except HTTPException as error:
                warnings.append(row.get("source_name", row["id"]) + ": " + str(error.detail))
        retry_merge = snapshot.get("combined_status") == "failed" and any(
            row.get("status") == "ready" for row in snapshot["files"]
        )
        if not prepared and not retry_merge:
            raise HTTPException(
                status_code=422,
                detail="No recoverable failed files remain. Upload invalid or expired inputs again.",
            )
        combined = snapshot.get("combined_asset_id")
        previous = snapshot.get("previous_combined_asset_ids", [])
        if combined:
            previous = [*previous, combined]
        # Older independent batches have no collection asset. Build one when
        # resuming so their successful members join the recovered workspace.
        combined = str(uuid4())
        write_job(
            combined,
            status="queued",
            source_name="Combined learning context",
            source_type="collection",
            batch_id=identifier,
            source_file_ids=[row["id"] for row in snapshot["files"]],
            organization_mode="combined",
            files=[],
            stages=[],
        )
        job_id = start_job(
            identifier,
            "Upload batch",
            "batch",
            "batch_ingestion_resume",
            {"organization_mode": snapshot.get("organization_mode")},
            source_file_ids=[item.asset_id for item in prepared],
        )
        _write(
            identifier,
            status="queued",
            organization_mode="combined",
            combined_asset_id=combined,
            previous_combined_asset_ids=previous,
            warnings=warnings,
            job_id=job_id,
        )
        return PreparedBatch(identifier, tuple(prepared), job_id)


def process_batch(batch):
    from helper.job_results import complete_job, fail_job

    _write(batch.id, status="processing")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wizora-batch") as executor:
        list(executor.map(_process_file, batch.prepared))
    snapshot = read_batch(batch.id)
    successes = [row["id"] for row in snapshot["files"] if row["status"] == "ready"]
    failed = len(snapshot["files"]) - len(successes)
    warnings = []
    if snapshot["combined_asset_id"]:
        if successes:
            try:
                combine_assets(
                    successes, combined_asset_id=snapshot["combined_asset_id"], batch_id=batch.id
                )
            except Exception as exc:
                write_job(
                    snapshot["combined_asset_id"],
                    status="failed",
                    error={
                        "category": type(exc).__name__,
                        "detail": "Combined content could not be stored. Individual successful files remain available.",
                    },
                )
                warnings.append(
                    "Combined context creation failed; successful individual files remain available."
                )
        else:
            write_job(
                snapshot["combined_asset_id"],
                status="failed",
                error={"category": "empty_batch", "detail": "No files completed successfully."},
            )
    status = "failed" if not successes else "partial" if failed or warnings else "ready"
    _write(batch.id, status=status, warnings=warnings)
    result = read_batch(batch.id)
    if successes:
        complete_job(batch.job_id, {"batch": result}, warnings=warnings)
    else:
        fail_job(
            batch.job_id,
            {"category": "empty_batch", "detail": "No files completed successfully."},
            results={"batch": result},
        )
    return result


def _namespaced(file_id, reference):
    validate_asset_id(file_id)
    if (
        not isinstance(reference, str)
        or not reference
        or "\\" in reference
        or ":" in reference
        or PurePosixPath(reference).is_absolute()
        or any(part in {"", ".", ".."} for part in reference.split("/"))
    ):
        raise StorageError("Invalid original source reference")
    parts = PurePosixPath(reference).parts
    if parts[0] == "frames":
        return "frames/" + file_id + "_" + parts[-1]
    if parts[0] == "images":
        return "images/" + file_id + "/" + "/".join(parts[1:])
    return "sources/" + file_id + "/" + reference


def combine_assets(file_ids, *, combined_asset_id=None, batch_id=None):
    from helper.job_results import complete_job, fail_job, start_job

    identifiers = list(dict.fromkeys(file_ids))
    if not identifiers:
        raise EmptyContentError("No successful files are available to combine")
    combined_id = combined_asset_id or str(uuid4())
    if combined_id in identifiers:
        raise StorageError("A combined context cannot contain itself")
    destination = asset_directory(combined_id)
    if destination.exists():
        raise StorageError("Combined asset already exists")
    job_id = start_job(
        combined_id,
        "Combined learning context",
        "combined",
        "combined_content_merging",
        processing_config={"organization_mode": "combined"},
        source_file_ids=identifiers,
    )
    write_job(
        combined_id,
        status="processing",
        source_name="Combined learning context",
        source_type="collection",
        source_file_ids=identifiers,
        organization_mode="combined",
        batch_id=batch_id,
        stages=[],
    )
    records, seen, source_map, duplicates, size = [], {}, {}, 0, 0
    try:
        for file_id in identifiers:
            source_job = read_job(file_id)
            size += source_job.get("size_bytes", 0)
            for reference in source_job.get("files", []):
                mapped = _namespaced(file_id, reference)
                source_map[mapped] = {"source_file_id": file_id, "reference": reference}
            for document in load_asset_documents(file_id):
                original = document.metadata
                origin = {key: original[key] for key in _ORIGIN_KEYS if key in original}
                origin.update(source_file_id=file_id, origin_record_id=original["record_id"])
                metadata = {
                    **original,
                    "asset_id": combined_id,
                    "source_file_id": file_id,
                    "origin_record_id": original["record_id"],
                    "origin_locations": json.dumps([origin], ensure_ascii=False),
                }
                for key in ("artifact_reference", "frame_reference"):
                    if key in metadata:
                        metadata[key] = _namespaced(file_id, metadata[key])
                if "image_references" in metadata:
                    images = json.loads(metadata["image_references"])
                    for index, image in enumerate(images):
                        if isinstance(image, str):
                            images[index] = _namespaced(file_id, image)
                        elif image.get("reference"):
                            image["reference"] = _namespaced(file_id, image["reference"])
                    metadata["image_references"] = json.dumps(images, ensure_ascii=False)
                normalized = " ".join(unicodedata.normalize("NFKC", document.page_content).split())
                # Exact equivalence only: altered numbers, qualifiers and diagram
                # relationships remain separate evidence. Bound each origin group
                # so both generation and retrieval can still fit its provenance.
                content_type = original.get("content_type", "text")
                equivalence_type = (
                    "text"
                    if content_type in {"text", "page_text", "paragraph", "heading", "transcript"}
                    else content_type
                )
                key = (normalized, equivalence_type)
                previous = seen.get(key) if len(normalized) >= 40 else None
                if previous is not None:
                    locations = json.loads(records[previous].metadata["origin_locations"]) + [
                        origin
                    ]
                    encoded = json.dumps(locations, ensure_ascii=False)
                    if len(encoded) <= 4096:
                        records[previous].metadata["origin_locations"] = encoded
                        duplicates += 1
                        continue
                seen[key] = len(records)
                records.append(Document(page_content=document.page_content, metadata=metadata))
        from helper.semantic_dedup import semantic_deduplicate

        records, semantic = semantic_deduplicate(records, get_settings())
        duplicates += semantic["semantic_duplicate_count"]
        destination.mkdir(parents=True)
        version = hashlib.sha256(
            json.dumps(
                [{"text": record.page_content, "metadata": record.metadata} for record in records],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        persisted = write_source_records(
            records,
            destination / "documents.json",
            combined_id,
            "collection",
            "Combined learning context",
            version,
        )
        (destination / "extracted_text.txt").write_text(
            flatten_documents(persisted), encoding="utf-8"
        )
        write_job(
            combined_id,
            status="ready",
            source_map=source_map,
            files=sorted(source_map),
            size_bytes=size,
            record_count=len(persisted),
            duplicate_record_count=duplicates,
            index_status="deferred",
            semantic_deduplication=semantic,
            warnings=semantic["warnings"],
            processing_scope="combined",
            stages=[{"name": "combined_content_merging", "status": "completed", "at": _now()}],
        )
        complete_job(
            job_id,
            {
                "combined_asset_id": combined_id,
                "source_file_ids": identifiers,
                "duplicate_record_count": duplicates,
                "semantic_deduplication": semantic,
                "merged_content": [
                    {"page_content": record.page_content, "metadata": record.metadata}
                    for record in persisted
                ],
            },
            warnings=semantic["warnings"],
        )
        return combined_id
    except Exception as exc:
        write_job(
            combined_id,
            status="failed",
            error={"category": type(exc).__name__, "detail": "Combined content processing failed."},
        )
        fail_job(
            job_id,
            {"category": type(exc).__name__, "detail": "Combined content processing failed."},
        )
        raise
