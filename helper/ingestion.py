import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from helper import asset_paths, get_settings
from helper.asset_catalog import _status_io, record_stage, write_job
from helper.asset_paths import asset_directory
from helper.asset_storage import flatten_documents, write_source_records
from helper.extraction_jobs import ExtractionJobs
from helper.ingestion_errors import EmptyContentError, ProcessingError, StorageError
from helper.job_results import complete_job, fail_job, start_job
from helper.vlm_policy import normalize_vlm_mode, resolve_vlm_mode, vlm_unavailable_reason
from tools.media_processing import process_audio, process_video
from tools.text_extractor_tool import extract_text_documents

FORMATS = {
    ".pdf": ("pdf", {"application/pdf"}),
    ".txt": ("text", {"text/plain"}),
    ".docx": (
        "docx",
        {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        },
    ),
    ".mp3": ("audio", {"audio/mpeg", "audio/mp3"}),
    ".wav": ("audio", {"audio/wav", "audio/x-wav", "audio/wave"}),
    ".m4a": ("audio", {"audio/mp4", "audio/x-m4a", "video/mp4"}),
    ".flac": ("audio", {"audio/flac", "audio/x-flac"}),
    ".ogg": ("audio", {"audio/ogg", "application/ogg"}),
    ".aac": ("audio", {"audio/aac", "audio/x-aac"}),
    ".mp4": ("video", {"video/mp4"}),
    ".mov": ("video", {"video/quicktime"}),
    ".mkv": ("video", {"video/x-matroska", "video/matroska"}),
    ".webm": ("video", {"video/webm"}),
    ".avi": ("video", {"video/x-msvideo", "video/avi"}),
    ".flv": ("video", {"video/x-flv"}),
}


@dataclass(frozen=True)
class PreparedUpload:
    asset_id: str
    source_name: str
    source_type: str
    staging: Path
    original: Path
    version: str
    requested_vlm_mode: str = "disabled"
    vlm_mode: str = "disabled"


def source_kind(name, content_type):
    suffix = Path(name).suffix.lower()
    if suffix not in FORMATS:
        raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, TXT, audio, or video file")
    kind, types = FORMATS[suffix]
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared not in types | {"", "application/octet-stream", "binary/octet-stream"}:
        raise HTTPException(status_code=415, detail="The file extension and MIME type do not agree")
    return kind, suffix


def cleanup_staging(staging):
    root = (asset_paths.ASSETS_ROOT.resolve() / ".staging").resolve()
    target = Path(staging).resolve()
    if target.is_relative_to(root) and target != root and target.is_dir():
        shutil.rmtree(target)


def prepare_upload(file, asset_id, source_name, source_type, suffix, vlm_mode="disabled"):
    settings = get_settings()
    requested_mode = normalize_vlm_mode(vlm_mode)
    effective_mode = resolve_vlm_mode(requested_mode, settings, source_type)
    root = asset_paths.ASSETS_ROOT.resolve()
    staging_root = (root / ".staging").resolve()
    if not staging_root.is_relative_to(root):
        raise StorageError("Invalid temporary asset directory")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=asset_id + "-", dir=staging_root))
    original = staging / ("original" + suffix)
    validation_job = start_job(
        asset_id,
        source_name,
        source_type,
        "ingestion_validation",
        {"vlm_mode": effective_mode, "requested_vlm_mode": requested_mode},
    )
    size = 0
    try:
        digest, size = hashlib.sha256(), 0
        with original.open("wb") as output:
            while data := file.read(1024 * 1024):
                size += len(data)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="The upload exceeds the configured size limit"
                    )
                digest.update(data)
                output.write(data)
        if not size:
            raise EmptyContentError("Uploaded file is empty")
        with original.open("rb") as stream:
            header = stream.read(1024)
        if source_type == "pdf" and b"%PDF-" not in header:
            raise HTTPException(status_code=422, detail="The file content is not a PDF")
        if source_type == "docx" and not header.startswith(b"PK"):
            raise HTTPException(status_code=422, detail="The file content is not a DOCX archive")
        write_job(
            asset_id,
            source_name=source_name,
            source_type=source_type,
            size_bytes=size,
            status="queued",
            index_status="deferred",
            staging_name=staging.name,
            requested_vlm_mode=requested_mode,
            vlm_mode=effective_mode,
            stages=[{"name": "received", "status": "completed"}],
        )
        (staging / ".resume.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "asset_id": asset_id,
                    "source_name": source_name,
                    "source_type": source_type,
                    "original_name": original.name,
                    "version": digest.hexdigest(),
                    "requested_vlm_mode": requested_mode,
                    "vlm_mode": effective_mode,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_job(asset_id, resumable=True)
        complete_job(
            validation_job,
            {
                "valid": True,
                "size_bytes": size,
                "file_type": source_type,
                "content_sha256": digest.hexdigest(),
            },
        )
        return PreparedUpload(
            asset_id,
            source_name,
            source_type,
            staging,
            original,
            digest.hexdigest(),
            requested_mode,
            effective_mode,
        )
    except BaseException:
        fail_job(
            validation_job,
            {
                "category": "validation",
                "detail": "The uploaded file did not pass ingestion validation.",
            },
            results={"valid": False, "size_bytes": size, "file_type": source_type},
        )
        cleanup_staging(staging)
        raise


def prepare_retry(asset_id):
    from helper.asset_catalog import _lock, read_job
    from helper.asset_paths import validate_asset_id

    validate_asset_id(asset_id)
    with _lock:
        job = read_job(asset_id)
        if job.get("status") != "failed":
            raise HTTPException(
                status_code=409, detail="Only failed or interrupted files can be resumed."
            )
        root = (asset_paths.ASSETS_ROOT.resolve() / ".staging").resolve()
        directory = (root / str(job.get("staging_name", ""))).resolve()
        if directory == root or not directory.is_relative_to(root) or not directory.is_dir():
            raise HTTPException(
                status_code=422, detail="No recoverable input remains. Upload this file again."
            )
        try:
            manifest = json.loads((directory / ".resume.json").read_text(encoding="utf-8"))
            original = (directory / manifest["original_name"]).resolve()
            if (
                manifest.get("schema_version") != 1
                or manifest.get("asset_id") != asset_id
                or not original.is_relative_to(directory)
                or original.parent != directory
                or not original.is_file()
            ):
                raise ValueError("Invalid recovery input")
            digest = hashlib.sha256()
            with original.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != manifest["version"]:
                raise ValueError("Recovery input changed")
            prepared = PreparedUpload(
                asset_id,
                manifest["source_name"],
                manifest["source_type"],
                directory,
                original,
                manifest["version"],
                normalize_vlm_mode(manifest["requested_vlm_mode"]),
                normalize_vlm_mode(manifest["vlm_mode"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(
                status_code=422,
                detail="The recovery checkpoint is invalid. Upload this file again.",
            ) from None
        write_job(
            asset_id,
            status="queued",
            error=None,
            resumable=True,
            retry_count=int(job.get("retry_count", 0)) + 1,
        )
        return prepared


def recover_published_asset(asset_id):
    from helper.asset_catalog import read_job
    from helper.asset_storage import _validated_asset_documents

    directory = asset_directory(asset_id)
    if (
        not directory.is_dir()
        or not (directory / "documents.json").is_file()
        or not (directory / "extracted_text.txt").is_file()
    ):
        return False
    try:
        manifest = json.loads((directory / ".resume.json").read_text(encoding="utf-8"))
        if manifest.get("asset_id") != asset_id or manifest.get("schema_version") != 1:
            return False
        original = (directory / manifest["original_name"]).resolve()
        if original.parent != directory or not original.is_file():
            return False
        digest = hashlib.sha256()
        with original.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != manifest["version"]:
            return False
        documents = _validated_asset_documents(
            asset_id, (directory / "extracted_text.txt").read_text(encoding="utf-8")
        )
        if not documents:
            return False
        files = [original.name]
        for folder in ("images", "frames", "rendered"):
            for path in (directory / folder).rglob("*") if (directory / folder).is_dir() else ():
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(directory)
                ):
                    files.append(path.relative_to(directory).as_posix())
        previous = read_job(asset_id)
        warnings = [
            *previous.get("warnings", []),
            "Recovered a validated published extraction after interruption. Indexing is deferred.",
        ]
        write_job(
            asset_id,
            status="ready",
            resumable=False,
            error=None,
            index_status="deferred",
            files=sorted(files),
            record_count=len(documents),
            source_name=manifest["source_name"],
            source_type=manifest["source_type"],
            vlm_mode=manifest["vlm_mode"],
            requested_vlm_mode=manifest["requested_vlm_mode"],
            warnings=warnings,
        )
        identifier = start_job(
            asset_id,
            manifest["source_name"],
            manifest["source_type"],
            "ingestion_recovery",
            {
                "vlm_mode": manifest["vlm_mode"],
                "requested_vlm_mode": manifest["requested_vlm_mode"],
            },
        )
        complete_job(
            identifier,
            {"record_count": len(documents), "recovered_published_asset": True},
            warnings=warnings[-1:],
        )
        return True
    except (OSError, ValueError, KeyError, TypeError, ProcessingError):
        return False


def process_prepared(prepared):
    settings = get_settings()
    asset_id = prepared.asset_id
    operation_jobs = ExtractionJobs(prepared, settings)
    wall_start, cpu_start = time.monotonic(), time.process_time()
    processing_job = start_job(
        asset_id,
        prepared.source_name,
        prepared.source_type,
        "ingestion_processing",
        {"vlm_mode": prepared.vlm_mode, "requested_vlm_mode": prepared.requested_vlm_mode},
    )

    def metrics():
        values = {
            "wall_seconds": round(time.monotonic() - wall_start, 6),
            "process_cpu_seconds": round(time.process_time() - cpu_start, 6),
            "cpu_scope": "shared_process",
        }
        try:
            import psutil

            values["process_rss_bytes"] = psutil.Process().memory_info().rss
            values["memory_scope"] = "shared_process_snapshot"
        except (ImportError, OSError):
            from helper.resource_usage import process_memory

            values.update(process_memory())
        return values

    def progress(stage, details=None):
        operation_jobs.progress(stage, details)
        public_details = {
            key: item
            for key, item in (details or {}).items()
            if key not in {"records", "artifacts"}
        }
        for key, count_key in (("records", "record_count"), ("artifacts", "artifact_count")):
            if isinstance((details or {}).get(key), list):
                public_details[count_key] = len(details[key])
                public_details.setdefault("count", len(details[key]))
        record_stage(asset_id, stage, public_details)

    metadata = {}
    deadline = time.monotonic() + settings.MEDIA_TIMEOUT_SECONDS
    try:
        progress("processing")
        if prepared.source_type in {"pdf", "docx", "text"}:
            operation_jobs.begin("text_extraction")
        if prepared.source_type == "pdf":
            from tools.pdf_extractor_tool import extract_pdf_hybrid

            documents, metadata = extract_pdf_hybrid(
                str(prepared.original),
                asset_id=asset_id,
                source_name=prepared.source_name,
                settings=settings,
                vlm_mode=prepared.vlm_mode,
                deadline=deadline,
                progress=progress,
            )
            progress("pdf_extracted")
        elif prepared.source_type == "docx":
            from tools.docx_extractor_tool import extract_docx_hybrid

            documents, metadata = extract_docx_hybrid(
                str(prepared.original),
                asset_id=asset_id,
                source_name=prepared.source_name,
                settings=settings,
                vlm_mode=prepared.vlm_mode,
                deadline=deadline,
                progress=progress,
            )
            progress("docx_extracted")
        elif prepared.source_type == "text":
            documents = extract_text_documents(
                str(prepared.original), asset_id=asset_id, source_name=prepared.source_name
            )
            metadata.update(processing_scope="text_only", visual_status="disabled")
            progress("text_extracted")
        else:
            processor = process_video if prepared.source_type == "video" else process_audio
            mode_arguments = (
                {"vlm_mode": prepared.vlm_mode} if prepared.source_type == "video" else {}
            )
            result = processor(
                str(prepared.original),
                asset_id=asset_id,
                source_name=prepared.source_name,
                settings=settings,
                progress=progress,
                **mode_arguments,
            )
            documents, metadata = result.documents, result.metadata
        operation_jobs.outputs_ready()
        metadata.update(vlm_mode=prepared.vlm_mode, requested_vlm_mode=prepared.requested_vlm_mode)
        if (
            prepared.requested_vlm_mode != "disabled"
            and prepared.vlm_mode == "disabled"
            and prepared.source_type in {"pdf", "docx", "video"}
        ):
            metadata.setdefault("warnings", []).append(
                vlm_unavailable_reason(settings)
                or "Visual understanding was unavailable when this upload was submitted."
            )
        for document in documents:
            document.metadata.update(
                vlm_mode=prepared.vlm_mode, requested_vlm_mode=prepared.requested_vlm_mode
            )
            document.metadata.setdefault("extraction_method", "traditional")
            document.metadata.setdefault("processing_status", "completed")
        documents = write_source_records(
            documents,
            prepared.staging / "documents.json",
            asset_id,
            prepared.source_type,
            prepared.source_name,
            prepared.version,
        )
        if prepared.source_type == "video":
            from helper.video_evidence import build_timeline

            timeline = build_timeline(
                documents,
                float(metadata.get("duration_seconds", 0)),
                metadata.get("scene_transitions", []),
            )
            (prepared.staging / "timeline.json").write_text(
                json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            metadata.setdefault("files", []).append("timeline.json")
            metadata["timeline_reference"] = "timeline.json"
        operation_jobs.finish(documents, metadata)
        text = flatten_documents(documents)
        (prepared.staging / "extracted_text.txt").write_text(text, encoding="utf-8")
        progress("records_validated")
        destination = asset_directory(asset_id)
        if destination.exists():
            raise StorageError("Asset already exists")
        # A Windows scanner can briefly retain the closed directory handle.
        _status_io(lambda: prepared.staging.rename(destination))
        files = {prepared.original.name, *metadata.get("files", [])}
        summary = {
            key: metadata[key]
            for key in (
                "duration_seconds",
                "audio_status",
                "audio_reason",
                "processing_scope",
                "visual_status",
                "frame_count",
                "warnings",
                "vlm_mode",
                "requested_vlm_mode",
                "timeline_reference",
            )
            if key in metadata
        }
        summary.update(
            {key: value for key, value in metadata.items() if key.endswith("_visual_count")}
        )
        summary.update(
            record_count=len(documents),
            content_types=dict(
                Counter(doc.metadata.get("content_type", "text") for doc in documents)
            ),
            pages=sorted({doc.metadata["page"] for doc in documents if "page" in doc.metadata}),
            sections=list(
                dict.fromkeys(
                    doc.metadata["section"] for doc in documents if "section" in doc.metadata
                )
            ),
            languages=sorted(
                {doc.metadata["language"] for doc in documents if "language" in doc.metadata}
            ),
            files=sorted(files),
        )
        write_job(asset_id, **summary)
        if settings.INDEX_ON_UPLOAD:
            from helper.asset_index import ensure_asset_index

            progress("indexing")
            try:
                index = ensure_asset_index(asset_id, text, settings, documents=documents)
                summary.update(
                    index_status="ready",
                    index_version=index.index_version,
                    chunk_count=index.chunk_count,
                )
            except Exception:
                summary.update(index_status="deferred")
                summary.setdefault("warnings", []).append(
                    "Extraction succeeded; indexing failed and can be retried by a learning workflow."
                )
        else:
            summary["index_status"] = "deferred"
        progress("ready")
        summary["metrics"] = metrics()
        write_job(asset_id, status="ready", resumable=False, **summary)
        complete_job(
            processing_job,
            {"record_count": len(documents), "metrics": summary["metrics"]},
            warnings=summary.get("warnings"),
        )
        return {
            "message": "File processed successfully",
            "id": asset_id,
            "text_path": str(destination / "extracted_text.txt"),
            "original_file_path": str(destination / prepared.original.name),
            "status": "ready",
            **summary,
        }
    except Exception as exc:
        from helper.asset_http import processing_http_error

        public = exc if isinstance(exc, HTTPException) else processing_http_error(exc)
        operation_jobs.fail({"category": type(exc).__name__, "detail": public.detail})
        values = metrics()
        write_job(
            asset_id,
            status="failed",
            resumable=prepared.original.is_file(),
            metrics=values,
            error={"category": type(exc).__name__, "detail": public.detail},
        )
        fail_job(
            processing_job,
            {"category": type(exc).__name__, "detail": str(public.detail)},
            results={"metrics": values},
        )
        raise
