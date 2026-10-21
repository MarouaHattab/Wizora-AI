import hashlib
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from helper import asset_paths, get_settings
from helper.asset_catalog import record_stage, write_job
from helper.asset_paths import asset_directory
from helper.asset_storage import flatten_documents, write_source_records
from helper.ingestion_errors import EmptyContentError, ProcessingError, StorageError
from tools.docx_extractor_tool import extract_docx_documents
from tools.media_processing import process_audio, process_video
from tools.pdf_extractor_tool import extract_text_from_pdf
from tools.text_extractor_tool import extract_text_documents

FORMATS = {
    ".pdf": ("pdf", {"application/pdf"}),
    ".txt": ("text", {"text/plain"}),
    ".docx": ("docx", {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"}),
    ".mp3": ("audio", {"audio/mpeg", "audio/mp3"}), ".wav": ("audio", {"audio/wav", "audio/x-wav", "audio/wave"}),
    ".m4a": ("audio", {"audio/mp4", "audio/x-m4a", "video/mp4"}), ".flac": ("audio", {"audio/flac", "audio/x-flac"}),
    ".ogg": ("audio", {"audio/ogg", "application/ogg"}), ".aac": ("audio", {"audio/aac", "audio/x-aac"}),
    ".mp4": ("video", {"video/mp4"}), ".mov": ("video", {"video/quicktime"}),
    ".mkv": ("video", {"video/x-matroska", "video/matroska"}), ".webm": ("video", {"video/webm"}),
    ".avi": ("video", {"video/x-msvideo", "video/avi"}), ".flv": ("video", {"video/x-flv"}),
}


@dataclass(frozen=True)
class PreparedUpload:
    asset_id: str
    source_name: str
    source_type: str
    staging: Path
    original: Path
    version: str


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


def prepare_upload(file, asset_id, source_name, source_type, suffix):
    settings = get_settings()
    root = asset_paths.ASSETS_ROOT.resolve()
    staging_root = (root / ".staging").resolve()
    if not staging_root.is_relative_to(root):
        raise StorageError("Invalid temporary asset directory")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=asset_id + "-", dir=staging_root))
    original = staging / ("original" + suffix)
    try:
        digest, size = hashlib.sha256(), 0
        with original.open("wb") as output:
            while data := file.read(1024 * 1024):
                size += len(data)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The upload exceeds the configured size limit")
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
        write_job(asset_id, source_name=source_name, source_type=source_type, size_bytes=size,
                  status="queued", index_status="deferred", staging_name=staging.name,
                  stages=[{"name": "received", "status": "completed"}])
        return PreparedUpload(asset_id, source_name, source_type, staging, original, digest.hexdigest())
    except BaseException:
        cleanup_staging(staging)
        raise


def process_prepared(prepared):
    settings = get_settings()
    asset_id = prepared.asset_id
    progress = lambda stage, details=None: record_stage(asset_id, stage, details)
    metadata = {}
    try:
        progress("processing")
        if prepared.source_type == "pdf":
            documents = extract_text_from_pdf(str(prepared.original), asset_id=asset_id, source_name=prepared.source_name,
                ocr=settings.PDF_OCR_ENABLED, ocr_language=settings.PDF_OCR_LANGUAGE, max_ocr_pages=settings.PDF_MAX_OCR_PAGES)
            progress("pdf_extracted")
            pages_with_text = {document.metadata["page"] for document in documents}
            total_pages = documents[0].metadata.get("page_count", len(pages_with_text))
            if len(pages_with_text) < total_pages:
                metadata["warnings"] = [f"Only {len(pages_with_text)} of {total_pages} PDF pages contain extracted text. Blank or scanned pages were omitted; enable PDF_OCR_ENABLED for scanned pages."]
            if documents[0].metadata.get("scanned_page_count", 0) and not settings.PDF_OCR_ENABLED:
                metadata.setdefault("warnings", []).append("Scanned PDF pages were detected without OCR. Captions may be extracted while text inside page images remains unavailable.")
        elif prepared.source_type == "docx":
            documents = extract_docx_documents(str(prepared.original), asset_id=asset_id, source_name=prepared.source_name)
            progress("docx_extracted")
            images = prepared.staging / "images"
            metadata["files"] = [path.relative_to(prepared.staging).as_posix() for path in images.iterdir() if path.is_file()] if images.is_dir() else []
        elif prepared.source_type == "text":
            documents = extract_text_documents(str(prepared.original), asset_id=asset_id, source_name=prepared.source_name)
            progress("text_extracted")
        else:
            processor = process_video if prepared.source_type == "video" else process_audio
            result = processor(str(prepared.original), asset_id=asset_id, source_name=prepared.source_name,
                               settings=settings, progress=progress)
            documents, metadata = result.documents, result.metadata
        documents = write_source_records(documents, prepared.staging / "documents.json", asset_id,
                                         prepared.source_type, prepared.source_name, prepared.version)
        text = flatten_documents(documents)
        (prepared.staging / "extracted_text.txt").write_text(text, encoding="utf-8")
        progress("records_validated")
        destination = asset_directory(asset_id)
        if destination.exists():
            raise StorageError("Asset already exists")
        prepared.staging.rename(destination)
        files = {prepared.original.name, *metadata.get("files", [])}
        summary = {key: metadata[key] for key in ("duration_seconds", "audio_status", "audio_reason", "processing_scope",
                                                  "visual_status", "frame_count", "warnings") if key in metadata}
        summary.update(record_count=len(documents), content_types=dict(Counter(doc.metadata.get("content_type", "text") for doc in documents)),
            pages=sorted({doc.metadata["page"] for doc in documents if "page" in doc.metadata}),
            sections=list(dict.fromkeys(doc.metadata["section"] for doc in documents if "section" in doc.metadata)),
            languages=sorted({doc.metadata["language"] for doc in documents if "language" in doc.metadata}),
            files=sorted(files))
        write_job(asset_id, **summary)
        if settings.INDEX_ON_UPLOAD:
            from helper.asset_index import ensure_asset_index
            progress("indexing")
            index = ensure_asset_index(asset_id, text, settings, documents=documents)
            summary.update(index_status="ready", index_version=index.index_version, chunk_count=index.chunk_count)
        else:
            summary["index_status"] = "deferred"
        progress("ready")
        write_job(asset_id, status="ready", **summary)
        return {"message": "File processed successfully", "id": asset_id,
                "text_path": str(destination / "extracted_text.txt"), "original_file_path": str(destination / prepared.original.name),
                "status": "ready", **summary}
    except Exception as exc:
        from helper.asset_http import processing_http_error
        public = exc if isinstance(exc, HTTPException) else processing_http_error(exc)
        write_job(asset_id, status="failed", error={"category": type(exc).__name__, "detail": public.detail})
        raise
    finally:
        cleanup_staging(prepared.staging)
