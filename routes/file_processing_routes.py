import asyncio
import hashlib
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import anyio
from fastapi import APIRouter, File, UploadFile, HTTPException

from helper import asset_paths
from helper.asset_http import processing_http_error, read_asset_text
from helper.asset_paths import asset_directory, source_filename
from helper.asset_storage import flatten_documents, write_source_records
from helper.ingestion_errors import EmptyContentError, ProcessingError, StorageError
from tools.pdf_extractor_tool import extract_text_from_pdf
from tools.transcript_tool import transcribe_documents

router = APIRouter()
logger = logging.getLogger(__name__)
_ingestion_limiter = anyio.CapacityLimiter(2)
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}


def _source_kind(name, content_type):
    suffix = Path(name).suffix.lower()
    content_type = (content_type or "").split(";", 1)[0].lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        return "pdf", ".pdf"
    if suffix in _VIDEO_EXTENSIONS or content_type.startswith("video/"):
        return "video", suffix if suffix in _VIDEO_EXTENSIONS else ".media"
    if suffix in _AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        return "audio", suffix if suffix in _AUDIO_EXTENSIONS else ".media"
    raise HTTPException(status_code=415, detail="Only PDF, audio, or video files are supported")


def _process_upload(file, asset_id, source_name, source_type, suffix):
    root = asset_paths.ASSETS_ROOT.resolve()
    staging_root = (root / ".staging").resolve()
    if not staging_root.is_relative_to(root) or staging_root == root:
        raise StorageError("Invalid temporary asset directory")
    staging = None
    try:
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=asset_id + "-", dir=staging_root))
        if not staging.resolve().is_relative_to(staging_root):
            raise StorageError("Invalid temporary asset path")
        original = staging / ("original" + suffix)
        digest = hashlib.sha256()
        size = 0
        with original.open("wb") as output:
            while data := file.read(1024 * 1024):
                digest.update(data)
                size += len(data)
                output.write(data)
        if not size:
            raise EmptyContentError("Uploaded file is empty")
        if source_type == "pdf":
            documents = extract_text_from_pdf(str(original), asset_id=asset_id, source_name=source_name)
        else:
            documents = transcribe_documents(
                str(original), asset_id=asset_id, source_name=source_name, source_type=source_type,
            )
        documents = write_source_records(
            documents, staging / "documents.json", asset_id, source_type, source_name, digest.hexdigest(),
        )
        (staging / "extracted_text.txt").write_text(flatten_documents(documents), encoding="utf-8")
        destination = asset_directory(asset_id)
        if destination.exists():
            raise StorageError("Asset already exists")
        staging.rename(destination)
        staging = None
        return {
            "message": "File processed successfully", "id": asset_id,
            "text_path": str(destination / "extracted_text.txt"),
            "original_file_path": str(destination / original.name),
        }
    except OSError as exc:
        raise StorageError("Unable to store processed asset") from exc
    finally:
        if staging is not None and staging.exists():
            resolved = staging.resolve()
            if resolved.is_relative_to(staging_root) and resolved != staging_root:
                shutil.rmtree(resolved)


@router.post("/upload_file/")
async def upload_file_api(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        source_name = source_filename(file.filename)
        source_type, suffix = _source_kind(source_name, file.content_type)
        task = asyncio.create_task(anyio.to_thread.run_sync(
            _process_upload, file.file, str(uuid.uuid4()), source_name, source_type, suffix,
            limiter=_ingestion_limiter,
        ))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep the upload open until the worker has released it.
            with anyio.CancelScope(shield=True):
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            if not task.cancelled():
                task.exception()
            raise
    except HTTPException:
        raise
    except ProcessingError as exc:
        logger.warning("Asset ingestion failed: %s", type(exc).__name__)
        raise processing_http_error(exc) from None
    except Exception as exc:
        logger.error("Unexpected asset ingestion failure: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Asset processing failed") from None
    finally:
        if file:
            with anyio.CancelScope(shield=True):
                await file.close()


@router.get("/get_extracted_text/{asset_id}")
async def get_extracted_text(asset_id: str):
    text = await read_asset_text(asset_id)
    return {"asset_id": asset_id, "extracted_text": text}
