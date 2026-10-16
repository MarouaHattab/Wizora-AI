import asyncio
import logging
import mimetypes
import uuid

import anyio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from helper import get_settings
from helper.asset_catalog import list_assets, public_job, read_job, source_file, write_job
from helper.asset_http import processing_http_error, read_asset_text
from helper.asset_paths import source_filename
from helper.ingestion import FORMATS, PreparedUpload, cleanup_staging, prepare_upload, process_prepared, source_kind
from helper.ingestion_errors import ProcessingError

router = APIRouter()
logger = logging.getLogger(__name__)
_ingestion_limiter = anyio.CapacityLimiter(2)


def _source_kind(name, content_type):
    return source_kind(name, content_type)


def _process_upload(file, asset_id, source_name, source_type, suffix):
    return process_prepared(prepare_upload(file, asset_id, source_name, source_type, suffix))


async def _worker(function, *arguments):
    task = asyncio.create_task(anyio.to_thread.run_sync(function, *arguments, limiter=_ingestion_limiter))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # The spooled upload must remain open until its worker has released it.
        with anyio.CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        if not task.cancelled():
            error = task.exception()
            if error is None and isinstance(task.result(), PreparedUpload):
                prepared = task.result()
                cleanup_staging(prepared.staging)
                write_job(prepared.asset_id, status="failed", error={"category": "interrupted", "detail": "Upload was interrupted before processing."})
        raise


async def _background(prepared):
    try:
        await _worker(process_prepared, prepared)
    except Exception as error:
        logger.warning("Background ingestion failed category=%s", type(error).__name__)


async def _upload(file, background=None):
    try:
        name = source_filename(file.filename)
        kind, suffix = source_kind(name, file.content_type)
        identifier = str(uuid.uuid4())
        if background is None:
            return await _worker(_process_upload, file.file, identifier, name, kind, suffix)
        prepared = await _worker(prepare_upload, file.file, identifier, name, kind, suffix)
        background.add_task(_background, prepared)
        return public_job(read_job(identifier))
    except HTTPException:
        raise
    except ProcessingError as error:
        logger.warning("Asset ingestion failed category=%s", type(error).__name__)
        raise processing_http_error(error) from None
    except Exception as error:
        logger.error("Asset ingestion failed category=%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="Asset processing failed") from None
    finally:
        with anyio.CancelScope(shield=True):
            await file.close()


@router.post("/upload_file/")
async def upload_file_api(file: UploadFile = File(...)):
    return await _upload(file)


@router.post("/uploads", status_code=202)
async def queue_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    return await _upload(file, background_tasks)


@router.get("/uploads/{asset_id}")
async def upload_status(asset_id: str):
    try:
        return public_job(await anyio.to_thread.run_sync(read_job, asset_id))
    except ProcessingError as error:
        raise processing_http_error(error) from None


@router.get("/assets")
async def asset_library():
    return {"assets": await anyio.to_thread.run_sync(list_assets)}


@router.get("/assets/config")
async def upload_configuration():
    settings = get_settings()
    return {"max_upload_bytes": settings.MAX_UPLOAD_BYTES, "extensions": sorted(FORMATS),
            "video_processing": "transcript_only",
            "index_on_upload": settings.INDEX_ON_UPLOAD}


@router.get("/assets/{asset_id}")
async def asset_metadata(asset_id: str):
    return await upload_status(asset_id)


@router.get("/assets/{asset_id}/files/{reference:path}")
async def asset_source(asset_id: str, reference: str):
    try:
        path, metadata = await anyio.to_thread.run_sync(source_file, asset_id, reference)
    except ProcessingError as error:
        raise processing_http_error(error) from None
    name = metadata.get("source_name", path.name) if reference.startswith("original.") else path.name
    return FileResponse(path, filename=name, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        content_disposition_type="inline")


@router.get("/get_extracted_text/{asset_id}")
async def get_extracted_text(asset_id: str):
    return {"asset_id": asset_id, "extracted_text": await read_asset_text(asset_id)}
