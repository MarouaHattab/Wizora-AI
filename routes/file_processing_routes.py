import asyncio
import logging
import mimetypes
import uuid

import anyio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from helper import get_settings
from helper.asset_catalog import list_assets, public_job, read_job, source_file, write_job
from helper.asset_http import processing_http_error, read_asset_text
from helper.asset_paths import source_filename
from helper.ingestion import (
    FORMATS,
    PreparedUpload,
    cleanup_staging,
    prepare_upload,
    process_prepared,
    source_kind,
)
from helper.ingestion_errors import ProcessingError
from helper.job_results import fail_job, start_job
from helper.vlm_policy import VLMMode, frontend_vlm_capabilities

router = APIRouter()
logger = logging.getLogger(__name__)
_ingestion_limiter = anyio.CapacityLimiter(2)


def _process_upload(file, asset_id, source_name, source_type, suffix, vlm_mode="disabled"):
    return process_prepared(
        prepare_upload(file, asset_id, source_name, source_type, suffix, vlm_mode)
    )


async def _worker(function, *arguments):
    task = asyncio.create_task(
        anyio.to_thread.run_sync(function, *arguments, limiter=_ingestion_limiter)
    )
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
                write_job(
                    prepared.asset_id,
                    status="failed",
                    error={
                        "category": "interrupted",
                        "detail": "Upload was interrupted before processing.",
                    },
                )
        raise


async def _background(prepared):
    try:
        await _worker(process_prepared, prepared)
    except Exception as error:
        logger.warning("Background ingestion failed category=%s", type(error).__name__)


async def _upload(file, background=None, vlm_mode="disabled"):
    identifier = str(uuid.uuid4())
    try:
        try:
            name = source_filename(file.filename)
            kind, suffix = source_kind(name, file.content_type)
        except (HTTPException, ProcessingError) as error:
            public = error if isinstance(error, HTTPException) else processing_http_error(error)
            job_id = start_job(
                identifier,
                file.filename or "Unnamed upload",
                "unknown",
                "ingestion_validation",
                {"requested_vlm_mode": vlm_mode},
            )
            fail_job(
                job_id,
                {"category": "validation", "detail": public.detail},
                results={"valid": False},
            )
            write_job(
                identifier,
                status="failed",
                source_type="unknown",
                error={"category": "validation", "detail": public.detail},
            )
            public.headers = {
                **(public.headers or {}),
                "X-Wizora-File-ID": identifier,
                "X-Wizora-Job-ID": job_id,
            }
            raise public
        if background is None:
            return await _worker(
                _process_upload, file.file, identifier, name, kind, suffix, vlm_mode
            )
        prepared = await _worker(
            prepare_upload, file.file, identifier, name, kind, suffix, vlm_mode
        )
        background.add_task(_background, prepared)
        return public_job(read_job(identifier))
    except HTTPException as error:
        error.headers = {**(error.headers or {}), "X-Wizora-File-ID": identifier}
        raise
    except ProcessingError as error:
        logger.warning("Asset ingestion failed category=%s", type(error).__name__)
        public = processing_http_error(error)
        public.headers = {**(public.headers or {}), "X-Wizora-File-ID": identifier}
        raise public from None
    except Exception as error:
        logger.error("Asset ingestion failed category=%s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Asset processing failed",
            headers={"X-Wizora-File-ID": identifier},
        ) from None
    finally:
        with anyio.CancelScope(shield=True):
            await file.close()


@router.post("/upload_file/")
async def upload_file_api(file: UploadFile = File(...), vlm_mode: VLMMode = Form("disabled")):
    return await _upload(file, vlm_mode=vlm_mode)


@router.post("/uploads", status_code=202)
async def queue_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vlm_mode: VLMMode = Form("disabled"),
):
    return await _upload(file, background_tasks, vlm_mode=vlm_mode)


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
    capabilities = await anyio.to_thread.run_sync(frontend_vlm_capabilities, settings)
    return {
        "max_upload_bytes": settings.MAX_UPLOAD_BYTES,
        "extensions": sorted(FORMATS),
        "max_batch_files": getattr(settings, "MAX_BATCH_FILES", 32),
        "max_batch_bytes": getattr(settings, "MAX_BATCH_BYTES", settings.MAX_UPLOAD_BYTES * 4),
        "video_processing": "transcript_and_visuals"
        if capabilities["vlm_available"]
        else "transcript_only",
        "vision_provider": getattr(settings, "VISION_PROVIDER", "disabled"),
        "vision_model": getattr(settings, "VISION_MODEL", None),
        "max_media_duration_seconds": settings.VIDEO_MAX_DURATION_SECONDS,
        "index_on_upload": settings.INDEX_ON_UPLOAD,
        **capabilities,
    }


@router.get("/assets/{asset_id}")
async def asset_metadata(asset_id: str):
    return await upload_status(asset_id)


@router.get("/assets/{asset_id}/content")
async def asset_content(asset_id: str):
    """Inspect validated, persisted source text and its original source locators."""
    from helper.asset_storage import load_asset_documents

    try:
        job = await anyio.to_thread.run_sync(read_job, asset_id)
        if job.get("status") != "ready":
            raise HTTPException(status_code=409, detail="The source is still processing or failed")
        documents = await anyio.to_thread.run_sync(load_asset_documents, asset_id)
        return {
            "asset_id": asset_id,
            "processing_scope": job.get("processing_scope"),
            "warnings": job.get("warnings", []),
            "documents": [
                {"page_content": item.page_content, "metadata": item.metadata} for item in documents
            ],
        }
    except ProcessingError as error:
        raise processing_http_error(error) from None


@router.get("/assets/{asset_id}/files/{reference:path}")
async def asset_source(asset_id: str, reference: str):
    try:
        path, metadata = await anyio.to_thread.run_sync(source_file, asset_id, reference)
    except ProcessingError as error:
        raise processing_http_error(error) from None
    name = (
        metadata.get("source_name", path.name) if reference.startswith("original.") else path.name
    )
    return FileResponse(
        path,
        filename=name,
        media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
        content_disposition_type="inline",
    )


@router.get("/assets/{asset_id}/derived-frames/{evidence_id}")
async def derived_video_frame(asset_id: str, evidence_id: str):
    from helper.video_evidence import derived_frame_file

    try:
        path = await anyio.to_thread.run_sync(derived_frame_file, asset_id, evidence_id)
    except (ProcessingError, OSError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=404, detail="Derived video evidence is unavailable"
        ) from None
    return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline")


@router.get("/get_extracted_text/{asset_id}")
async def get_extracted_text(asset_id: str):
    return {"asset_id": asset_id, "extracted_text": await read_asset_text(asset_id)}
