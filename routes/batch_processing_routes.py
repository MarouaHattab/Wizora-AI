import asyncio
import json
import logging
from functools import partial

import anyio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from helper.asset_catalog import write_job
from helper.asset_http import processing_http_error
from helper.batch_ingestion import (
    _write,
    prepare_batch,
    prepare_batch_resume,
    process_batch,
    read_batch,
)
from helper.ingestion import cleanup_staging
from helper.ingestion_errors import ProcessingError
from helper.vlm_policy import VLMMode

router = APIRouter()
logger = logging.getLogger(__name__)
_batch_limiter = anyio.CapacityLimiter(2)


async def _prepare(files, vlm_mode, configs):
    task = asyncio.create_task(
        anyio.to_thread.run_sync(
            partial(prepare_batch, files, vlm_mode=vlm_mode, file_configs=configs)
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Uploaded streams must remain open until the staging worker finishes.
        with anyio.CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        if not task.cancelled() and task.exception() is None:
            batch = task.result()
            for item in batch.prepared:
                cleanup_staging(item.staging)
                write_job(
                    item.asset_id,
                    status="failed",
                    error={
                        "category": "interrupted",
                        "detail": "Upload was interrupted before processing.",
                    },
                )
            _write(batch.id, status="failed", warnings=["The upload request was interrupted."])
            snapshot = read_batch(batch.id)
            if snapshot.get("combined_asset_id"):
                write_job(
                    snapshot["combined_asset_id"],
                    status="failed",
                    error={
                        "category": "interrupted",
                        "detail": "Upload was interrupted before processing.",
                    },
                )
            from helper.job_results import fail_job

            fail_job(
                batch.job_id,
                {"category": "interrupted", "detail": "The upload request was interrupted."},
            )
        raise


async def _background(batch):
    try:
        await anyio.to_thread.run_sync(process_batch, batch, limiter=_batch_limiter)
    except Exception as error:
        logger.warning("Batch ingestion failed category=%s", type(error).__name__)
        _write(
            batch.id,
            status="failed",
            warnings=[
                "Batch processing was interrupted. Completed individual files remain available."
            ],
        )
        from helper.job_results import fail_job

        fail_job(
            batch.job_id, {"category": type(error).__name__, "detail": "Batch processing failed."}
        )


@router.post("/batches", status_code=202)
async def queue_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    vlm_mode: VLMMode = Form("disabled"),
    file_configs: str | None = Form(None),
):
    try:
        try:
            configs = json.loads(file_configs) if file_configs else []
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="file_configs must be a JSON list of index/vlm_mode objects"
            ) from exc
        if not isinstance(configs, list):
            raise HTTPException(status_code=422, detail="file_configs must be a JSON list")
        batch = await _prepare(files, vlm_mode, configs)
        initial = await anyio.to_thread.run_sync(read_batch, batch.id)
        background_tasks.add_task(_background, batch)
        return initial
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except ProcessingError as error:
        raise processing_http_error(error) from None
    except Exception as error:
        logger.error("Batch upload failed category=%s", type(error).__name__)
        raise HTTPException(status_code=500, detail="Batch upload failed") from None
    finally:
        with anyio.CancelScope(shield=True):
            for upload in files:
                await upload.close()


@router.get("/batches/{batch_id}")
async def batch_status(batch_id: str):
    try:
        return await anyio.to_thread.run_sync(read_batch, batch_id)
    except ProcessingError as error:
        raise processing_http_error(error) from None


@router.post("/batches/{batch_id}/resume", status_code=202)
async def resume_failed_batch(batch_id: str, background_tasks: BackgroundTasks):
    try:
        batch = await anyio.to_thread.run_sync(prepare_batch_resume, batch_id)
        initial = await anyio.to_thread.run_sync(read_batch, batch.id)
        background_tasks.add_task(_background, batch)
        return initial
    except ProcessingError as error:
        raise processing_http_error(error) from None
