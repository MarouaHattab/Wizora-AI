import io
import json

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from helper.asset_http import processing_http_error
from helper.ingestion_errors import ProcessingError
from helper.job_results import export_job_results, get_job_result, list_job_results

router = APIRouter()


class JobExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_ids: list[str] = Field(min_length=1, max_length=200)


@router.get("/assets/{file_id}/jobs")
async def source_jobs(file_id: str):
    try:
        return {
            "file_id": file_id,
            "jobs": await anyio.to_thread.run_sync(list_job_results, file_id),
        }
    except ProcessingError as error:
        raise processing_http_error(error) from None


@router.get("/assets/{file_id}/jobs/{job_id}")
async def source_job_result(file_id: str, job_id: str):
    try:
        return await anyio.to_thread.run_sync(get_job_result, file_id, job_id)
    except ProcessingError as error:
        raise processing_http_error(error) from None


@router.get("/assets/{file_id}/jobs/{job_id}/download")
async def source_job_download(file_id: str, job_id: str):
    value = await source_job_result(file_id, job_id)
    return Response(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{value["job_id"]}.json"'},
    )


@router.post("/job-results/export")
async def export_source_jobs(request: JobExportRequest):
    try:
        payload = await anyio.to_thread.run_sync(export_job_results, request.file_ids)
    except ProcessingError as error:
        raise processing_http_error(error) from None
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Select valid files for the job export"
        ) from None
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="wizora-job-results.zip"'},
    )
