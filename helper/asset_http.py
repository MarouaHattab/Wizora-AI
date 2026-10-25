import anyio
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from .asset_storage import load_asset_text
from .ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    InvalidAssetError,
    InvalidDocumentError,
    ModelInitializationError,
    ProcessingError,
    StorageError,
    TranscriptionError,
)


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SingleAssetRequest(WorkflowRequest):
    asset_id: str


class ReviewFeedbackRequest(WorkflowRequest):
    thread_id: str
    feedback: str
    review_id: str | None = None


def processing_http_error(error):
    if isinstance(error, InvalidAssetError):
        return HTTPException(status_code=400, detail="Invalid asset identifier or filename")
    if isinstance(error, AssetNotFoundError):
        return HTTPException(status_code=404, detail="Asset not found")
    if isinstance(error, EmptyContentError):
        return HTTPException(
            status_code=422, detail=str(error) or "The source contains no extractable text"
        )
    if isinstance(error, InvalidDocumentError):
        return HTTPException(
            status_code=422,
            detail=str(error) or "The source could not be processed as a valid document",
        )
    if isinstance(error, ModelInitializationError):
        return HTTPException(
            status_code=503, detail=str(error) or "The processing model is unavailable"
        )
    if isinstance(error, TranscriptionError):
        return HTTPException(status_code=500, detail=str(error) or "Media transcription failed")
    if isinstance(error, StorageError):
        return HTTPException(status_code=500, detail="Asset storage is unavailable")
    if type(error) is ProcessingError:
        return HTTPException(status_code=500, detail=str(error) or "Asset processing failed")
    return HTTPException(status_code=500, detail="Asset processing failed")


async def read_asset_text(asset_id, *, for_question=False):
    try:
        return await anyio.to_thread.run_sync(load_asset_text, asset_id)
    except ProcessingError as exc:
        if for_question and isinstance(exc, EmptyContentError):
            raise HTTPException(
                status_code=422,
                detail={
                    "status": "insufficient_source",
                    "reason": "The source contains no educational material.",
                },
            ) from None
        raise processing_http_error(exc) from None


async def validate_session_asset(state):
    if not state:
        return
    if not state.get("asset_id") or not isinstance(state.get("context"), str):
        raise HTTPException(
            status_code=409, detail="Session has no asset context. Start a new session."
        )
    context = await read_asset_text(state["asset_id"])
    if context != state["context"]:
        raise HTTPException(
            status_code=409,
            detail="Session content does not match the selected asset. Start a new session.",
        )
