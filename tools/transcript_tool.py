from collections.abc import Mapping, Sequence
import math
import time
from numbers import Real
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

import whisper
import numpy as np
from langchain_core.documents import Document

from helper.ingestion_errors import (
    AssetNotFoundError,
    EmptyContentError,
    ModelInitializationError,
    TranscriptionError,
)


_WHISPER_MODEL = "tiny"
_model = None
_model_config = None
_model_lock = Lock()
_inference_slots = BoundedSemaphore(1)


def _setting(settings, name, default=None):
    return settings.get(name, default) if isinstance(settings, Mapping) else getattr(settings, name, default)


def _get_model(settings=None) -> Any:
    global _model, _model_config
    name = _setting(settings, "WHISPER_MODEL", _WHISPER_MODEL)
    device = _setting(settings, "WHISPER_DEVICE", "cpu")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    configuration = (name, device)
    with _model_lock:
        if _model is None or _model_config != configuration:
            try:
                _model = whisper.load_model(name, device=device)
                _model_config = configuration
            except Exception as exc:
                raise ModelInitializationError(
                    "The transcription model could not be initialized."
                ) from exc
        return _model


def _transcribe(media_path: str, *, audio_samples: np.ndarray | None = None,
                settings=None, deadline: float | None = None) -> Mapping[str, Any]:
    path = Path(media_path)
    if not path.is_file():
        raise AssetNotFoundError("The media file was not found.")
    if not path.stat().st_size:
        raise EmptyContentError("The media file is empty.")
    if audio_samples is not None and (
        not isinstance(audio_samples, np.ndarray)
        or audio_samples.dtype != np.float32
        or audio_samples.ndim != 1
        or not 0 < audio_samples.size <= 16000 * 600
        or not np.isfinite(audio_samples).all()
    ):
        raise TranscriptionError("The normalized transcription audio is invalid.")
    model = _get_model(settings)
    acquired = _inference_slots.acquire(timeout=max(0, deadline - time.monotonic())) if deadline is not None else _inference_slots.acquire()
    if not acquired:
        raise TranscriptionError("The transcription deadline was exceeded while waiting for the model.")
    try:
        options = {"task": "transcribe", "temperature": 0.0, "condition_on_previous_text": False,
                   "no_speech_threshold": float(_setting(settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.6)),
                   "fp16": str(getattr(model, "device", "cpu")).startswith("cuda")}
        language = _setting(settings, "WHISPER_LANGUAGE")
        if language:
            options["language"] = language
        result = model.transcribe(audio_samples if audio_samples is not None else str(media_path), **options)
    except Exception as exc:
        raise TranscriptionError("The media could not be transcribed.") from exc
    finally:
        _inference_slots.release()
    if not isinstance(result, Mapping):
        raise TranscriptionError("The transcription result is invalid.")
    return result


def _documents_from_result(
    result: Mapping[str, Any],
    media_path: str,
    asset_id: str | None,
    source_name: str | None,
    source_type: str,
) -> list[Document]:
    full_text = result.get("text", "")
    if full_text is None:
        full_text = ""
    if not isinstance(full_text, str):
        raise TranscriptionError("The transcription text is invalid.")
    segments = result.get("segments")
    if segments is None:
        segments = []
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise TranscriptionError("The transcription segments are invalid.")

    metadata = {
        "source": Path(source_name or media_path).name,
        "source_type": source_type,
    }
    language = result.get("language")
    if isinstance(language, str) and language.strip():
        metadata["language"] = language.strip()[:64]
    if asset_id is not None:
        metadata["asset_id"] = asset_id

    documents = []
    segment_texts = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise TranscriptionError("A transcription segment is invalid.")
        text = segment.get("text")
        if not isinstance(text, str):
            raise TranscriptionError("A transcription segment has invalid text.")
        segment_texts.append(text)
        text = text.strip()
        if not text:
            continue
        start, end = segment.get("start"), segment.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Real)
            or not isinstance(end, Real)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start <= end
        ):
            raise TranscriptionError("A transcription segment has invalid timestamps.")
        segment_metadata = {
            **metadata,
            "record_index": len(documents),
            "start": float(start),
            "end": float(end),
            "timestamp_precision": "segment",
        }
        segment_id = segment.get("id")
        if isinstance(segment_id, (str, int)) and not isinstance(segment_id, bool):
            segment_metadata["segment_id"] = segment_id
        documents.append(Document(page_content=text, metadata=segment_metadata))

    if documents:
        segment_versions = {
            " ".join("".join(segment_texts).split()),
            " ".join(" ".join(segment_texts).split()),
        }
        if full_text.strip() and " ".join(full_text.split()) not in segment_versions:
            raise TranscriptionError(
                "The transcription text and segments are inconsistent."
            )
        return documents
    if full_text.strip():
        return [
            Document(
                page_content=full_text.strip(),
                metadata={**metadata, "record_index": 0},
            )
        ]
    raise EmptyContentError("The media contains no transcribed text.")


def transcribe_documents(
    media_path: str,
    asset_id: str | None = None,
    source_name: str | None = None,
    source_type: str = "audio",
    *,
    audio_samples: np.ndarray | None = None,
    settings=None,
    deadline: float | None = None,
) -> list[Document]:
    """Transcribe media into source records without discarding segment timing."""
    if source_type not in {"audio", "video"}:
        raise TranscriptionError("The media source type must be audio or video.")
    result = _transcribe(media_path, audio_samples=audio_samples, settings=settings, deadline=deadline)
    return _documents_from_result(result, media_path, asset_id, source_name, source_type)


def transcript_loader(audio_path: str) -> str:
    """Keep the legacy text interface while validating transcription failures."""
    result = _transcribe(audio_path)
    documents = _documents_from_result(result, audio_path, None, None, "audio")
    full_text = result.get("text")
    if isinstance(full_text, str) and full_text.strip():
        return full_text
    return "\n\n".join(document.page_content for document in documents)
