import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from langchain_core.documents import Document

from helper.ingestion_errors import (
    AssetNotFoundError, EmptyContentError, InvalidDocumentError, ModelInitializationError, ProcessingError,
)
from tools.transcript_tool import transcribe_documents


@dataclass(frozen=True)
class ProcessingResult:
    documents: list[Document]
    metadata: dict


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    has_audio: bool
    has_video: bool
    width: int = 0
    height: int = 0


_slots = BoundedSemaphore(2)
_MAX_PROBE_BYTES = 1024 * 1024


def _setting(settings, name, default=None):
    return settings.get(name, default) if isinstance(settings, Mapping) else getattr(settings, name, default)


def _binary(settings, name, *, required=False):
    configured = _setting(settings, name.upper() + "_BINARY")
    candidate = configured.strip() if isinstance(configured, str) and configured.strip() else shutil.which(name)
    if candidate:
        resolved = shutil.which(candidate) or (str(Path(candidate).resolve()) if Path(candidate).is_file() else None)
        if resolved:
            return resolved
        if configured:
            raise ProcessingError(f"The configured {name} executable is unavailable.")
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError) as exc:
            if required:
                raise ProcessingError("FFmpeg is required for media processing.") from exc
    return None


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProcessingError("The media processing deadline was exceeded.")
    return remaining


def _run(arguments, deadline, *, expect_success=True):
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(arguments, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     timeout=_remaining(deadline), check=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise ProcessingError("The media processing deadline was exceeded.") from exc
        except OSError as exc:
            raise ProcessingError("The media processor could not be started.") from exc
        stdout.seek(0, 2)
        stderr.seek(0, 2)
        if stdout.tell() > _MAX_PROBE_BYTES or stderr.tell() > _MAX_PROBE_BYTES:
            raise InvalidDocumentError("The media metadata exceeded the processing limit.")
        stdout.seek(0)
        stderr.seek(0)
        output, diagnostic = stdout.read(), stderr.read(_MAX_PROBE_BYTES)
        if expect_success and process.returncode:
            raise InvalidDocumentError("The media could not be decoded.")
        return output, diagnostic, process.returncode


def inspect_media(path, *, settings, deadline=None) -> MediaInfo:
    source = Path(path).resolve()
    if not source.is_file():
        raise AssetNotFoundError("The media file was not found.")
    if not source.stat().st_size:
        raise EmptyContentError("The media file is empty.")
    if deadline is None:
        deadline = time.monotonic() + float(_setting(settings, "MEDIA_TIMEOUT_SECONDS", 180))
    probe = _binary(settings, "ffprobe")
    if probe:
        raw, _, _ = _run([probe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries", "format=duration:stream=codec_type,width,height",
                          "-of", "json", str(source)], deadline)
        try:
            metadata = json.loads(raw)
            duration = float(metadata["format"]["duration"])
            streams = metadata.get("streams", [])
            audio = any(stream.get("codec_type") == "audio" for stream in streams)
            videos = [stream for stream in streams if stream.get("codec_type") == "video"]
            width, height = (int(videos[0].get("width", 0)), int(videos[0].get("height", 0))) if videos else (0, 0)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise InvalidDocumentError("The media metadata is invalid.") from exc
    else:
        ffmpeg = _binary(settings, "ffmpeg", required=True)
        _, diagnostic, _ = _run([ffmpeg, "-hide_banner", "-nostdin", "-protocol_whitelist", "file,pipe", "-i", str(source)], deadline, expect_success=False)
        text = diagnostic.decode("utf-8", errors="replace")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if not match:
            raise InvalidDocumentError("The media has no readable finite duration.")
        duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
        audio = bool(re.search(r"Stream #\S+.*?: Audio:", text))
        video = re.search(r"Stream #\S+.*?: Video:.*?(\d{2,6})x(\d{2,6})", text)
        videos = [video] if video else []
        width, height = (int(video[1]), int(video[2])) if video else (0, 0)
    limit = min(float(_setting(settings, "MAX_MEDIA_DURATION_SECONDS", 7200)),
                float(_setting(settings, "VIDEO_MAX_DURATION_SECONDS", 7200)))
    if not math.isfinite(duration) or not 0 < duration <= limit:
        raise InvalidDocumentError("The media duration exceeds the configured limit or is invalid.")
    if not audio and not videos:
        raise InvalidDocumentError("The file contains no supported audio or video streams.")
    if videos and (width <= 0 or height <= 0 or width * height > 40_000_000 or max(width, height) > 20000):
        raise InvalidDocumentError("The video dimensions exceed the processing limit.")
    return MediaInfo(duration, audio, bool(videos), width, height)


def _progress(callback, stage, **details):
    if callable(callback):
        callback(stage, details)


def _transcript(source, info, *, asset_id, source_name, settings, deadline, progress, source_type):
    documents = []
    if not info.has_audio:
        return documents, "unavailable", "The video has no audio track."
    ffmpeg = _binary(settings, "ffmpeg", required=True)
    chunk_seconds = min(600, max(1, float(_setting(settings, "AUDIO_CHUNK_SECONDS", 300))))
    chunk_count = math.ceil(info.duration_seconds / chunk_seconds)
    with tempfile.TemporaryDirectory(prefix=".media-", dir=source.parent) as temporary:
        audio_path = Path(temporary) / "audio.wav"
        for chunk_index in range(chunk_count):
            start = chunk_index * chunk_seconds
            length = min(chunk_seconds, info.duration_seconds - start)
            _progress(progress, "extracting_audio", segment=chunk_index + 1, segments=chunk_count)
            try:
                _run([ffmpeg, "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-ss", str(start), "-i", str(source),
                      "-t", str(length), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                      "-af", "aresample=16000:async=1:first_pts=0", "-c:a", "pcm_s16le", "-threads", "1", str(audio_path)], deadline)
            except ProcessingError:
                _remaining(deadline)
                return documents, "partial" if documents else "failed", "The audio track could not be decoded."
            _remaining(deadline)
            try:
                with wave.open(str(audio_path), "rb") as audio:
                    frame_count = audio.getnframes()
                    if (audio.getnchannels() != 1 or audio.getframerate() != 16000
                            or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE"
                            or frame_count > 16000 * 600):
                        raise ProcessingError("The extracted audio format is invalid.")
                    samples = np.frombuffer(audio.readframes(frame_count), dtype="<i2").astype(np.float32) / 32768.0
            except (OSError, EOFError, wave.Error) as exc:
                raise ProcessingError("The extracted audio could not be read.") from exc
            if not frame_count:
                # Container duration may include encoder padding beyond decoded samples.
                audio_path.unlink(missing_ok=True)
                continue
            _progress(progress, "transcribing", segment=chunk_index + 1, segments=chunk_count)
            try:
                segments = transcribe_documents(str(audio_path), asset_id, source_name, source_type,
                                                audio_samples=samples, settings=settings, deadline=deadline)
            except ModelInitializationError:
                raise
            except EmptyContentError:
                continue
            except ProcessingError:
                return documents, "partial" if documents else "failed", "Audio transcription could not be completed."
            finally:
                audio_path.unlink(missing_ok=True)
            _remaining(deadline)
            for segment_index, document in enumerate(segments):
                metadata = dict(document.metadata)
                if "start" in metadata:
                    metadata["start"] = min(info.duration_seconds, start + float(metadata["start"]))
                    metadata["end"] = min(info.duration_seconds, start + float(metadata["end"]))
                else:
                    metadata.update(start=start, end=start + length, timestamp_precision="audio_window")
                metadata.update(modality=source_type, content_type="transcript", record_index=len(documents),
                                segment_id=f"audio-{chunk_index:04d}-{segment_index:04d}")
                documents.append(Document(page_content=document.page_content, metadata=metadata))
            _progress(progress, "transcribing", segment=chunk_index + 1, segments=chunk_count, status="completed")
    return documents, "completed" if documents else "empty", "" if documents else "No spoken content was detected."


def _process(path, *, asset_id, source_name, settings, progress, video):
    source = Path(path).resolve()
    deadline = time.monotonic() + float(_setting(settings, "MEDIA_TIMEOUT_SECONDS", 180))
    if not _slots.acquire(timeout=_remaining(deadline)):
        raise ProcessingError("Media processing is busy; the processing deadline was reached.")
    try:
        _progress(progress, "inspecting_media")
        info = inspect_media(source, settings=settings, deadline=deadline)
        if video and not info.has_video:
            raise InvalidDocumentError("The uploaded video contains no video track.")
        vision_enabled = video and _setting(settings, "VISION_PROVIDER", "disabled") != "disabled"
        if video and not info.has_audio and not vision_enabled:
            raise EmptyContentError("The video has no audio track. Configure video vision to extract visible content.")
        if not video and not info.has_audio:
            raise InvalidDocumentError("The uploaded audio contains no audio track.")
        source_type = "video" if video else "audio"
        source_name = Path(source_name or source.name).name
        documents, audio_status, audio_reason = _transcript(source, info, asset_id=asset_id, source_name=source_name,
            settings=settings, deadline=deadline, progress=progress, source_type=source_type)
        if audio_status in {"failed", "partial"}:
            raise ProcessingError(audio_reason)
        warnings = []
        files = []
        frame_count = 0
        if vision_enabled:
            from tools.video_vision import analyze_video_frames
            visual_documents, files = analyze_video_frames(source, info, asset_id=asset_id, source_name=source_name,
                settings=settings, deadline=deadline, progress=progress)
            frame_count = len(visual_documents)
            documents.extend(visual_documents)
            documents.sort(key=lambda document: (document.metadata.get("start", 0),
                                                  document.metadata.get("content_type") == "visual_description"))
            warnings.append("Video visuals use sampled frames; content between sampled timestamps may be omitted.")
            if audio_status in {"unavailable", "empty"}:
                warnings.append(audio_reason)
        elif video:
            warnings.append("Video vision is disabled. Only spoken audio was extracted; slides, diagrams and demonstrations were not analyzed.")
        for number, document in enumerate(documents):
            document.metadata["record_index"] = number
        has_transcript = any(document.metadata.get("content_type") == "transcript" for document in documents)
        scope = "transcript_and_visuals" if frame_count and has_transcript else "visual_only" if frame_count else "transcript_only"
        metadata = {"duration_seconds": info.duration_seconds, "audio_status": audio_status,
                    "audio_reason": audio_reason, "processing_scope": scope, "files": files,
                    "visual_status": "completed" if frame_count else "disabled", "frame_count": frame_count,
                    "warnings": warnings}
        _remaining(deadline)
        if not documents:
            raise EmptyContentError("The media contains no transcribed speech or analyzed visual content.")
        metadata["record_count"] = len(documents)
        return ProcessingResult(documents, metadata)
    finally:
        _slots.release()


def process_audio(path, *, asset_id, source_name, settings, progress=None) -> ProcessingResult:
    return _process(path, asset_id=asset_id, source_name=source_name, settings=settings, progress=progress, video=False)


def process_video(path, *, asset_id, source_name, settings, progress=None) -> ProcessingResult:
    """Extract speech and optionally interpret bounded, timestamped visual frames."""
    return _process(path, asset_id=asset_id, source_name=source_name, settings=settings, progress=progress, video=True)
