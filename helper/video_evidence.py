import hashlib
import json
import math
import re
import time
from pathlib import Path
from threading import RLock
from uuid import uuid4
from weakref import WeakValueDictionary

from langchain_core.documents import Document

from helper.ingestion_errors import InvalidDocumentError, ProcessingError

_LOCK = RLock()
_KEY_LOCKS = WeakValueDictionary()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _cache_directory(key):
    from helper import asset_paths

    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise InvalidDocumentError("Invalid derived evidence reference")
    raw_root = asset_paths.ASSETS_ROOT.resolve() / ".video-evidence"
    if raw_root.is_symlink() or (raw_root / key).is_symlink():
        raise InvalidDocumentError("Derived evidence cannot use linked cache directories")
    root = raw_root.resolve()
    target = (root / key).resolve()
    if (
        not root.is_relative_to(asset_paths.ASSETS_ROOT.resolve())
        or not target.is_relative_to(root)
        or target == root
    ):
        raise InvalidDocumentError("Invalid derived evidence path")
    return target


def _load_cache(key, asset_id, index_version=None):
    folder = _cache_directory(key)
    if (folder / "evidence.json").stat().st_size > 128000:
        raise InvalidDocumentError("Derived evidence exceeds the metadata limit")
    payload = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
    identity = payload["identity"]
    document = Document.model_validate(payload["document"])
    metadata = document.metadata
    frame = folder / "frames" / ("derived_" + key + ".jpg")
    if not frame.resolve().is_relative_to(folder.resolve()):
        raise InvalidDocumentError("Derived frame is outside its cache")
    if (
        payload.get("schema_version") != 1
        or _digest(identity) != key
        or identity.get("asset_id") != asset_id
        or index_version is not None
        and identity.get("index_version") != index_version
        or metadata.get("asset_id") != asset_id
        or metadata.get("index_version") != identity["index_version"]
        or metadata.get("derived_evidence_id") != key
        or document.id != key
        or metadata.get("chunk_id") != key
        or metadata.get("content_hash")
        != hashlib.sha256(document.page_content.encode()).hexdigest()
        or payload.get("document_hash") != _digest(payload["document"])
        or metadata.get("derived_source_hash") != identity.get("source_hash")
        or metadata.get("asset_version") != identity.get("asset_version")
        or metadata.get("start") != identity.get("timestamp")
        or metadata.get("end") != identity.get("timestamp")
        or metadata.get("vision_provider") != identity.get("provider")
        or metadata.get("vision_model") != identity.get("model")
        or metadata.get("frame_reference") != "frames/derived_" + key + ".jpg"
        or metadata.get("derived_reference") != metadata.get("frame_reference")
        or metadata.get("vlm_mode") != identity.get("mode")
        or metadata.get("timestamp_precision") != "frame"
        or metadata.get("content_type") != "visual_description"
        or metadata.get("source_type") != "video"
        or not frame.is_file()
        or _file_hash(frame) != payload.get("frame_hash")
    ):
        raise InvalidDocumentError("Derived evidence does not match its source or request")
    return document, frame


def _validate_current_source(document, asset_id):
    from helper.asset_catalog import read_job, source_file
    from helper.asset_paths import validate_asset_id

    validate_asset_id(asset_id)
    job = read_job(asset_id)
    if job.get("status") != "ready":
        raise InvalidDocumentError("The source is not ready")
    if job.get("vlm_mode") not in {"automatic", "enhanced"}:
        raise InvalidDocumentError("Visual processing is disabled")
    original = next(
        (value for value in job.get("files", []) if value.startswith("original.")), None
    )
    source, _ = source_file(asset_id, original)
    if (
        _file_hash(source) != document.metadata["derived_source_hash"]
        or document.metadata["asset_version"] != document.metadata["derived_source_hash"]
    ):
        raise InvalidDocumentError("The original video changed after analysis")


def validate_derived_document(document, asset_id, index_version):
    try:
        stored, _ = _load_cache(
            document.metadata.get("derived_evidence_id"), asset_id, index_version
        )
        if stored.model_dump() != document.model_dump():
            raise InvalidDocumentError("Derived evidence was modified")
        _validate_current_source(document, asset_id)
    except (ProcessingError, OSError, ValueError, KeyError, TypeError) as error:
        raise InvalidDocumentError(
            "Derived evidence is unavailable or does not match its source"
        ) from error


def derived_frame_file(asset_id, key):
    document, frame = _load_cache(key, asset_id)
    _validate_current_source(document, asset_id)
    return frame


def _extract_target_frame(source, destination, *, timestamp, settings, deadline):
    from tools.media_processing import _binary, _remaining, _run, _setting, _slots

    if not _slots.acquire(timeout=min(10, _remaining(deadline))):
        raise ProcessingError("Media processing is busy")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        maximum = min(2048, max(128, int(_setting(settings, "VIDEO_FRAME_MAX_DIMENSION", 960))))
        _run(
            [
                _binary(settings, "ffmpeg", required=True),
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-protocol_whitelist",
                "file,pipe",
                "-ss",
                str(timestamp),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-vf",
                f"scale=w='min({maximum},iw)':h='min({maximum},ih)':force_original_aspect_ratio=decrease",
                "-threads",
                "1",
                "-q:v",
                "3",
                "-update",
                "1",
                str(destination),
            ],
            deadline,
        )
    finally:
        _slots.release()


def _prune_cache(current):
    import shutil

    root = current.parent.resolve()
    entries = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and re.fullmatch(r"[0-9a-f]{64}", path.name)
        and (path / "evidence.json").is_file()
    ]
    for entry in sorted(entries, key=lambda path: path.stat().st_mtime)[:-128]:
        target = entry.resolve()
        if target != current and target != root and target.is_relative_to(root):
            shutil.rmtree(target, ignore_errors=True)


def video_query_evidence(asset_id, question, documents, index, settings, *, attempted=False):
    from helper.asset_catalog import read_job, source_file
    from helper.asset_index import RetrievalHit
    from helper.provider_status import probe_provider, setting

    job = read_job(asset_id)
    if job.get("source_type") != "video":
        return [], {"status": "not_video", "attempted": False}
    duration = float(
        job.get("duration_seconds")
        or max((item.metadata.get("end", 0) for item in documents), default=0)
    )
    target = parse_timestamp(question, duration)
    if target["status"] != "resolved":
        return [], {"status": target["status"], "attempted": False}
    timestamp = target["seconds"]
    temporal = index.temporal_hits(timestamp)
    spoken = bool(
        re.search(
            r"\b(?:narrator|narration|spoken|transcript|say|says|said|speech|audio|narrateur|parole|transcription|dit)\b|الراوي|يقول|الصوت|التفريغ",
            question,
            re.I,
        )
    )
    visual = bool(
        re.search(
            r"\b(?:visible|shown|screen|frame|see|look|displayed|affiché|image|écran)\b|يظهر|الشاشة|الصورة",
            question,
            re.I,
        )
    )
    if spoken and not visual:
        return [
            hit for hit in temporal if hit.document.metadata.get("content_type") == "transcript"
        ], {"status": "transcript_only", "timestamp": timestamp, "attempted": False}
    if any(
        hit.document.metadata.get("content_type") == "visual_description"
        and hit.document.metadata.get("timestamp_precision") == "frame"
        and hit.document.metadata.get("start") == hit.document.metadata.get("end") == timestamp
        for hit in temporal
    ):
        return temporal, {"status": "existing_visual", "timestamp": timestamp, "attempted": False}
    if job.get("vlm_mode") not in {"automatic", "enhanced"}:
        return temporal, {"status": "disabled", "timestamp": timestamp, "attempted": False}
    if attempted:
        return temporal, {
            "status": "attempt_already_used",
            "timestamp": timestamp,
            "attempted": True,
        }
    provider = setting(settings, "VISION_PROVIDER", "disabled")
    model = setting(settings, "VISION_MODEL")
    if provider not in {"OLLAMA", "GOOGLE_GENAI"} or not model:
        return temporal, {
            "status": "provider_unavailable",
            "attempted": True,
            "timestamp": timestamp,
        }
    try:
        original = next(value for value in job.get("files", []) if value.startswith("original."))
        source, _ = source_file(asset_id, original)
        source_hash = _file_hash(source)
        version = documents[0].metadata.get("asset_version")
        if source_hash != version:
            raise InvalidDocumentError("The original video changed after extraction")
        from tools.visual_understanding import _PROMPT_VERSION

        identity = {
            "asset_id": asset_id,
            "asset_version": version,
            "index_version": index.index_version,
            "source_hash": source_hash,
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "endpoint": _digest(setting(settings, "VISION_BASE_URL", "")),
            "prompt_version": _PROMPT_VERSION,
            "frame_dimension": setting(settings, "VIDEO_FRAME_MAX_DIMENSION", 960),
            "mode": job["vlm_mode"],
        }
        key = _digest(identity)
        with _LOCK:
            lock = _KEY_LOCKS.setdefault(key, RLock())
        with lock:
            try:
                document, _ = _load_cache(key, asset_id, index.index_version)
                return [RetrievalHit(document, 0.0, 1, priority=2), *temporal], {
                    "status": "cache_hit",
                    "timestamp": timestamp,
                    "attempted": True,
                    "derived_evidence_id": key,
                }
            except (OSError, ValueError, KeyError, TypeError):
                pass
            preflight = probe_provider(
                provider, model, settings, required_capabilities=("vision",), vision=True
            )
            if not preflight.get("available"):
                return temporal, {
                    "status": "provider_unavailable",
                    "timestamp": timestamp,
                    "attempted": True,
                    "provider_status": preflight.get("status"),
                }
            folder = _cache_directory(key)
            frame = folder / "frames" / ("derived_" + key + ".jpg")
            deadline = time.monotonic() + min(
                60, float(setting(settings, "VISION_TIMEOUT_SECONDS", 60))
            )
            try:
                _extract_target_frame(
                    source, frame, timestamp=timestamp, settings=settings, deadline=deadline
                )
                from tools.visual_understanding import analyze_artifacts

                reference = "frames/" + frame.name
                observations, _ = analyze_artifacts(
                    folder / "media.mp4",
                    [
                        {
                            "reference": reference,
                            "start": timestamp,
                            "end": timestamp,
                            "timestamp_precision": "frame",
                            "frame_index": 1,
                            "artifact_type": "video_frame",
                            "meaningful": True,
                        }
                    ],
                    asset_id=asset_id,
                    source_name=job.get("source_name", source.name),
                    source_type="video",
                    vlm_mode=job["vlm_mode"],
                    settings=settings,
                    deadline=deadline,
                )
                if not observations:
                    raise ProcessingError("The requested frame could not be interpreted")
                document = observations[0]
                document.id = key
                document.metadata.update(
                    chunk_id=key,
                    index_version=index.index_version,
                    asset_version=version,
                    record_id="derived_" + key,
                    content_hash=hashlib.sha256(document.page_content.encode()).hexdigest(),
                    derived_evidence_id=key,
                    derived_source_hash=source_hash,
                    derived_reference=reference,
                    evidence_origin="targeted_video_frame",
                )
                payload = {
                    "schema_version": 1,
                    "identity": identity,
                    "document": document.model_dump(),
                    "frame_hash": _file_hash(frame),
                }
                payload["document_hash"] = _digest(payload["document"])
                temporary = folder / (uuid4().hex + ".tmp")
                try:
                    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    temporary.replace(folder / "evidence.json")
                finally:
                    temporary.unlink(missing_ok=True)
                try:
                    _prune_cache(folder)
                except OSError:
                    pass  # Cache housekeeping must not discard valid source evidence.
                return [RetrievalHit(document, 0.0, 1, priority=2), *temporal], {
                    "status": "analyzed",
                    "timestamp": timestamp,
                    "attempted": True,
                    "derived_evidence_id": key,
                }
            finally:
                if not (folder / "evidence.json").exists():
                    frame.unlink(missing_ok=True)
                    try:
                        frame.parent.rmdir()
                        folder.rmdir()
                    except OSError:
                        pass
    except (ProcessingError, OSError, ValueError, KeyError, StopIteration, TypeError):
        return temporal, {"status": "unavailable", "timestamp": timestamp, "attempted": True}


def parse_timestamp(question, duration):
    values = []
    for match in re.finditer(
        r"(?<![\d:])(?:(\d{1,3}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)(?![\d:]|\.\d)", question
    ):
        hours, minutes, seconds = match.groups()
        if float(seconds) >= 60 or hours is not None and int(minutes) >= 60:
            return {"status": "invalid"}
        values.append((int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)))
    for match in re.finditer(
        r"(?<![\w.])(?:(\d+)\s*h\s*)?(\d+)\s*m(?=\s|\d|$)\s*(?:(\d+(?:\.\d+)?)\s*s?\b)?(?!\d|\.\d)",
        question,
        re.I,
    ):
        hours, minutes, seconds = match.groups()
        values.append(int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds or 0))
    values = list(dict.fromkeys(values))
    if not values:
        return {"status": "absent"}
    if len(values) != 1:
        return {"status": "ambiguous"}
    if not math.isfinite(duration) or not 0 <= values[0] < duration:
        return {"status": "out_of_range"}
    return {"status": "resolved", "seconds": float(values[0])}


def build_timeline(documents, duration, transitions=()):
    observations, concepts = [], []
    for document in documents:
        metadata = document.metadata
        if metadata.get("content_type") not in {"transcript", "visual_description"}:
            continue
        if not metadata.get("record_id"):
            raise ValueError("Timeline requires stable source record identifiers")
        if metadata.get("start") is None or metadata.get("end") is None:
            continue
        observation = {
            key: metadata[key]
            for key in (
                "record_id",
                "source",
                "source_type",
                "content_type",
                "start",
                "end",
                "timestamp_precision",
                "segment_id",
                "frame_index",
                "frame_reference",
            )
            if key in metadata
        }
        observation.update(
            text=document.page_content[:16000],
            text_truncated=len(document.page_content) > 16000,
            concept_status="unclassified_transcript"
            if metadata["content_type"] == "transcript"
            else "no_structured_visual_concepts",
        )
        if metadata["content_type"] == "visual_description":
            try:
                structure = json.loads(metadata.get("visual_structure", "{}"))
            except (ValueError, TypeError):
                structure = {}
            for field in ("elements", "labels", "operations", "annotations", "equations"):
                values = structure.get(field, []) if isinstance(structure, dict) else []
                if not isinstance(values, list):
                    continue
                for value in values[:64]:
                    text = (
                        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    )
                    concepts.append(
                        {
                            "source_record_id": metadata["record_id"],
                            "field": field,
                            "text": text[:1000],
                        }
                    )
                    observation["concept_status"] = "existing_visual_structure"
        observations.append(observation)
    observations.sort(
        key=lambda item: (item["start"], item["content_type"] == "visual_description")
    )
    speech = [item for item in observations if item["content_type"] == "transcript"]
    alignments = [
        {
            "visual_record_id": item["record_id"],
            "transcript_record_ids": [
                part["record_id"] for part in speech if part["start"] <= item["start"] < part["end"]
            ],
            "relation": "temporal_overlap_only",
        }
        for item in observations
        if item["content_type"] == "visual_description"
    ]
    return {
        "schema_version": 1,
        "duration_seconds": duration,
        "observations": observations,
        "alignments": alignments,
        "concepts": concepts,
        "transitions": [
            {"time": float(value), "method": "bounded_cpu_scene_detection"}
            for value in sorted(set(transitions))
            if isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value < duration
        ],
        "limitation": "Temporal overlap does not establish a causal relationship or prove that a frame was described in speech.",
    }
