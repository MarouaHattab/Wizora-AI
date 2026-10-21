import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePosixPath
from threading import RLock
from uuid import uuid4

from . import asset_paths
from .asset_paths import asset_directory, validate_asset_id
from .ingestion_errors import AssetNotFoundError, InvalidAssetError, StorageError

_lock = RLock()


def _status_io(operation):
    for attempt in range(6):
        try:
            return operation()
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            # Windows scanners can briefly hold an otherwise closed status file.
            time.sleep(.02 * 2 ** attempt)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _job_path(asset_id):
    validate_asset_id(asset_id)
    root = asset_paths.ASSETS_ROOT.resolve()
    directory = (root / ".jobs").resolve()
    if not directory.is_relative_to(root):
        raise StorageError("Invalid processing status directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{asset_id}.json"


def read_job(asset_id):
    path = _job_path(asset_id)
    try:
        with _lock:
            value = json.loads(_status_io(lambda: path.read_text(encoding="utf-8")))
        if not isinstance(value, dict) or value.get("id") != asset_id:
            raise StorageError("Invalid processing status")
        return value
    except FileNotFoundError:
        return _legacy_asset(asset_id)
    except (OSError, ValueError) as exc:
        raise StorageError("Unable to read processing status") from exc


def write_job(asset_id, **updates):
    with _lock:
        path = _job_path(asset_id)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            value = json.loads(_status_io(lambda: path.read_text(encoding="utf-8"))) if path.exists() else {"id": asset_id, "created_at": _now(), "stages": []}
            if not isinstance(value, dict) or value.get("id") != asset_id:
                raise StorageError("Invalid processing status")
            value.update(updates, updated_at=_now())
            temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            _status_io(lambda: temporary.replace(path))
        except (OSError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise StorageError("Unable to record processing status") from exc
        return value


def record_stage(asset_id, stage, details=None):
    with _lock:
        value = read_job(asset_id)
        stages = list(value.get("stages", []))
        safe = {key: item for key, item in (details or {}).items()
                if key in {"segment", "segments", "count", "duration_seconds", "timestamp", "status"}
                and isinstance(item, (str, int, float, bool))}
        event = {"name": stage, "at": _now(), **safe}
        if stages and stages[-1]["name"] == stage:
            stages[-1] = event
        else:
            stages.append(event)
        return write_job(asset_id, status="processing", stages=stages[-256:])


def _legacy_asset(asset_id):
    from .asset_storage import load_asset_documents
    documents = load_asset_documents(asset_id)
    originals = sorted(asset_directory(asset_id).glob("original.*"))
    return {"id": asset_id, "status": "ready", "source_name": documents[0].metadata.get("source", asset_id),
            "source_type": documents[0].metadata.get("source_type", "text"), "record_count": len(documents),
            "index_status": "deferred", "files": [path.name for path in originals], "stages": [],
            "content_types": dict(Counter(document.metadata.get("content_type", "text") for document in documents))}


def public_job(value):
    allowed = {"id", "status", "source_name", "source_type", "size_bytes", "created_at", "updated_at",
               "stages", "error", "record_count", "content_types", "language", "languages", "pages", "sections",
               "duration_seconds", "audio_status", "audio_reason", "visual_status", "frame_count", "processing_scope", "index_status",
               "index_version", "chunk_count", "files", "warnings"}
    public = {key: item for key, item in value.items() if key in allowed}
    if public.get("index_status") == "ready" and value.get("index_process") != os.getpid():
        public.update(index_status="deferred")
        public.pop("index_version", None)
        public.pop("chunk_count", None)
    return public


def list_assets(limit=100):
    root = asset_paths.ASSETS_ROOT.resolve()
    if not root.exists():
        return []
    results = []
    for directory in sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            value = read_job(directory.name)
            if value.get("status") == "ready":
                results.append(public_job(value))
        except Exception:
            continue
        if len(results) >= limit:
            break
    return results


def mark_index_ready(asset_id, index_version, chunk_count):
    try:
        if _job_path(asset_id).exists():
            write_job(asset_id, index_status="ready", index_version=index_version,
                      chunk_count=chunk_count, index_process=os.getpid())
    except StorageError:
        # An unavailable status cache must not invalidate a verified search index.
        pass


def source_file(asset_id, reference):
    value = read_job(asset_id)
    if value.get("status") != "ready":
        raise AssetNotFoundError("The asset is not ready")
    if not isinstance(reference, str) or reference not in value.get("files", []):
        raise InvalidAssetError("The source reference is unavailable")
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or "\\" in reference:
        raise InvalidAssetError("Invalid source reference")
    directory = asset_directory(asset_id, must_exist=True)
    target = (directory / reference).resolve()
    if not target.is_relative_to(directory) or not target.is_file():
        raise AssetNotFoundError("The source file is unavailable")
    return target, value


def recover_interrupted_uploads():
    directory = asset_paths.ASSETS_ROOT.resolve() / ".jobs"
    for path in directory.glob("*.json") if directory.is_dir() else ():
        try:
            with _lock:
                value = json.loads(_status_io(lambda: path.read_text(encoding="utf-8")))
            if (isinstance(value, dict) and value.get("id") == path.stem
                    and value.get("status") in ("queued", "processing")):
                write_job(value["id"], status="failed", error={"category": "interrupted", "detail": "Processing was interrupted. Upload the file again."})
        except (OSError, ValueError, StorageError, InvalidAssetError):
            continue
    root = asset_paths.ASSETS_ROOT.resolve()
    staging = (root / ".staging").resolve()
    if staging.is_relative_to(root) and staging != root and staging.is_dir():
        for entry in staging.iterdir():
            target = entry.resolve()
            if target.is_relative_to(staging) and target != staging and target.is_dir() and not entry.is_symlink():
                shutil.rmtree(target)
