import io
import json
import math
import re
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from helper import asset_paths
from helper.asset_paths import validate_asset_id
from helper.ingestion_errors import AssetNotFoundError, InvalidAssetError, StorageError

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
_TERMINAL = {"completed", "failed", "cancelled"}
_lock = RLock()
_SECRET_KEY = re.compile(
    r"api.?key|credential|password|secret|authorization|access.?token|refresh.?token|base64|binary|raw_bytes",
    re.I,
)
_CONFIG_KEYS = {
    "vlm_mode",
    "requested_vlm_mode",
    "organization_mode",
    "processing_mode",
    "question_type",
    "difficulty",
    "question_count",
    "requested_count",
    "number_of_questions",
    "source_scope",
    "action",
    "language",
    "ocr_enabled",
    "max_visuals",
    "automatic_max_visuals",
    "max_concurrency",
    "image_max_dimension",
    "source_file_count",
}
_MODEL_KEYS = {
    "provider",
    "model",
    "model_id",
    "temperature",
    "purpose",
    "cache_hit",
    "cached_visual_count",
    "requested_provider",
    "requested_model",
    "fallback_used",
    "fallback_reason",
    "preflight_status",
    "execution_verified",
    "executions",
}


class JobError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=128)
    detail: str = Field(min_length=1, max_length=2000)


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    file_id: str
    file_name: str
    file_type: str
    job_type: str
    status: JobStatus
    source_file_ids: list[str] = Field(default_factory=list)
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    processing_config: dict[str, JsonValue] = Field(default_factory=dict)
    model_info: dict[str, JsonValue] = Field(default_factory=dict)
    results: dict[str, JsonValue] = Field(default_factory=dict)
    errors: list[JobError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _public_text(value):
    text = str(value)
    text = re.sub(r"\bBearer\s+[A-Za-z0-9._~+/=\-]+", "[credential omitted]", text, flags=re.I)
    text = re.sub(
        r"(?i)(?:api[_ -]?key|password|access[_ -]?token|secret)\s*[:=]\s*[^\s,;]+",
        "[credential omitted]",
        text,
    )
    text = re.sub(r"(?i)\b[a-z]:[\\/][^\s\"<>]*", "[private path omitted]", text)
    text = re.sub(
        r"(?<!\w)/(?:Users|home|tmp|var|etc|mnt|opt|workspace|root|private)/[^\s\"<>]*",
        "[private path omitted]",
        text,
    )
    text = re.sub(r"data:[^;\s]+;base64,[A-Za-z0-9+/=]+", "[binary omitted]", text, flags=re.I)
    return text


def public_value(value, *, depth=0):
    if depth > 30:
        return "[nested content omitted]"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if hasattr(value, "page_content") and hasattr(value, "metadata"):
        value = {"text": value.page_content, "metadata": value.metadata}
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[binary omitted]"
    if isinstance(value, Path):
        return "[private path omitted]"
    if isinstance(value, str):
        return _public_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): public_value(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, (str, int))
            and not _SECRET_KEY.search(str(key))
            and not str(key).lower().endswith(("_path", "_directory"))
        }
    if isinstance(value, (list, tuple, set)):
        return [public_value(item, depth=depth + 1) for item in value]
    return "[unsupported value omitted]"


def _configuration(value, allowed):
    if not isinstance(value, Mapping):
        return {}
    return {key: public_value(item) for key, item in value.items() if key in allowed}


def _directory():
    root = asset_paths.ASSETS_ROOT.resolve()
    directory = (root / ".job-results").resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise StorageError("Invalid operation result directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _job_path(job_id):
    try:
        if not isinstance(job_id, str) or str(UUID(job_id)) != job_id:
            raise ValueError
    except ValueError:
        raise InvalidAssetError("Invalid operation identifier") from None
    directory = _directory()
    target = (directory / (job_id + ".json")).resolve()
    if not target.is_relative_to(directory):
        raise InvalidAssetError("Invalid operation result path")
    return target


def _read(job_id):
    try:
        return JobResult.model_validate_json(
            _job_path(job_id).read_text(encoding="utf-8")
        ).model_dump(mode="json")
    except FileNotFoundError:
        raise AssetNotFoundError("Processing job not found") from None
    except (OSError, ValueError) as error:
        raise StorageError("Processing job results could not be read") from error


def _write(value):
    value = JobResult.model_validate(value).model_dump(mode="json")
    path = _job_path(value["job_id"])
    temporary = path.with_suffix("." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
        # Reuse the existing Windows status-file retry policy.
        from helper.asset_catalog import _status_io

        _status_io(lambda: temporary.replace(path))
    except OSError as error:
        raise StorageError("Processing job results could not be saved") from error
    finally:
        temporary.unlink(missing_ok=True)
    return value


def start_job(
    file_id,
    file_name,
    file_type,
    job_type,
    processing_config=None,
    source_file_ids=None,
    *,
    status="running",
    model_info=None,
):
    validate_asset_id(file_id)
    sources = list(dict.fromkeys(source_file_ids or [file_id]))
    for source in sources:
        validate_asset_id(source)
    if status not in {"pending", "running"}:
        raise ValueError("A new job must be pending or running")
    if not isinstance(job_type, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", job_type) is None:
        raise ValueError("Invalid processing operation type")
    if not isinstance(file_type, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,39}", file_type) is None:
        raise ValueError("Invalid source type")
    now, identifier = _now(), str(uuid4())
    name = Path(PureWindowsPath(str(file_name)).name).name[:255] or "Unnamed source"
    with _lock:
        _write(
            {
                "schema_version": "1.0",
                "job_id": identifier,
                "file_id": file_id,
                "file_name": _public_text(name),
                "file_type": file_type,
                "job_type": job_type,
                "status": status,
                "source_file_ids": sources,
                "created_at": now,
                "started_at": now if status == "running" else None,
                "processing_config": _configuration(processing_config, _CONFIG_KEYS),
                "model_info": _configuration(model_info, _MODEL_KEYS),
            }
        )
    return identifier


def mark_job_running(job_id):
    with _lock:
        value = _read(job_id)
        if value["status"] in _TERMINAL:
            raise ValueError("A terminal operation cannot be restarted")
        if value["status"] == "pending":
            value.update(status="running", started_at=_now())
            value = _write(value)
        return value


def _finish(job_id, status, results, warnings, model_info, errors, *, completed_at=None):
    with _lock:
        value = _read(job_id)
        if value["status"] in _TERMINAL:
            raise ValueError(
                "A completed operation is immutable; start a new job for another version"
            )
        now = completed_at or _now()
        datetime.fromisoformat(now)
        started = value["started_at"] or value["created_at"]
        elapsed = max(
            0.0, (datetime.fromisoformat(now) - datetime.fromisoformat(started)).total_seconds()
        )
        safe_results = public_value(results or {})
        if not isinstance(safe_results, dict):
            raise ValueError("Operation results must be a JSON object")
        value.update(
            status=status,
            started_at=started,
            completed_at=now,
            duration_seconds=round(elapsed, 6),
            results=safe_results,
            warnings=list(dict.fromkeys(_public_text(item) for item in (warnings or []))),
            errors=errors or [],
        )
        if model_info is not None:
            value["model_info"].update(_configuration(model_info, _MODEL_KEYS))
        return _write(value)


def complete_job(job_id, results, warnings=None, model_info=None, *, completed_at=None):
    return _finish(
        job_id, "completed", results, warnings, model_info, [], completed_at=completed_at
    )


def fail_job(job_id, error, warnings=None, results=None, *, completed_at=None, model_info=None):
    if isinstance(error, dict):
        category = str(error.get("category", "processing"))[:128]
        detail = _public_text(error.get("detail", "The operation could not be completed."))[:2000]
    else:
        category = "processing"
        detail = "The operation could not be completed. Inspect the input and configured processing service."
    return _finish(
        job_id,
        "failed",
        results,
        warnings,
        model_info,
        [{"category": category, "detail": detail}],
        completed_at=completed_at,
    )


def cancel_job(job_id, *, model_info=None, warnings=None):
    return _finish(
        job_id,
        "cancelled",
        {},
        ["The operation was cancelled before it completed.", *(warnings or [])],
        model_info,
        [],
    )


def record_job(
    file_id,
    file_name,
    file_type,
    job_type,
    results,
    processing_config=None,
    source_file_ids=None,
    warnings=None,
    model_info=None,
):
    identifier = start_job(
        file_id,
        file_name,
        file_type,
        job_type,
        processing_config,
        source_file_ids,
        model_info=model_info,
    )
    return complete_job(identifier, results, warnings, model_info)


def get_job_result(file_id, job_id):
    validate_asset_id(file_id)
    with _lock:
        value = _read(job_id)
    if value["file_id"] != file_id:
        raise AssetNotFoundError("Processing job not found for this source")
    return value


def list_job_results(file_id):
    validate_asset_id(file_id)
    values = []
    with _lock:
        for path in _directory().glob("*.json"):
            try:
                value = _read(path.stem)
            except (StorageError, InvalidAssetError, AssetNotFoundError):
                continue
            if value["file_id"] == file_id:
                values.append({key: item for key, item in value.items() if key != "results"})
    return sorted(values, key=lambda item: (item["created_at"], item["job_id"]), reverse=True)


def export_job_results(file_ids):
    if not isinstance(file_ids, list) or not 1 <= len(file_ids) <= 200:
        raise ValueError("Select between one and 200 source files")
    ids = list(dict.fromkeys(file_ids))
    for file_id in ids:
        validate_asset_id(file_id)
    buffer, manifest = io.BytesIO(), []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_id in ids:
            for summary in list_job_results(file_id):
                value = get_job_result(file_id, summary["job_id"])
                name = file_id + "/" + value["job_id"] + ".json"
                archive.writestr(
                    name, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
                )
                manifest.append(
                    {
                        "file_id": file_id,
                        "job_id": value["job_id"],
                        "job_type": value["job_type"],
                        "reference": name,
                    }
                )
        archive.writestr(
            "manifest.json",
            json.dumps(
                {"schema_version": "1.0", "exported_at": _now(), "file_ids": ids, "jobs": manifest},
                ensure_ascii=False,
                indent=2,
            ),
        )
    return buffer.getvalue()


def recover_interrupted_jobs():
    recovered = 0
    with _lock:
        for path in _directory().glob("*.json"):
            try:
                value = _read(path.stem)
            except (AssetNotFoundError, InvalidAssetError, StorageError):
                continue
            if value["status"] in {"pending", "running"}:
                fail_job(
                    value["job_id"],
                    {
                        "category": "interrupted",
                        "detail": "The process stopped before this operation completed. Start a new operation to retry.",
                    },
                    warnings=value["warnings"],
                    results=value["results"],
                )
                recovered += 1
    return recovered
