import json
from datetime import datetime, timezone
from threading import RLock

from helper.job_results import complete_job, fail_job, public_value, start_job

_STAGES = {
    "extracting_document_text": "text_extraction",
    "extracting_document_ocr": "ocr",
    "text_extraction_completed": "text_extraction",
    "table_extraction_completed": "table_extraction",
    "detecting_visual_content": "visual_detection",
    "detecting_visuals": "visual_detection",
    "sampling_frames": "visual_detection",
    "visual_artifacts_detected": "visual_detection",
    "visual_analysis_completed": "visual_detection",
    "analyzing_visuals": "vlm_understanding",
    "reusing_visuals": "visual_cache_reuse",
    "extracting_audio": "audio_extraction",
    "transcribing": "transcription",
    "merging_multimodal": "multimodal_merging",
    "inspecting_media": "media_inspection",
}


def _setting(settings, key, default=None):
    return (
        settings.get(key, default)
        if isinstance(settings, dict)
        else getattr(settings, key, default)
    )


def _record(document):
    if isinstance(document, dict):
        metadata = dict(document.get("metadata", {}))
        text = document.get("page_content", document.get("text", ""))
    else:
        metadata, text = dict(document.metadata), document.page_content
    result = {"text": text, "metadata": metadata}
    for key in ("visual_structure", "table_data", "bbox"):
        if key in metadata and isinstance(metadata[key], str):
            try:
                result[key] = json.loads(metadata[key])
            except ValueError:
                pass
    return result


class ExtractionJobs:
    def __init__(self, prepared, settings):
        self.prepared, self.settings = prepared, settings
        self.jobs, self.events, self.outputs = {}, {}, {}
        self.end_times = {}
        self.completed = set()
        self.artifacts = []
        self._lock = RLock()
        self.config = {
            "vlm_mode": prepared.vlm_mode,
            "requested_vlm_mode": prepared.requested_vlm_mode,
        }

    def begin(self, job_type):
        with self._lock:
            if job_type not in self.jobs:
                model_info = None
                if job_type in {"vlm_understanding", "visual_cache_reuse"}:
                    model_info = {
                        "provider": _setting(self.settings, "VISION_PROVIDER"),
                        "model": _setting(self.settings, "VISION_MODEL"),
                    }
                elif job_type == "transcription":
                    model_info = {
                        "provider": "WHISPER",
                        "model": _setting(self.settings, "WHISPER_MODEL", "tiny"),
                    }
                self.jobs[job_type] = start_job(
                    self.prepared.asset_id,
                    self.prepared.source_name,
                    self.prepared.source_type,
                    job_type,
                    self.config,
                    model_info=model_info,
                )
            return self.jobs[job_type]

    def progress(self, stage, details=None):
        details = details or {}
        job_type = _STAGES.get(stage)
        if job_type is None:
            return
        with self._lock:
            self.begin(job_type)
            now = datetime.now(timezone.utc).isoformat()
            if stage in {"detecting_visuals", "visual_artifacts_detected"}:
                for finished in ("text_extraction", "media_inspection"):
                    if finished in self.jobs:
                        self.end_times.setdefault(finished, now)
            if stage == "visual_artifacts_detected":
                self.end_times["visual_detection"] = now
            if stage == "transcribing":
                if "audio_extraction" in self.jobs:
                    self.end_times["audio_extraction"] = now
                if details.get("status") == "completed":
                    self.end_times["transcription"] = now
            if stage == "extracting_audio" and "media_inspection" in self.jobs:
                self.end_times.setdefault("media_inspection", now)
            if stage == "merging_multimodal":
                for finished in ("vlm_understanding", "visual_cache_reuse"):
                    if finished in self.jobs:
                        self.end_times[finished] = now
            summary = {
                key: value for key, value in details.items() if key not in {"records", "artifacts"}
            }
            if isinstance(details.get("records"), list):
                summary["record_count"] = len(details["records"])
            if isinstance(details.get("artifacts"), list):
                summary["artifact_count"] = len(details["artifacts"])
            self.events.setdefault(job_type, []).append(
                {"stage": stage, "at": now, **public_value(summary)}
            )
            if stage in {"visual_artifacts_detected", "visual_analysis_completed"}:
                self.artifacts = details.get("artifacts", [])
            if isinstance(details.get("records"), list):
                self.outputs.setdefault(job_type, []).extend(details["records"])
            if (
                stage in {"text_extraction_completed", "table_extraction_completed"}
                and job_type not in self.completed
            ):
                records = [_record(record) for record in self.outputs.get(job_type, [])]
                tables = [
                    record
                    for record in records
                    if record["metadata"].get("content_type") == "table" or "table_data" in record
                ]
                payload = (
                    {"tables": tables}
                    if job_type == "table_extraction"
                    else {"text_blocks": [record for record in records if record not in tables]}
                )
                complete_job(
                    self.jobs[job_type],
                    {**payload, "execution_stages": self.events[job_type]},
                    warnings=details.get("warnings"),
                    completed_at=now,
                )
                self.completed.add(job_type)
                self.end_times[job_type] = now

    def outputs_ready(self):
        now = datetime.now(timezone.utc).isoformat()
        for job_type in self.jobs:
            self.end_times.setdefault(job_type, now)

    def finish(self, documents, metadata):
        records = [_record(document) for document in documents]
        ordinary = [
            record
            for record in records
            if record["metadata"].get("content_type") not in {"visual_description", "transcript"}
        ]
        tables = [
            record
            for record in ordinary
            if record["metadata"].get("content_type") == "table" or "table_data" in record
        ]
        visuals = [
            record
            for record in records
            if record["metadata"].get("content_type") == "visual_description"
        ]
        cached_visuals = [
            record for record in visuals if record["metadata"].get("cache_hit") is True
        ]
        inferred_visuals = [
            record for record in visuals if record["metadata"].get("cache_hit") is not True
        ]
        transcript = [
            record
            for record in records
            if record["metadata"].get("content_type") == "transcript"
            or (
                self.prepared.source_type in {"audio", "video"}
                and record["metadata"].get("content_type") != "visual_description"
                and "start" in record["metadata"]
                and "end" in record["metadata"]
            )
        ]
        ocr = [
            record
            for record in ordinary
            if record["metadata"].get("ocr") or record["metadata"].get("extraction_method") == "ocr"
        ]
        if ocr:
            self.begin("ocr")
        if self.prepared.source_type in {"pdf", "docx"} and "text_extraction" in self.jobs:
            self.begin("table_extraction")
        counts = {key: value for key, value in metadata.items() if key.endswith("_visual_count")}
        payloads = {
            "text_extraction": {
                "text_blocks": [record for record in ordinary if record not in tables]
            },
            "table_extraction": {"tables": tables},
            "ocr": {"text_blocks": ocr},
            "visual_detection": {"visual_artifacts": self.artifacts, **counts},
            "vlm_understanding": {
                "visual_artifacts": inferred_visuals,
                "processing_status": metadata.get("visual_status"),
                **counts,
            },
            "visual_cache_reuse": {
                "visual_artifacts": cached_visuals,
                "cached_visual_count": len(cached_visuals),
            },
            "transcription": {
                "transcript": transcript,
                "audio_status": metadata.get("audio_status"),
            },
            "audio_extraction": {
                "audio_status": metadata.get("audio_status"),
                "duration_seconds": metadata.get("duration_seconds"),
            },
            "media_inspection": {
                key: metadata[key]
                for key in ("duration_seconds", "audio_status", "audio_reason")
                if key in metadata
            },
            "multimodal_merging": {
                "merged_content": records,
                "processing_scope": metadata.get("processing_scope"),
            },
        }
        for job_type, identifier in self.jobs.items():
            if job_type in self.completed:
                continue
            results = {
                **payloads.get(job_type, {}),
                "execution_stages": self.events.get(job_type, []),
            }
            model = None
            if job_type in {"vlm_understanding", "visual_cache_reuse"}:
                model = {
                    "provider": _setting(self.settings, "VISION_PROVIDER"),
                    "model": _setting(self.settings, "VISION_MODEL"),
                    "cached_visual_count": metadata.get("cached_visual_count", 0),
                }
            elif job_type == "transcription":
                model = {
                    "provider": "WHISPER",
                    "model": _setting(self.settings, "WHISPER_MODEL", "tiny"),
                }
            warnings = metadata.get("warnings", [])
            failed = (
                (job_type == "vlm_understanding" and metadata.get("visual_status") == "failed")
                or (
                    job_type == "transcription"
                    and metadata.get("audio_status") in {"failed", "partial"}
                )
                or (job_type == "ocr" and not ocr)
            )
            if failed:
                fail_job(
                    identifier,
                    {
                        "category": job_type,
                        "detail": "This stage did not complete successfully; available outputs are preserved.",
                    },
                    warnings=warnings,
                    results=results,
                )
            else:
                complete_job(
                    identifier,
                    results,
                    warnings=warnings,
                    model_info=model,
                    completed_at=self.end_times.get(job_type),
                )
            self.completed.add(job_type)

    def fail(self, error):
        from helper.job_results import _read

        for job_type, identifier in self.jobs.items():
            if _read(identifier)["status"] not in {"completed", "failed", "cancelled"}:
                fail_job(
                    identifier,
                    error,
                    results={
                        "execution_stages": self.events.get(job_type, []),
                        "partial_records": self.outputs.get(job_type, []),
                    },
                )
