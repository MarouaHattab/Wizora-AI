import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

from helper.ingestion_errors import ProcessingError


def setting(settings, name, default=None):
    return (
        settings.get(name, default)
        if isinstance(settings, Mapping)
        else getattr(settings, name, default)
    )


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ProcessingError("The document processing deadline was exceeded.")


def notify(progress, stage, **details):
    if callable(progress):
        progress(stage, details)


def store_png(source, data):
    name = hashlib.sha256(data).hexdigest() + ".png"
    destination = Path(source).parent / "images" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        destination.write_bytes(data)
    return "images/" + name


def bounds(metadata):
    value = metadata.get("bbox", [0, 0, 0, 0])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
    return value if isinstance(value, (list, tuple)) and len(value) == 4 else [0, 0, 0, 0]


def finish_records(documents, *, vlm_mode):
    for index, document in enumerate(documents):
        document.metadata.update(record_index=index, vlm_mode=vlm_mode)
        document.metadata.setdefault("processing_status", "completed")
        if isinstance(document.metadata.get("bbox"), (tuple, list)):
            document.metadata["bbox"] = json.dumps(list(document.metadata["bbox"]))
    return documents


def visual_metadata(vlm_mode):
    return {
        "vlm_mode": vlm_mode,
        "visual_status": "disabled" if vlm_mode == "disabled" else "no_visuals",
        "detected_visual_count": 0,
        "meaningful_visual_count": 0,
        "selected_visual_count": 0,
        "analyzed_visual_count": 0,
        "duplicate_visual_count": 0,
        "cached_visual_count": 0,
        "files": [],
        "warnings": [],
    }


def merge_metadata(result, visual):
    warnings = result.get("warnings", []) + visual.get("warnings", [])
    files = result.get("files", []) + visual.get("files", [])
    result.update(visual)
    result.update(warnings=list(dict.fromkeys(warnings)), files=sorted(set(files)))
