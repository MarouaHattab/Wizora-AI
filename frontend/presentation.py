import math
import re


def markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", str(value))


def timestamp_text(value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        return ""
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def asset_details(asset):
    rows = []
    if asset.get("source_name"):
        rows.append("**File:** " + markdown_text(asset["source_name"]))
    facts = []
    if asset.get("source_type"):
        facts.append(str(asset["source_type"]).upper())
    size = asset.get("size_bytes")
    if (
        isinstance(size, (int, float))
        and not isinstance(size, bool)
        and math.isfinite(size)
        and size >= 0
    ):
        facts.append(f"{size / 1024**2:.1f} MB" if size >= 1024**2 else f"{size / 1024:.1f} KB")
    duration = timestamp_text(asset.get("duration_seconds"))
    if duration:
        facts.append("Duration " + duration)
    if asset.get("record_count") is not None:
        facts.append(f"{asset['record_count']} content records")
    if asset.get("frame_count") is not None:
        facts.append(f"{asset['frame_count']} analyzed frames")
    if asset.get("pages"):
        pages = asset["pages"]
        facts.append(
            "Pages "
            + (", ".join(str(page) for page in pages) if isinstance(pages, list) else str(pages))
        )
    languages = asset.get("languages") or asset.get("language")
    if languages:
        facts.append(
            "Language "
            + (
                ", ".join(str(item) for item in languages)
                if isinstance(languages, list)
                else str(languages)
            )
        )
    if facts:
        rows.append(markdown_text(" · ".join(facts)))
    scope = asset.get("processing_scope")
    scopes = {
        "audio_visual": "Audio transcript and visual analysis",
        "transcript_and_visuals": "Audio transcript and visual analysis",
        "visual_only": "Visual analysis only",
        "transcript_only": "Transcript only",
        "text_only": "Text extraction",
    }
    if scope:
        rows.append("**Processing scope:** " + markdown_text(scopes.get(scope, scope)))
    for field, label in (("audio_status", "Audio"), ("visual_status", "Visual analysis")):
        if asset.get(field):
            rows.append(f"**{label}:** " + markdown_text(str(asset[field]).replace("_", " ")))
    mode_names = {
        "disabled": "Disabled — Text / Audio Only",
        "automatic": "Automatic — Hybrid Extraction",
        "enhanced": "Enhanced — Comprehensive Visual Understanding",
    }
    requested, effective = asset.get("requested_vlm_mode"), asset.get("vlm_mode")
    if requested in mode_names:
        rows.append("**Requested visual mode:** " + markdown_text(mode_names[requested]))
    if effective in mode_names:
        rows.append("**Applied visual mode:** " + markdown_text(mode_names[effective]))
    visual_counts = []
    for field, label in (
        ("detected_visual_count", "detected"),
        ("selected_visual_count", "selected"),
        ("analyzed_visual_count", "analyzed"),
    ):
        count = asset.get(field)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            visual_counts.append(f"{count} {label}")
    if visual_counts:
        rows.append("**Visual content:** " + " · ".join(visual_counts))
    warnings = asset.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    if asset.get("source_type") == "video" and scope == "transcript_only":
        warnings = [
            "Only spoken audio is available. Slides, on-screen text and visual actions were not analyzed.",
            *warnings,
        ]
    if warnings:
        rows.append(
            "**Processing warnings**\n\n"
            + "\n".join("- " + markdown_text(item) for item in warnings)
        )
    files = asset.get("files") or []
    if files:
        rows.append("**Available files:** " + ", ".join(markdown_text(item) for item in files))
    return "\n\n".join(rows)


def upload_progress(job):
    labels = {
        "received": "File received",
        "processing": "Processing file",
        "inspecting_media": "Inspecting media",
        "extracting_audio": "Extracting audio",
        "transcribing": "Transcribing audio",
        "extracting_frames": "Extracting video frames",
        "analyzing_frames": "Analyzing video visuals",
        "extracting_document": "Extracting document text and tables",
        "extracting_document_text": "Extracting document text and tables",
        "extracting_document_ocr": "Recognizing scanned document text",
        "detecting_visuals": "Detecting relevant visual content",
        "detecting_visual_content": "Detecting relevant visual content",
        "analyzing_visuals": "Analyzing diagrams with VLM",
        "visual_analysis": "Analyzing video visuals",
        "merging_multimodal": "Merging multimodal information",
        "pdf_extracted": "PDF text extracted",
        "docx_extracted": "DOCX text extracted",
        "text_extracted": "Text extracted",
        "records_validated": "Content records validated",
        "indexing": "Building source index",
        "ready": "Ready",
    }
    rows = []
    for stage in (job.get("stages") or [])[-8:]:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name", "processing"))
        label = labels.get(name, name.replace("_", " ").capitalize())
        if stage.get("segment") is not None:
            label += f" · segment {stage['segment']}" + (
                f" of {stage['segments']}" if stage.get("segments") else ""
            )
        if name == "analyzing_visuals" and stage.get("count") is not None:
            label += f" · {stage['count']}" + (
                f" of {stage['total']}" if stage.get("total") is not None else ""
            )
        if stage.get("status") == "completed":
            label += " · complete"
        rows.append("- " + markdown_text(label))
    title = (
        "🔄 File queued for processing…"
        if job.get("status") == "queued"
        else "🔄 Processing your file…"
    )
    return title + ("\n\n" + "\n".join(rows) if rows else "")
