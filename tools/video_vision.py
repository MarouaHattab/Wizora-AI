import hashlib
import math
import re
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from helper.ingestion_errors import ProcessingError
from helper.vlm_policy import normalize_vlm_mode, vlm_available


def frame_timestamps(duration_seconds, *, interval_seconds=30, max_frames=12):
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or not math.isfinite(interval_seconds)
        or interval_seconds <= 0
        or type(max_frames) is not int
        or not 1 <= max_frames <= 64
    ):
        raise ValueError(
            "Frame sampling requires positive duration, interval and a bounded frame count."
        )
    count = min(max_frames, max(1, math.ceil(duration_seconds / interval_seconds)))
    return [(position + 0.5) * duration_seconds / count for position in range(count)]


def _scene_timestamps(source, info, *, settings, deadline):
    from tools.media_processing import _binary, _remaining, _run, _setting

    budget = min(20, max(1, _remaining(deadline) / 4))
    scene_deadline = min(deadline, time.monotonic() + budget)
    fps = min(16, max(2, float(_setting(settings, "VIDEO_SCENE_SCAN_FPS", 8))))
    # Limit diagnostic candidates across the whole duration rather than stopping
    # decoding after early animation changes. The inference frame limit is separate.
    spacing = max(1 / fps, info.duration_seconds / 512)
    try:
        _, diagnostic, _ = _run(
            [
                _binary(settings, "ffmpeg", required=True),
                "-hide_banner",
                "-nostdin",
                "-nostats",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"fps={fps:g},scale=320:-2,select='gt(scene,0.0005)*(isnan(prev_selected_t)+gte(t-prev_selected_t,{spacing:g}))',metadata=print:key=lavfi.scene_score",
                "-f",
                "null",
                "-",
            ],
            scene_deadline,
        )
        values = [
            float(value)
            for value in re.findall(
                r"pts_time:([\d.eE+\-]+)", diagnostic.decode("utf-8", errors="replace")
            )
        ]
        return [
            min(info.duration_seconds - 0.001, value + 0.5 / fps)
            for value in values
            if math.isfinite(value) and 0 <= value < info.duration_seconds
        ]
    except ProcessingError:
        return []


def _representative_timestamps(source, info, *, settings, deadline, discovery=None):
    from tools.media_processing import _setting

    maximum = min(64, max(1, int(_setting(settings, "VIDEO_MAX_FRAMES", 12))))
    anchors = frame_timestamps(
        info.duration_seconds,
        interval_seconds=float(_setting(settings, "VIDEO_FRAME_INTERVAL_SECONDS", 30)),
        max_frames=maximum,
    )
    scenes = _scene_timestamps(source, info, settings=settings, deadline=deadline)
    if discovery is not None:
        discovery.extend(scenes)
    if not scenes:
        return anchors
    scene_count = min(len(scenes), max(1, maximum * 2 // 3))
    selected = []
    for bucket in range(scene_count):
        left, right = (
            bucket * info.duration_seconds / scene_count,
            (bucket + 1) * info.duration_seconds / scene_count,
        )
        choices = [(index, value) for index, value in enumerate(scenes) if left <= value < right]
        if choices:
            # Prefer a brief entry inside each time bucket; animation cannot
            # consume every slot in the first seconds of a long source.
            _, value = min(
                choices,
                key=lambda pair: (
                    not (pair[0] + 1 < len(scenes) and 0 < scenes[pair[0] + 1] - pair[1] <= 0.75),
                    abs(pair[1] - (left + right) / 2),
                ),
            )
            selected.append(value)
    for index in range(scene_count):
        value = scenes[min(len(scenes) - 1, int((index + 0.5) * len(scenes) / scene_count))]
        if value not in selected:
            selected.append(value)
    selected = selected[:scene_count]
    anchor_count = maximum - len(selected)
    if anchor_count:
        selected.extend(
            frame_timestamps(
                info.duration_seconds,
                interval_seconds=info.duration_seconds / anchor_count,
                max_frames=anchor_count,
            )
        )
    return sorted(set(round(value, 3) for value in selected))


def frame_priority(image, previous=None):
    from tools.visual_understanding import _educational_layout

    similarity = 0.0
    identical = False
    if previous is not None and image.size == previous.size:
        difference = ImageChops.difference(image.convert("RGB"), previous.convert("RGB"))
        identical = difference.getbbox() is None
        similarity = 1 - sum(ImageStat.Stat(difference).mean) / (3 * 255)
    educational = _educational_layout(image, mode="automatic")
    return {
        "priority": float(educational) + (0.5 if 0.9 < similarity < 1 else 0),
        "similarity": round(similarity, 6),
        "identical": identical,
    }


def sample_video_frames(source, info, *, settings, deadline, progress=None):
    from tools.media_processing import _binary, _progress, _run, _setting

    source = Path(source)
    _progress(progress, "detecting_visuals")
    discovery = []
    timestamps = _representative_timestamps(
        source, info, settings=settings, deadline=deadline, discovery=discovery
    )
    maximum = min(2048, max(128, int(_setting(settings, "VIDEO_FRAME_MAX_DIMENSION", 960))))
    directory = source.parent / "frames"
    directory.mkdir(exist_ok=True)
    temporary = directory / ".sample.jpg"
    ffmpeg = _binary(settings, "ffmpeg", required=True)
    frames, fingerprints, previous = [], set(), None
    try:
        for position, timestamp in enumerate(timestamps, 1):
            _progress(
                progress,
                "sampling_frames",
                count=position,
                total=len(timestamps),
                timestamp=timestamp,
            )
            temporary.unlink(missing_ok=True)
            try:
                _run(
                    [
                        ffmpeg,
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
                        "-vf",
                        f"scale=w='min({maximum},iw)':h='min({maximum},ih)':force_original_aspect_ratio=decrease",
                        "-q:v",
                        "3",
                        "-threads",
                        "1",
                        "-update",
                        "1",
                        str(temporary),
                    ],
                    deadline,
                )
            except ProcessingError:
                frames.append(
                    {
                        "timestamp": timestamp,
                        "start": timestamp,
                        "end": timestamp,
                        "reference": "",
                        "frame_index": position,
                        "artifact_type": "video_frame",
                        "processing_status": "failed",
                        "reason": "frame_decode_failed",
                    }
                )
                if time.monotonic() >= deadline:
                    frames.extend(
                        {
                            "start": value,
                            "end": value,
                            "reference": "",
                            "artifact_type": "video_frame",
                            "processing_status": "skipped_budget",
                            "reason": "processing_deadline",
                        }
                        for value in timestamps[position:]
                    )
                    break
                continue
            if not temporary.is_file():
                frames.append(
                    {
                        "start": timestamp,
                        "end": timestamp,
                        "reference": "",
                        "artifact_type": "video_frame",
                        "processing_status": "missing",
                        "reason": "no_decoded_frame",
                    }
                )
                continue
            try:
                with Image.open(temporary) as image:
                    rgb = image.convert("RGB")
                    fingerprint = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    ranking = frame_priority(rgb, previous)
                    matching = next(
                        (
                            index
                            for index, value in enumerate(discovery)
                            if abs(timestamp - value) < 0.002
                        ),
                        None,
                    )
                    if matching is not None:
                        ranking["priority"] += 1
                        if (
                            matching + 1 < len(discovery)
                            and discovery[matching + 1] - discovery[matching] <= 0.75
                        ):
                            ranking["priority"] += 2
                    previous = rgb.copy()
            except (OSError, ValueError):
                continue
            if fingerprint in fingerprints:
                frames.append(
                    {
                        "start": timestamp,
                        "end": timestamp,
                        "reference": "",
                        "artifact_type": "video_frame",
                        "processing_status": "duplicate",
                        "reason": "identical_frame",
                    }
                )
                continue
            payload = temporary.read_bytes()
            reference = "frames/" + hashlib.sha256(payload).hexdigest() + ".jpg"
            (source.parent / reference).write_bytes(payload)
            frames.append(
                {
                    "timestamp": timestamp,
                    "start": timestamp,
                    "end": timestamp,
                    "timestamp_precision": "frame",
                    "reference": reference,
                    "frame_index": position,
                    "artifact_type": "video_frame",
                    "sampling_kind": "transition"
                    if any(abs(timestamp - value) < 0.002 for value in discovery)
                    else "anchor",
                    **ranking,
                }
            )
            fingerprints.add(fingerprint)
    finally:
        temporary.unlink(missing_ok=True)
    return frames


def analyze_video_frames(
    source,
    info,
    *,
    asset_id,
    source_name,
    settings,
    deadline,
    progress=None,
    vlm_mode="disabled",
    transcript_documents=(),
):
    from tools.visual_understanding import analyze_artifacts

    mode = normalize_vlm_mode(vlm_mode)
    if mode == "disabled" or not vlm_available(settings):
        return analyze_artifacts(
            source,
            [],
            asset_id=asset_id,
            source_name=source_name,
            source_type="video",
            vlm_mode=mode,
            settings=settings,
            deadline=deadline,
            progress=progress,
        )
    frames = sample_video_frames(
        source, info, settings=settings, deadline=deadline, progress=progress
    )
    for frame in frames:
        timestamp = frame.get("start", 0)
        nearby = []
        for record in transcript_documents:
            if record.metadata.get("content_type") != "transcript":
                continue
            start, end = record.metadata.get("start"), record.metadata.get("end")
            if (
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and end >= timestamp - 8
                and start <= timestamp + 8
            ):
                nearby.append(record.page_content)
        if nearby:
            frame["transcript_context"] = " ".join(nearby)[:2000]
    documents, metadata = analyze_artifacts(
        source,
        frames,
        asset_id=asset_id,
        source_name=source_name,
        source_type="video",
        vlm_mode=mode,
        settings=settings,
        deadline=deadline,
        progress=progress,
    )
    metadata["warnings"].append(
        "Video visuals use bounded scene-change candidates and timeline samples; brief or subtle content between sampled timestamps may be omitted."
    )
    metadata["scene_transitions"] = [
        frame["start"] for frame in frames if frame.get("sampling_kind") == "transition"
    ]
    return documents, metadata
