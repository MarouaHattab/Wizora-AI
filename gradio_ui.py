from copy import deepcopy
from pathlib import Path
import math
import re

import gradio as gr

from helper.frontend_client import BackendClient, BackendError, get_frontend_settings


SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
                        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv"]
CUSTOM_CSS = """
/* Preserve Wizora-AI's full-width purple workflow in both color modes. */
body { background: #f8fafc !important; }
body.dark { background: #0f1423 !important; }
.gradio-container { width: 100% !important; max-width: none !important; margin: 0 !important;
    --wiz-page: #f8fafc; --wiz-surface: #ffffff; --wiz-text: #1f2937; --wiz-muted: #4b5563;
    --wiz-border: #e5e7eb; --wiz-heading: #312e81; --wiz-subheading: #6b21a8; --wiz-link: #4338ca;
    --wiz-accent-surface: #eef2ff; --wiz-accent-text: #312e81; --wiz-accent-border: #c7d2fe;
    --wiz-success-bg: #f0fdf4; --wiz-success-text: #166534; --wiz-success-border: #86efac;
    --wiz-error-bg: #fef2f2; --wiz-error-text: #991b1b; --wiz-error-border: #fca5a5;
    padding: clamp(16px, 2.2vw, 28px) clamp(10px, 2.55vw, 32px) clamp(28px, 3.5vw, 44px) !important;
    background: var(--wiz-page) !important;
    color: var(--wiz-text) !important; color-scheme: light; }
.dark .gradio-container, .gradio-container.dark {
    --wiz-page: #0f1423; --wiz-surface: #191f31; --wiz-text: #e5e7eb; --wiz-muted: #b9c3d5;
    --wiz-border: #3d455d; --wiz-heading: #ddd6fe; --wiz-subheading: #d8b4fe; --wiz-link: #c4b5fd;
    --wiz-accent-surface: #25213f; --wiz-accent-text: #ddd6fe; --wiz-accent-border: #65588c;
    --wiz-success-bg: #14322a; --wiz-success-text: #bbf7d0; --wiz-success-border: #34755b;
    --wiz-error-bg: #3c202a; --wiz-error-text: #fecaca; --wiz-error-border: #99535e;
    color-scheme: dark; }
.header { text-align: center; margin: 0 auto 22px !important; padding: 8px 16px 18px !important;
    background: transparent; }
.header h1 { margin-bottom: 4px !important; color: var(--wiz-heading); letter-spacing: -.02em; }
.header h3 { color: var(--wiz-subheading); font-weight: 600; }
.gradio-container textarea, .gradio-container input { color: var(--wiz-text) !important;
    background: var(--wiz-surface) !important; }
.gradio-container label span { color: var(--wiz-text) !important; font-weight: 600; }
.gradio-container #workflow-tabs { overflow: hidden; border-radius: 18px;
    background: var(--wiz-surface); box-shadow: 0 8px 28px rgba(76, 29, 149, .10); }
.gradio-container #workflow-tabs .tab-wrapper { height: auto; min-height: 56px; padding: 0; }
.gradio-container #workflow-tabs .tab-container { display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr)); align-items: stretch;
    box-sizing: border-box; height: auto; gap: 6px; padding: 7px !important; }
.gradio-container #workflow-tabs [role="tablist"] {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 0; }
.gradio-container #workflow-tabs .tab-container button { justify-content: center;
    min-width: 0; min-height: 46px; padding: 10px 8px; white-space: normal;
    line-height: 1.3; text-align: center; }
.gradio-container #workflow-tabs [role="tab"] { color: #f8fafc !important;
    border-radius: 11px !important; border: 1px solid transparent !important;
    background: transparent !important; font-weight: 600; }
.gradio-container #workflow-tabs [role="tab"]:hover:not(:disabled),
.gradio-container #workflow-tabs [role="tab"][aria-selected="true"] {
    background: rgba(49, 46, 129, .72) !important; color: #ffffff !important; }
.gradio-container #workflow-tabs [role="tab"]:focus-visible {
    outline: 3px solid #fef08a !important; outline-offset: -5px; }
.gradio-container #workflow-tabs [role="tab"][aria-selected="true"]::after {
    background: #ffffff !important; }
.workflow-panel { padding: 24px 24px 28px !important; }
.workflow-description { margin: 0 0 18px !important; color: var(--wiz-muted); }
.workflow-row { gap: 22px !important; align-items: start; }
.workflow-card { padding: 18px !important; border-radius: 16px;
    border: 1px solid var(--wiz-border); background: var(--wiz-surface); min-width: 0; }
.workflow-results { background: var(--wiz-surface); }
.upload-area { border: 2px dashed #e0e7ff !important; border-radius: 16px !important;
    background: var(--wiz-surface) !important; transition: border-color .2s ease, background .2s ease !important; }
.upload-area:hover, .upload-area:focus-within { border-color: #4f46e5 !important;
    background: var(--wiz-accent-surface) !important; }
.button-primary { background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important; color: #ffffff !important; border-radius: 11px !important;
    padding: 11px 20px !important; font-weight: 700 !important;
    transition: transform .2s ease, box-shadow .2s ease !important; }
.button-primary:hover:not(:disabled) { transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, .28); }
.gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible { outline: 3px solid #a78bfa !important;
    outline-offset: 2px; }
.extracted-text textarea, .workflow-results textarea { line-height: 1.55 !important; }
.extracted-text textarea { font-family: "Courier New", Courier, monospace !important;
    background: var(--wiz-page) !important; max-height: 600px !important; overflow-y: auto !important; }
.status { border: 1px solid var(--wiz-accent-border); background: var(--wiz-accent-surface); color: var(--wiz-accent-text);
    border-radius: 11px; padding: 12px 14px; margin-top: 16px; }
.status-success { border-color: var(--wiz-success-border); background: var(--wiz-success-bg); color: var(--wiz-success-text); }
.status-error { border-color: var(--wiz-error-border); background: var(--wiz-error-bg); color: var(--wiz-error-text); }
.chat-container { max-height: 500px !important; overflow-y: auto !important; background: var(--wiz-surface) !important;
    border: 1px solid var(--wiz-border) !important; border-radius: 14px !important; }
.asset-details, .source-preview { overflow-wrap: anywhere; color: var(--wiz-text); }
.source-preview { max-height: 440px; overflow-y: auto; }
.source-preview a { color: var(--wiz-link) !important; }
@media (max-width: 720px) {
    .header { padding: 18px 14px !important; border-radius: 16px; }
    #workflow-tabs .tab-container { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .workflow-panel { padding: 18px 12px 22px !important; }
    .workflow-row { gap: 12px !important; }
    .workflow-card { padding: 13px !important; }
}
@media (prefers-reduced-motion: reduce) {
    .button-primary, .upload-area { transition: none !important; }
    .button-primary:hover { transform: none; }
}
"""


def empty_state():
    return {
        "asset_id": "", "asset": {}, "extracted_text": "", "asset_details": "", "source_preview": "",
        "question": {}, "summary": {}, "qa": {},
        "upload_status": "📋 Ready to upload a file",
        "question_status": "📋 Generate a question first to enable revision and Accept / Save.",
        "summary_status": "📋 Click Generate Summary first to enable Update Summary and Accept / Save.",
        "qa_status": "📋 Ready to start Q&A session",
    }


def markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", str(value))


def timestamp_text(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
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
    if isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size) and size >= 0:
        facts.append(f"{size / 1024 ** 2:.1f} MB" if size >= 1024 ** 2 else f"{size / 1024:.1f} KB")
    duration = timestamp_text(asset.get("duration_seconds"))
    if duration:
        facts.append("Duration " + duration)
    if asset.get("record_count") is not None:
        facts.append(f"{asset['record_count']} content records")
    if asset.get("frame_count") is not None:
        facts.append(f"{asset['frame_count']} analyzed frames")
    if asset.get("pages"):
        pages = asset["pages"]
        facts.append("Pages " + (", ".join(str(page) for page in pages) if isinstance(pages, list) else str(pages)))
    languages = asset.get("languages") or asset.get("language")
    if languages:
        facts.append("Language " + (", ".join(str(item) for item in languages) if isinstance(languages, list) else str(languages)))
    if facts:
        rows.append(markdown_text(" · ".join(facts)))
    scope = asset.get("processing_scope")
    scopes = {"audio_visual": "Audio transcript and visual analysis", "visual_only": "Visual analysis only",
              "transcript_only": "Transcript only", "text_only": "Text extraction"}
    if scope:
        rows.append("**Processing scope:** " + markdown_text(scopes.get(scope, scope)))
    for field, label in (("audio_status", "Audio"), ("visual_status", "Visual analysis")):
        if asset.get(field):
            rows.append(f"**{label}:** " + markdown_text(str(asset[field]).replace("_", " ")))
    warnings = asset.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    if asset.get("source_type") == "video" and scope == "transcript_only":
        warnings = ["Only spoken audio is available. Slides, on-screen text and visual actions were not analyzed.", *warnings]
    if warnings:
        rows.append("**Processing warnings**\n\n" + "\n".join("- " + markdown_text(item) for item in warnings))
    files = asset.get("files") or []
    if files:
        rows.append("**Available files:** " + ", ".join(markdown_text(item) for item in files))
    return "\n\n".join(rows)


def upload_progress(job):
    labels = {"received": "File received", "processing": "Processing file", "inspecting_media": "Inspecting media",
              "extracting_audio": "Extracting audio", "transcribing": "Transcribing audio",
              "extracting_frames": "Extracting video frames", "analyzing_frames": "Analyzing video visuals",
              "analyzing_visuals": "Analyzing video visuals", "visual_analysis": "Analyzing video visuals",
              "pdf_extracted": "PDF text extracted", "docx_extracted": "DOCX text extracted",
              "text_extracted": "Text extracted", "records_validated": "Content records validated",
              "indexing": "Building source index", "ready": "Ready"}
    rows = []
    for stage in (job.get("stages") or [])[-8:]:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name", "processing"))
        label = labels.get(name, name.replace("_", " ").capitalize())
        if stage.get("segment") is not None:
            label += f" · segment {stage['segment']}" + (f" of {stage['segments']}" if stage.get("segments") else "")
        if stage.get("status") == "completed":
            label += " · complete"
        rows.append("- " + markdown_text(label))
    title = "🔄 File queued for processing…" if job.get("status") == "queued" else "🔄 Processing your file…"
    return title + ("\n\n" + "\n".join(rows) if rows else "")


def create_theme():
    return gr.themes.Soft(primary_hue="purple").set(
        background_fill_primary="#f8fafc", background_fill_primary_dark="#0f1423",
        background_fill_secondary="#ffffff", background_fill_secondary_dark="#191f31",
        body_text_color="#1f2937", body_text_color_dark="#e5e7eb",
        block_background_fill="#ffffff", block_background_fill_dark="#191f31",
    )


class GradioUI:
    def __init__(self, client=None):
        self.client = client or BackendClient()

    @staticmethod
    def _state(value):
        return deepcopy(value) if isinstance(value, dict) else empty_state()

    @staticmethod
    def _review_pending(current):
        return bool(current.get("status") == "interrupted_for_feedback" and current.get("thread_id")
                    and current.get("review_id") and not current.get("busy"))

    @staticmethod
    def _status_update(text):
        status_class = ("status-success" if text.startswith("✅") else
                        "status-error" if text.startswith("❌") else "status")
        return {
            "__type__": "update",
            "value": text,
            "elem_classes": ["status", status_class],
        }

    @staticmethod
    def render(state, *, clear_inputs=False, clear_message=False):
        question, summary, qa = (state.get(key, {}) for key in ("question", "summary", "qa"))
        draft = question.get("result", {})
        history = deepcopy(qa.get("messages", []))
        if qa.get("pending"):
            pending = qa["pending"]
            history.extend([
                {"role": "user", "content": pending["question"]},
                {"role": "assistant", "content": pending.get("text", "")},
            ])
        clear = "" if clear_inputs else gr.skip()
        question_ready = GradioUI._review_pending(question)
        summary_ready = GradioUI._review_pending(summary)
        return (
            state, *([state.get("asset_id", "")] * 4),
            GradioUI._status_update(state["upload_status"]), state.get("extracted_text", ""),
            draft.get("question", ""), "\n".join(str(item) for item in (draft.get("options") or [])),
            str(draft.get("answer", "")), draft.get("explanation", ""),
            GradioUI._status_update(state["question_status"]),
            summary.get("preview_outline", summary.get("outline", "")),
            summary.get("preview_summary", summary.get("text", "")),
            GradioUI._status_update(state["summary_status"]),
            history, GradioUI._status_update(state["qa_status"]), clear, clear, clear,
            "" if clear_inputs or clear_message else gr.skip(),
            *[gr.update(interactive=question_ready) for _ in range(3)],
            *[gr.update(interactive=summary_ready) for _ in range(2)],
            state.get("asset_details", ""), state.get("source_preview", ""),
        )

    def select_asset(self, asset_id, value):
        state = self._state(value)
        selected = (asset_id or "").strip()
        if selected == state.get("asset_id"):
            return self.render(state)
        state = empty_state()
        state["asset_id"] = selected
        state["upload_status"] = "📋 Asset changed. Load extracted text to check this file." if selected else "📋 Ready to upload a file"
        return self.render(state, clear_inputs=True)

    async def _ready(self, asset_id, state):
        selected = (asset_id or "").strip()
        if not selected:
            raise BackendError("Upload a file or enter its Asset ID first", 422)
        if selected != state.get("asset_id"):
            state.clear()
            state.update(empty_state(), asset_id=selected)
        asset = await self.client.get_asset(selected)
        if asset.get("id") != selected or asset.get("status") != "ready":
            raise BackendError("The selected file is not ready. Upload it again or wait for processing", 409)
        state["asset"] = asset
        state["asset_details"] = asset_details(asset)
        return selected

    async def _load_content(self, asset_id, state):
        extracted = await self.client.extracted_text(asset_id)
        text = extracted.get("extracted_text", "")
        if not isinstance(text, str) or not text.strip():
            raise BackendError("The file was stored but no extracted content was returned", 502)
        state["extracted_text"] = text
        state["source_preview"] = ""
        state.pop("content_warning", None)
        if not hasattr(self.client, "get_content"):
            return
        try:
            content = await self.client.get_content(asset_id)
        except BackendError as error:
            if error.status_code in {404, 405}:
                return  # Older backends expose the combined text only.
            state["content_warning"] = "Source metadata could not be loaded. " + markdown_text(error.detail)
            state["source_preview"] = state["content_warning"]
            return
        documents = content.get("documents")
        if content.get("asset_id", asset_id) != asset_id or not isinstance(documents, list):
            raise BackendError("The backend returned invalid source records", 502)
        original = next((item for item in state["asset"].get("files", [])
                         if isinstance(item, str) and item.startswith("original.")), None)
        rows = []
        for index, document in enumerate(documents[:8], 1):
            if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
                continue
            metadata = document["metadata"]
            labels = [str(metadata.get("content_type") or "Text").replace("_", " ").capitalize()]
            for key, label in (("page", "Page"), ("section", "Section"), ("paragraph", "Paragraph")):
                if metadata.get(key) is not None:
                    labels.append(f"{label} {metadata[key]}")
            start = metadata.get("start", metadata.get("timestamp"))
            at = timestamp_text(start)
            if at:
                end = timestamp_text(metadata.get("end"))
                labels.append(at + ("–" + end if end else ""))
            if metadata.get("timestamp_precision") == "audio_window":
                labels.append("audio window; approximate timing")
            if metadata.get("language"):
                labels.append(str(metadata["language"]))
            title = markdown_text(" · ".join(labels))
            if original:
                try:
                    url = self.client.source_url(asset_id, original, page=metadata.get("page"), timestamp=start if at else None)
                    title = f"[{title}]({url})"
                except BackendError:
                    pass
            excerpt = str(document.get("page_content", "")).strip()
            rows.append(f"**{index}. {title}**\n\n" + markdown_text(excerpt[:700]) + ("…" if len(excerpt) > 700 else ""))
        if len(documents) > 8:
            rows.append(f"Showing 8 of {len(documents)} source records. The combined content above includes all records.")
        state["source_preview"] = "\n\n---\n\n".join(rows)

    async def upload_file(self, file, value):
        state = self._state(value)
        try:
            if not file:
                raise BackendError("Please select a file to upload", 422)
            path = Path(file if isinstance(file, str) else file.name)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise BackendError("Unsupported file format. Choose TXT, PDF, DOCX, audio or video from the supported formats listed above", 415)
            state = empty_state()
            state["upload_status"] = "🔄 Uploading " + markdown_text(path.name) + "…"
            yield self.render(state, clear_inputs=True)
            if hasattr(self.client, "upload_events"):
                async for result in self.client.upload_events(path):
                    asset_id = result.get("id")
                    if not isinstance(asset_id, str) or not asset_id:
                        raise BackendError("The backend did not return a valid Asset ID", 502)
                    state.update(asset_id=asset_id, asset=result, asset_details=asset_details(result),
                                 upload_status=upload_progress(result))
                    yield self.render(state)
            else:
                result = await self.client.upload(path)
            asset_id = result.get("id")
            if not isinstance(asset_id, str) or not asset_id:
                raise BackendError("The backend did not return a valid Asset ID", 502)
            state["asset_id"] = asset_id
            await self._ready(asset_id, state)
            await self._load_content(asset_id, state)
            state["upload_status"] = f"✅ File processed successfully. Ready for questions, summaries and Q&A.\n\nAsset ID: `{asset_id}`"
            if state.get("content_warning"):
                state["upload_status"] += "\n\n" + state["content_warning"]
        except BackendError as error:
            state["upload_status"] = "❌ " + error.detail
        except Exception:
            state["upload_status"] = "❌ The file could not be uploaded. Check the backend logs and try again."
        yield self.render(state, clear_inputs=True)

    async def load_text(self, asset_id, value):
        state = self._state(value)
        try:
            selected = await self._ready(asset_id, state)
            await self._load_content(selected, state)
            state["upload_status"] = "✅ Extracted content loaded. This Asset ID is selected in all four tabs."
            if state.get("content_warning"):
                state["upload_status"] += "\n\n" + state["content_warning"]
        except BackendError as error:
            state["extracted_text"] = ""
            state["upload_status"] = "❌ " + error.detail
        except Exception:
            state["upload_status"] = "❌ Extracted text could not be loaded. Check the backend logs."
        yield self.render(state)

    @staticmethod
    def _review(current, workflow):
        if not GradioUI._review_pending(current):
            label = "Generate " + workflow
            if current.get("error"):
                message = f"The previous {workflow.lower()} attempt failed: {current['error']}. Click {label} to try again before updating or saving."
            elif current.get("status") == "completed":
                message = f"This {workflow.lower()} is already saved. Click {label} to create another result."
            else:
                message = f"Click {label} first. Revision and Accept / Save become available after a {workflow.lower()} is generated."
            raise BackendError(message, 409)

    @staticmethod
    def _question_result(response):
        status = response.get("status")
        if status not in {"interrupted_for_feedback", "completed"} or not response.get("thread_id"):
            raise BackendError("Question generation did not finish successfully", 502)
        if status == "interrupted_for_feedback" and not response.get("review_id"):
            raise BackendError("The backend did not identify the question review", 502)
        result = response.get("current_state") or response.get("final_state") or {}
        feedback = response.get("data_for_feedback") or {}
        result = {**result, **{key: feedback[key] for key in ("options", "answer", "explanation") if key in feedback}}
        if "generated_question" in feedback:
            result["question"] = feedback["generated_question"]
        if not all(isinstance(result.get(key), str) and result[key].strip() for key in ("question", "answer", "explanation")):
            raise BackendError("The backend returned an incomplete question", 502)
        return {"thread_id": response["thread_id"], "review_id": response.get("review_id"),
                "status": status, "result": result, "notice": feedback.get("revision_notice") or result.get("review_notice", "")}

    async def _question(self, value, *, asset_id=None, question_type="MCQ", feedback=None):
        state = self._state(value)
        current = state.get("question", {})
        try:
            if feedback is None:
                selected = await self._ready(asset_id, state)
                current = {}
                state["question"] = current
            else:
                self._review(current, "Question")
                if not feedback.strip():
                    raise BackendError("Please provide feedback for the question", 422)
            state["question_status"] = "🔄 Generating question…" if feedback is None else "🔄 Updating question…"
            current["busy"] = True
            yield self.render(state)
            response = (await self.client.start_question(selected, question_type) if feedback is None else
                        await self.client.feedback_question(current["thread_id"], feedback.strip(), current["review_id"]))
            result = response.get("current_state") or response.get("final_state") or {}
            if result.get("asset_id", state["asset_id"]) != state["asset_id"]:
                raise BackendError("The backend returned a question for a different file", 502)
            current = self._question_result(response)
            state["question"] = current
            state["question_status"] = ("✅ Question accepted and saved." if current["status"] == "completed" else
                                        "✅ Question ready. Provide feedback, refine automatically, or accept/save.")
            if current.get("notice"):
                state["question_status"] += "\n\n" + markdown_text(current["notice"])
        except BackendError as error:
            if feedback is None:
                current = state.setdefault("question", {})
            state["question_status"] = "❌ " + error.detail
            if feedback is not None and current.get("busy"):
                current["status"] = "uncertain"
                current["review_id"] = None
                current["error"] = error.detail
                state["question_status"] += " Generate a new question before submitting more feedback."
            elif feedback is None:
                current.update(status="failed", review_id=None, error=error.detail)
        except Exception:
            current = state.setdefault("question", {})
            current.update(status="failed", review_id=None, error="Question generation failed. Check the backend logs")
            state["question_status"] = "❌ Question generation failed. Check the backend logs and generate a new question."
        finally:
            state.get("question", {}).pop("busy", None)
        yield self.render(state)

    async def start_question(self, asset_id, question_type, value):
        async for output in self._question(value, asset_id=asset_id, question_type=question_type):
            yield output

    async def update_question(self, feedback, value):
        async for output in self._question(value, feedback=feedback or ""):
            yield output

    async def refine_question(self, value):
        async for output in self._question(value, feedback="auto"):
            yield output

    async def accept_question(self, value):
        async for output in self._question(value, feedback="save"):
            yield output

    async def _summary(self, value, *, asset_id=None, feedback=None):
        state = self._state(value)
        current = state.get("summary", {})
        try:
            if feedback is None:
                selected = await self._ready(asset_id, state)
                current = {"outline": "", "text": ""}
                state["summary"] = current
                events = self.client.summary_events(asset_id=selected)
            else:
                self._review(current, "Summary")
                if not feedback.strip():
                    raise BackendError("Please provide feedback for the summary", 422)
                events = self.client.summary_events(thread_id=current["thread_id"], feedback=feedback.strip(), review_id=current["review_id"])
            current["status"] = "running"
            current["busy"] = True
            state["summary_status"] = "🔄 Generating summary…" if feedback is None else "🔄 Updating summary…"
            yield self.render(state)
            async for event in events:
                current["thread_id"] = event["thread_id"]
                kind, data, stage = event["event"], event["data"], event.get("stage")
                if kind == "stage_started":
                    if stage == "main_point_summarizer":
                        current["preview_outline"] = ""
                    elif stage in {"summarizer_writer", "summarizer_rewriter"}:
                        current["preview_summary"] = ""
                    state["summary_status"] = "🔄 " + {
                        "main_point_summarizer": "Generating main points…", "summarizer_writer": "Writing detailed summary…",
                        "summarizer_structure": "Revising outline and sections…", "summarizer_rewriter": "Revising summary…",
                    }.get(stage, "Finishing summary…")
                elif kind == "text_delta":
                    key = "preview_outline" if stage == "main_point_summarizer" else "preview_summary"
                    current[key] = current.get(key, "") + data.get("text", "")
                elif kind == "awaiting_feedback":
                    if not data.get("review_id") or not isinstance(data.get("summary"), str) or not data["summary"].strip():
                        raise BackendError("The backend returned an incomplete summary review", 502)
                    current.update(status="interrupted_for_feedback", review_id=data["review_id"],
                                   outline=data.get("table_of_contents", ""), text=data["summary"], notice=data.get("revision_notice", ""))
                    current.pop("preview_outline", None)
                    current.pop("preview_summary", None)
                    state["summary_status"] = "🔄 Finishing summary…"
                elif kind == "completed":
                    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
                        raise BackendError("The backend returned an empty summary", 502)
                    current.update(status="completed", review_id=None, text=data["summary"])
                    current.pop("preview_summary", None)
                    state["summary_status"] = "🔄 Saving summary…"
                elif kind == "failed":
                    raise BackendError(data.get("detail", "Summary generation failed"), 502)
                yield self.render(state)
            state["summary_status"] = ("✅ Summary accepted and saved." if current.get("status") == "completed" else
                                       "✅ Summary ready. Provide feedback to revise wording or structure, or accept/save.")
            if current.get("notice"):
                state["summary_status"] += "\n\n" + markdown_text(current["notice"])
        except BackendError as error:
            if feedback is None:
                current = state.setdefault("summary", {})
            if current.get("status") == "running" or error.status_code not in {400, 404, 409, 422}:
                current.update(status="failed", review_id=None, error=error.detail)
            elif feedback is None:
                current.update(status="failed", review_id=None, error=error.detail)
            current.pop("preview_outline", None)
            current.pop("preview_summary", None)
            state["summary_status"] = "❌ " + error.detail
            if current.get("status") == "failed" and "Generate Summary" not in error.detail:
                state["summary_status"] += " Click Generate Summary to try again. Update Summary and Accept / Save require a successfully generated summary."
        except Exception:
            current = state.setdefault("summary", {})
            current.update(status="failed", review_id=None, error="Summary generation failed. Check the backend logs")
            current.pop("preview_outline", None)
            current.pop("preview_summary", None)
            state["summary_status"] = "❌ Summary generation failed. Check the backend logs, then click Generate Summary to try again."
        finally:
            state.get("summary", {}).pop("busy", None)
        yield self.render(state)

    async def start_summary(self, asset_id, value):
        async for output in self._summary(value, asset_id=asset_id):
            yield output

    async def update_summary(self, feedback, value):
        async for output in self._summary(value, feedback=feedback or ""):
            yield output

    async def accept_summary(self, value):
        async for output in self._summary(value, feedback="save"):
            yield output

    def _citations(self, citations, state):
        rows = []
        asset = state.get("asset", {})
        original = next((item for item in asset.get("files", []) if isinstance(item, str) and item.startswith("original.")), None)
        for citation in citations:
            if not isinstance(citation, dict) or citation.get("asset_id", state["asset_id"]) != state["asset_id"]:
                continue
            label = f"[{citation.get('number', len(rows) + 1)}] {citation.get('source', asset.get('source_name', 'Source'))}"
            places = []
            for key, name in (("page", "Page"), ("section", "Section"), ("paragraph", "Paragraph"), ("table_index", "Table")):
                if citation.get(key) is not None:
                    places.append(f"{name} {citation[key]}")
            if citation.get("start") is not None:
                places.append(f"{citation['start']:g}s" if isinstance(citation["start"], (int, float)) else str(citation["start"]))
            if citation.get("source_type") == "video":
                places.append("Video visual" if citation.get("content_type") == "visual_description" else "Video transcript")
            title = markdown_text(label)
            if original:
                try:
                    url = self.client.source_url(state["asset_id"], original, page=citation.get("page"), timestamp=citation.get("start"))
                    title = f"[{title}]({url})"
                except BackendError:
                    pass
            rows.append("- " + title + (" — " + markdown_text(" · ".join(places)) if places else ""))
        return "\n\n**Sources**\n\n" + "\n".join(rows) if rows else ""

    def _cited_answer(self, text, citations, state):
        body, marker, appendix = text.rpartition("\n\nSources:\n")
        lines = appendix.splitlines()
        expected = [f"[{item['number']}] " for item in citations if isinstance(item, dict) and "number" in item]
        if marker and expected and len(lines) == len(expected) and all(line.startswith(prefix) for line, prefix in zip(lines, expected)):
            original = next((item for item in state.get("asset", {}).get("files", [])
                             if isinstance(item, str) and item.startswith("original.")), None)
            rows = []
            for line, citation, prefix in zip(lines, citations, expected):
                label = markdown_text(prefix.strip())
                if original:
                    try:
                        url = self.client.source_url(state["asset_id"], original, page=citation.get("page"), timestamp=citation.get("start"))
                        label = f"[{label}]({url})"
                    except BackendError:
                        pass
                rows.append("- " + label + " " + line[len(prefix):])
            return body + "\n\n**Sources**\n\n" + "\n".join(rows)
        return text + self._citations(citations, state)

    async def _chat(self, question, value, *, asset_id=None, start=False):
        state = self._state(value)
        current = state.get("qa", {})
        success = False
        try:
            question = (question or "").strip()
            if not question:
                raise BackendError("Please enter a question about the selected file", 422)
            if start:
                await self._ready(asset_id, state)
                current = {"messages": [], "citations": []}
                state["qa"] = current
            elif current.get("status") != "completed" or not current.get("thread_id"):
                raise BackendError("Start a Q&A session before sending a follow-up question", 409)
            current.update(status="running", pending={"question": question, "text": ""})
            state["qa_status"] = "🔄 Generating a source-grounded answer…"
            yield self.render(state)
            completed = None
            async for event in self.client.qa_events(asset_id=state["asset_id"], question=question,
                                                     thread_id=None if start else current["thread_id"]):
                current["thread_id"] = event["thread_id"]
                kind, data = event["event"], event["data"]
                if kind == "text_delta":
                    current["pending"]["text"] += data.get("text", "")
                elif kind == "completed":
                    completed = data
                    state["qa_status"] = "🔄 Finishing answer…"
                elif kind == "failed":
                    raise BackendError(data.get("detail", "Question answering failed"), 502)
                elif kind == "awaiting_feedback":
                    raise BackendError("The answer did not complete. Start a new Q&A session", 502)
                yield self.render(state)
            if completed is None or not current["pending"]["text"].strip():
                raise BackendError("The backend did not complete an answer", 502)
            pending = current["pending"]
            citations = completed.get("citations") or []
            if not isinstance(citations, list) or any(not isinstance(item, dict) or item.get("asset_id", state["asset_id"]) != state["asset_id"] for item in citations):
                raise BackendError("The answer returned invalid source citations", 502)
            current["messages"].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": self._cited_answer(pending["text"], citations, state)},
            ])
            current.pop("pending")
            current.setdefault("citations", []).append(citations)
            current.update(status="completed", answer_status=completed.get("answer_status"))
            state["qa_status"] = "✅ Response complete. Ask a follow-up question."
            if completed.get("answer_status") in {"insufficient_evidence", "unsupported", "insufficient"}:
                state["qa_status"] = "📋 This file does not contain enough evidence to answer the question."
            elif completed.get("answer_status") == "off_topic":
                state["qa_status"] = "📋 This question is outside the selected file."
            success = True
        except BackendError as error:
            if current.get("pending"):
                current.pop("pending", None)
                current["status"] = "failed"
            state["qa_status"] = "❌ " + error.detail
        except Exception:
            current.pop("pending", None)
            current["status"] = "failed"
            state["qa_status"] = "❌ The answer could not be completed. Check the backend logs and start a new Q&A session."
        yield self.render(state, clear_message=success)

    async def start_qa(self, asset_id, question, value):
        async for output in self._chat(question, value, asset_id=asset_id, start=True):
            yield output

    async def continue_qa(self, message, value):
        async for output in self._chat(message, value):
            yield output


def create_gradio_interface(client=None):
    ui = GradioUI(client)
    with gr.Blocks(title="Wizora-AI — AI Teaching Assistant", analytics_enabled=False) as app:
        state = gr.State(empty_state)
        with gr.Column():
            gr.Markdown("# 🎓 Wizora-AI\n### AI Teaching Assistant\nUpload content, generate questions, create summaries and chat with your materials.", elem_classes="header")
            with gr.Tabs(elem_id="workflow-tabs"):
                with gr.Tab("📁 Upload Content", elem_id="upload-tab", elem_classes="workflow-panel"):
                    gr.Markdown("### Upload text, documents, audio or video for processing\n"
                                "TXT · PDF · DOCX · MP3 · WAV · M4A · FLAC · OGG · AAC · MP4 · MOV · MKV · WEBM · AVI · FLV",
                                elem_classes="workflow-description")
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(scale=1, elem_classes=["workflow-card", "workflow-controls"]):
                            file_input = gr.File(label="Select File", file_count="single", type="filepath", file_types=SUPPORTED_EXTENSIONS, elem_classes="upload-area")
                            upload_btn = gr.Button("🚀 Upload File", variant="primary", elem_classes="button-primary")
                            asset_id = gr.Textbox(label="Generated Asset ID", placeholder="Asset ID will appear here after upload…")
                            load_btn = gr.Button("📄 Load Extracted Text")
                            upload_status = gr.Markdown(empty_state()["upload_status"], elem_classes="status")
                            details = gr.Markdown("", elem_classes="asset-details")
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            extracted = gr.Textbox(label="Extracted Content", placeholder="Extracted text, transcript and available visual descriptions will appear here after processing…", lines=15, max_lines=20, interactive=False, buttons=["copy"], elem_classes="extracted-text")
                            with gr.Accordion("Source records and timestamps", open=False):
                                preview = gr.Markdown("", elem_classes="source-preview")
                with gr.Tab("❓ Question Generation", elem_id="question-tab", elem_classes="workflow-panel"):
                    gr.Markdown("### Generate questions from your uploaded content", elem_classes="workflow-description")
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(scale=1, elem_classes=["workflow-card", "workflow-controls"]):
                            q_asset = gr.Textbox(label="Asset ID", placeholder="Populated automatically after upload…")
                            question_type = gr.Radio(choices=["T/F", "MCQ"], label="Question Type", value="MCQ")
                            generate_q = gr.Button("🎯 Generate Question", variant="primary", elem_classes="button-primary")
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            question = gr.Textbox(label="Generated Question", lines=3, interactive=False)
                            options = gr.Textbox(label="Answer Options", lines=4, interactive=False)
                            answer = gr.Textbox(label="Correct Answer", lines=1, interactive=False)
                            explanation = gr.Textbox(label="Explanation", lines=3, interactive=False)
                    with gr.Row():
                        q_feedback = gr.Textbox(label="Feedback", placeholder="Provide feedback to improve the question…", lines=2, scale=3)
                        update_q = gr.Button("🔄 Update Question", scale=1, interactive=False)
                    with gr.Row():
                        refine_q = gr.Button("✨ Automatically Refine", interactive=False)
                        accept_q = gr.Button("✅ Accept / Save Question", interactive=False)
                    q_status = gr.Markdown(empty_state()["question_status"], elem_classes="status")
                with gr.Tab("📝 Content Summarization", elem_id="summary-tab", elem_classes="workflow-panel"):
                    gr.Markdown("### Create summaries and main points from your content", elem_classes="workflow-description")
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(scale=1, elem_classes=["workflow-card", "workflow-controls"]):
                            s_asset = gr.Textbox(label="Asset ID", placeholder="Populated automatically after upload…")
                            generate_s = gr.Button("📊 Generate Summary", variant="primary", elem_classes="button-primary")
                            s_feedback = gr.Textbox(label="Feedback", placeholder="Describe wording changes or how to reorganize the outline and sections…", lines=3)
                            update_s = gr.Button("🔄 Update Summary", interactive=False)
                            accept_s = gr.Button("✅ Accept / Save Summary", interactive=False)
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            main_points = gr.Textbox(label="Main Points", lines=6, interactive=False, buttons=["copy"])
                            summary = gr.Textbox(label="Detailed Summary", lines=8, interactive=False, buttons=["copy"])
                    s_status = gr.Markdown(empty_state()["summary_status"], elem_classes="status")
                with gr.Tab("💬 Question Answering", elem_id="qa-tab", elem_classes="workflow-panel"):
                    gr.Markdown("### Ask questions about your uploaded content", elem_classes="workflow-description")
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(scale=1, elem_classes=["workflow-card", "workflow-controls"]):
                            qa_asset = gr.Textbox(label="Asset ID", placeholder="Populated automatically after upload…")
                            initial_question = gr.Textbox(label="Initial Question", placeholder="Ask your first question about the content…", lines=2)
                            start_qa = gr.Button("🚀 Start Q&A Session", variant="primary", elem_classes="button-primary")
                            qa_status = gr.Markdown(empty_state()["qa_status"], elem_classes="status")
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            chatbot = gr.Chatbot(label="Conversation", height=400, elem_classes="chat-container", buttons=["copy_all"], placeholder="Answers and source citations will appear here.")
                            with gr.Row():
                                message = gr.Textbox(label="Your Message", placeholder="Continue the conversation…", lines=1, scale=4)
                                send = gr.Button("📤 Send", scale=1)
            gr.Markdown("---\n### 💡 How to Use\n1. **Upload:** Select one text, PDF, DOCX, audio or video file. Follow extraction, transcription and visual analysis progress. Its Asset ID appears in every tab.\n2. **Review:** Check the processing scope, warnings and source timestamps before using the content.\n3. **Questions:** Generate a question, provide feedback, refine automatically, then accept/save.\n4. **Summary:** Create main points and a detailed summary, revise, then accept/save.\n5. **Chat:** Start a conversation and ask follow-up questions about the selected file.")
        outputs = [state, asset_id, q_asset, s_asset, qa_asset, upload_status, extracted, question, options, answer,
                   explanation, q_status, main_points, summary, s_status, chatbot, qa_status,
                   q_feedback, s_feedback, initial_question, message,
                   update_q, refine_q, accept_q, update_s, accept_s, details, preview]
        event_options = dict(outputs=outputs, concurrency_limit=1, concurrency_id="single-file-workflow", trigger_mode="once", show_progress="minimal")
        upload_btn.click(ui.upload_file, [file_input, state], **event_options)
        load_btn.click(ui.load_text, [asset_id, state], **event_options)
        for field in (asset_id, q_asset, s_asset, qa_asset):
            field.input(ui.select_asset, [field, state], **event_options)
        generate_q.click(ui.start_question, [q_asset, question_type, state], **event_options)
        update_q.click(ui.update_question, [q_feedback, state], **event_options)
        refine_q.click(ui.refine_question, [state], **event_options)
        accept_q.click(ui.accept_question, [state], **event_options)
        generate_s.click(ui.start_summary, [s_asset, state], **event_options)
        update_s.click(ui.update_summary, [s_feedback, state], **event_options)
        accept_s.click(ui.accept_summary, [state], **event_options)
        start_qa.click(ui.start_qa, [qa_asset, initial_question, state], **event_options)
        gr.on([send.click, message.submit], ui.continue_qa, [message, state], **event_options)
    return app.queue(default_concurrency_limit=1)


def launch():
    settings = get_frontend_settings()
    create_gradio_interface().launch(
        server_name=settings.FRONTEND_HOST, server_port=settings.FRONTEND_PORT,
        share=False, show_error=False, theme=create_theme(), css=CUSTOM_CSS,
        max_file_size=settings.MAX_UPLOAD_BYTES, footer_links=[],
    )


if __name__ == "__main__":
    launch()
