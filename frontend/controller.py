import json
import logging
from copy import deepcopy
from pathlib import Path

import gradio as gr

from frontend.client import BackendClient, BackendError
from frontend.constants import (
    SUPPORTED_EXTENSIONS,
    VISUAL_EXTENSIONS,
    VLM_CHOICES,
    VLM_DESCRIPTIONS,
)
from frontend.multifile import MultifileUI, file_paths
from frontend.presentation import asset_details, markdown_text, timestamp_text, upload_progress
from frontend.state import empty_state

logger = logging.getLogger("gradio_ui")


class GradioUI(MultifileUI):
    def __init__(self, client=None):
        self.client = client or BackendClient()

    async def configure_vlm(self, file, value):
        """Configure controls from backend capabilities without starting extraction."""
        preference = deepcopy(value) if isinstance(value, dict) else {}
        try:
            configuration = await self.client.upload_configuration()
            modes = configuration.get("vlm_modes", [])
            modes = (
                [mode for mode in modes if mode in VLM_DESCRIPTIONS]
                if isinstance(modes, list)
                else []
            )
            available = configuration.get("vlm_available") is True and any(
                mode != "disabled" for mode in modes
            )
            reason = configuration.get("vlm_unavailable_reason") or "No VLM provider is configured."
        except BackendError as error:
            configuration, modes, available = {}, [], False
            reason = "Backend capabilities could not be verified. " + error.detail
        except Exception:
            configuration, modes, available = {}, [], False
            reason = "Backend capabilities could not be verified. Check the backend connection."
        choices = [
            choice
            for choice in VLM_CHOICES
            if choice[1] == "disabled" or (available and choice[1] in modes)
        ]
        allowed = [choice[1] for choice in choices]
        preferred = preference.get("selected")
        if preferred not in VLM_DESCRIPTIONS:
            default = (
                configuration.get("vlm_default_mode", "automatic") if available else "disabled"
            )
            preferred = default if default in allowed else "disabled"
        paths = file_paths(file)
        visual_file = not paths or any(
            Path(path).suffix.lower() in VISUAL_EXTENSIONS for path in paths
        )
        enabled = available and visual_file
        selected = preferred if enabled and preferred in allowed else "disabled"
        preference.update(
            selected=preferred,
            available=available,
            allowed=allowed,
            visual_file=visual_file,
            vision_model=configuration.get("vision_model"),
            vision_provider=configuration.get("vision_provider"),
        )
        description = VLM_DESCRIPTIONS[selected]
        if not visual_file:
            description = (
                "Visual analysis is unnecessary for TXT and audio-only files. "
                + VLM_DESCRIPTIONS["disabled"]
            )
        if not available:
            description += "\n\nVisual Understanding is unavailable: " + markdown_text(reason)
        if configuration.get("vision_model"):
            description += "\n\nConfigured vision model: " + markdown_text(
                f"{configuration.get('vision_provider', '')} / {configuration['vision_model']}"
            )
        return (
            gr.update(choices=choices, value=selected, interactive=enabled, visible=True),
            description,
            preference,
        )

    @staticmethod
    def select_vlm_mode(vlm_mode, value):
        preference = deepcopy(value) if isinstance(value, dict) else {}
        if (
            preference.get("available")
            and preference.get("visual_file", True)
            and vlm_mode in preference.get("allowed", VLM_DESCRIPTIONS)
        ):
            preference["selected"] = vlm_mode
        return VLM_DESCRIPTIONS.get(vlm_mode, VLM_DESCRIPTIONS["disabled"]), preference

    @staticmethod
    def apply_vlm_mode(vlm_mode, value):
        """Restore a queued user choice after a capability refresh updates the control."""
        preference = deepcopy(value) if isinstance(value, dict) else {}
        enabled = preference.get("available") and preference.get("visual_file", True)
        selected = vlm_mode if enabled and vlm_mode in preference.get("allowed", []) else "disabled"
        description, preference = GradioUI.select_vlm_mode(selected, preference)
        if not preference.get("visual_file", True):
            description = (
                "Visual analysis is unnecessary for TXT and audio-only files. "
                + VLM_DESCRIPTIONS["disabled"]
            )
        elif not preference.get("available"):
            description += (
                "\n\nVisual Understanding is unavailable. Text, tables and audio still work."
            )
        if preference.get("vision_model"):
            description += "\n\nConfigured vision model: " + markdown_text(
                f"{preference.get('vision_provider', '')} / {preference['vision_model']}"
            )
        return gr.update(value=selected), description, preference

    @staticmethod
    def _state(value):
        return deepcopy(value) if isinstance(value, dict) else empty_state()

    @staticmethod
    def _review_pending(current):
        return bool(
            current.get("status") == "interrupted_for_feedback"
            and current.get("thread_id")
            and current.get("review_id")
            and not current.get("busy")
        )

    @staticmethod
    def _status_update(text):
        status_class = (
            "status-success"
            if text.startswith("✅")
            else "status-error"
            if text.startswith("❌")
            else "status"
        )
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
            history.extend(
                [
                    {"role": "user", "content": pending["question"]},
                    {"role": "assistant", "content": pending.get("text", "")},
                ]
            )
        clear = "" if clear_inputs else gr.skip()
        question_ready = GradioUI._review_pending(question)
        summary_ready = GradioUI._review_pending(summary)
        question_status = state["question_status"]
        validation = draft.get("validation") or {}
        if draft.get("review_required") or validation.get("status") == "review_required":
            question_status += "\n\n⚠️ Selected question requires review. Verify its source evidence, answer and difficulty before saving."
            warnings = validation.get("warnings") or []
            if warnings:
                question_status += "\n\n" + "\n".join(
                    "- " + markdown_text(str(item)) for item in warnings
                )
        elif validation.get("status") == "validated":
            question_status += (
                "\n\nSource review passed the available checks. Human review is still recommended."
            )
        return (
            state,
            *([state.get("asset_id", "")] * 4),
            GradioUI._status_update(state["upload_status"]),
            state.get("extracted_text", ""),
            draft.get("question", ""),
            "\n".join(str(item) for item in (draft.get("options") or [])),
            str(draft.get("answer", "")),
            draft.get("explanation", ""),
            GradioUI._status_update(question_status),
            summary.get("preview_outline", summary.get("outline", "")),
            summary.get("preview_summary", summary.get("text", "")),
            GradioUI._status_update(state["summary_status"]),
            history,
            GradioUI._status_update(state["qa_status"]),
            clear,
            clear,
            clear,
            "" if clear_inputs or clear_message else gr.skip(),
            *[gr.update(interactive=question_ready) for _ in range(3)],
            *[gr.update(interactive=summary_ready) for _ in range(2)],
            state.get("asset_details", ""),
            state.get("source_preview", ""),
        )

    def select_asset(self, asset_id, value):
        state = self._state(value)
        selected = (asset_id or "").strip()
        if selected == state.get("asset_id"):
            return self.render(state)
        batch = state.get("batch")
        state = empty_state()
        if batch:
            state["batch"] = batch
        state["asset_id"] = selected
        state["upload_status"] = (
            "📋 Asset changed. Load extracted text to check this file."
            if selected
            else "📋 Ready to upload a file"
        )
        return self.render(state, clear_inputs=True)

    async def _ready(self, asset_id, state):
        selected = (asset_id or "").strip()
        if not selected:
            raise BackendError("Upload a file or enter its Asset ID first", 422)
        if selected != state.get("asset_id"):
            batch = state.get("batch")
            state.clear()
            state.update(empty_state(), asset_id=selected)
            if batch:
                state["batch"] = batch
        asset = await self.client.get_asset(selected)
        if asset.get("id") != selected or asset.get("status") != "ready":
            raise BackendError(
                "The selected file is not ready. Upload it again or wait for processing", 409
            )
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
            state["content_warning"] = "Source metadata could not be loaded. " + markdown_text(
                error.detail
            )
            state["source_preview"] = state["content_warning"]
            return
        documents = content.get("documents")
        if content.get("asset_id", asset_id) != asset_id or not isinstance(documents, list):
            raise BackendError("The backend returned invalid source records", 502)
        rows = []
        images = [
            reference
            for reference in state.get("asset", {}).get("files", [])
            if isinstance(reference, str)
            and reference.startswith(("images/", "frames/"))
            and reference.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        for reference in images[:4]:
            try:
                url = self.client.source_url(asset_id, reference)
                rows.append(f"![Extracted source visual](<{url}>)")
            except BackendError:
                pass
        if len(images) > 4:
            rows.append(
                f"Showing 4 of {len(images)} extracted images. Original references remain available in source records and Job Results."
            )
        for index, document in enumerate(documents[:8], 1):
            if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
                continue
            metadata = document["metadata"]
            labels = [str(metadata.get("content_type") or "Text").replace("_", " ").capitalize()]
            if metadata.get("source"):
                labels.append(str(metadata["source"]))
            for key, label in (
                ("page", "Page"),
                ("section", "Section"),
                ("paragraph", "Paragraph"),
            ):
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
            url = self._source_link(metadata, state)
            if url:
                title = f"[{title}]({url})"
            excerpt = str(document.get("page_content", "")).strip()
            rows.append(
                f"**{index}. {title}**\n\n"
                + markdown_text(excerpt[:700])
                + ("…" if len(excerpt) > 700 else "")
            )
        if len(documents) > 8:
            rows.append(
                f"Showing 8 of {len(documents)} source records. The combined content above includes all records."
            )
        state["source_preview"] = "\n\n---\n\n".join(rows)

    def _source_link(self, locator, state):
        files = state.get("asset", {}).get("files", [])
        source_id = locator.get("source_file_id")
        prefix = f"sources/{source_id}/original." if source_id else "original."
        reference = next(
            (item for item in files if isinstance(item, str) and item.startswith(prefix)), None
        )
        if not reference:
            reference = locator.get("artifact_reference") or locator.get("frame_reference")
        if reference:
            try:
                return self.client.source_url(
                    state["asset_id"],
                    reference,
                    page=locator.get("page"),
                    timestamp=locator.get("start"),
                )
            except BackendError:
                pass
        return None

    async def upload_file(self, file, value, vlm_mode="disabled"):
        state = self._state(value)
        try:
            if not file:
                raise BackendError("Please select a file to upload", 422)
            path = Path(file if isinstance(file, str) else file.name)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise BackendError(
                    "Unsupported file format. Choose TXT, PDF, DOCX, audio or video from the supported formats listed above",
                    415,
                )
            if vlm_mode not in VLM_DESCRIPTIONS:
                raise BackendError(
                    "Choose Disabled, Automatic or Enhanced visual understanding", 422
                )
            # This local string is the mode submitted with this request, independent of later UI changes.
            submitted_mode = vlm_mode if path.suffix.lower() in VISUAL_EXTENSIONS else "disabled"
            state = empty_state()
            state["requested_vlm_mode"] = submitted_mode
            state["upload_status"] = "🔄 Uploading " + markdown_text(path.name) + "…"
            yield self.render(state, clear_inputs=True)
            if hasattr(self.client, "upload_events"):
                async for result in self.client.upload_events(path, vlm_mode=submitted_mode):
                    asset_id = result.get("id")
                    if not isinstance(asset_id, str) or not asset_id:
                        raise BackendError("The backend did not return a valid Asset ID", 502)
                    state.update(
                        asset_id=asset_id,
                        asset=result,
                        asset_details=asset_details(result),
                        upload_status=upload_progress(result),
                    )
                    yield self.render(state)
            else:
                result = await self.client.upload(path, vlm_mode=submitted_mode)
            asset_id = result.get("id")
            if not isinstance(asset_id, str) or not asset_id:
                raise BackendError("The backend did not return a valid Asset ID", 502)
            state["asset_id"] = asset_id
            await self._ready(asset_id, state)
            await self._load_content(asset_id, state)
            state["upload_status"] = (
                f"✅ File processed successfully. Ready for questions, summaries and Q&A.\n\nAsset ID: `{asset_id}`"
            )
            if state.get("content_warning"):
                state["upload_status"] += "\n\n" + state["content_warning"]
        except BackendError as error:
            state["upload_status"] = "❌ " + error.detail
        except Exception:
            logger.exception("Upload processing failed")
            state["upload_status"] = (
                "❌ The file could not be uploaded. Check the backend logs and try again."
            )
        yield self.render(state, clear_inputs=True)

    async def load_text(self, asset_id, value):
        state = self._state(value)
        try:
            selected = await self._ready(asset_id, state)
            await self._load_content(selected, state)
            state["upload_status"] = (
                "✅ Extracted content loaded. This Asset ID is selected in all four tabs."
            )
            if state.get("content_warning"):
                state["upload_status"] += "\n\n" + state["content_warning"]
        except BackendError as error:
            state["extracted_text"] = ""
            state["upload_status"] = "❌ " + error.detail
        except Exception:
            state["upload_status"] = (
                "❌ Extracted text could not be loaded. Check the backend logs."
            )
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
        result = {
            **result,
            **{
                key: feedback[key]
                for key in ("options", "answer", "explanation")
                if key in feedback
            },
        }
        if "generated_question" in feedback:
            result["question"] = feedback["generated_question"]
        if not all(
            isinstance(result.get(key), str) and result[key].strip()
            for key in ("question", "answer", "explanation")
        ):
            raise BackendError("The backend returned an incomplete question", 502)
        return {
            "thread_id": response["thread_id"],
            "review_id": response.get("review_id"),
            "status": status,
            "result": result,
            "notice": feedback.get("revision_notice") or result.get("review_notice", ""),
        }

    async def _stream_single_question_typewriter(
        self, state, question_item, status_prefix="✍️ Writing refined question…"
    ):
        import asyncio
        import re

        draft = deepcopy(question_item.get("result", {}))
        full_q = draft.get("question", "")
        full_options = draft.get("options") or []
        full_answer = str(draft.get("answer", ""))
        full_exp = draft.get("explanation", "")

        streaming_q = deepcopy(question_item)
        streaming_draft = streaming_q.get("result", {})
        streaming_draft.update(question="", options=[], answer="", explanation="")
        state["question"] = streaming_q
        state["question_status"] = status_prefix
        yield self.render(state)

        # 1. Stream question text word by word
        q_tokens = re.findall(r"\S+|\n+", full_q)
        for token in q_tokens:
            if "\n" in token:
                streaming_draft["question"] += token
            else:
                space = (
                    " "
                    if streaming_draft["question"]
                    and not streaming_draft["question"].endswith(("\n", " "))
                    else ""
                )
                streaming_draft["question"] += space + token
            yield self.render(state)
            await asyncio.sleep(0.015)

        # 2. Stream options if present
        if full_options:
            state["question_status"] = "✍️ Writing answer options…"
            curr_opts = []
            for opt in full_options:
                opt_str = str(opt)
                opt_tokens = re.findall(r"\S+|\n+", opt_str)
                curr_opt = ""
                for token in opt_tokens:
                    if "\n" in token:
                        curr_opt += token
                    else:
                        space = " " if curr_opt and not curr_opt.endswith(("\n", " ")) else ""
                        curr_opt += space + token
                    streaming_draft["options"] = curr_opts + [curr_opt]
                    yield self.render(state)
                    await asyncio.sleep(0.012)
                curr_opts.append(opt_str)
            streaming_draft["options"] = full_options

        # 3. Stream answer
        if full_answer:
            state["question_status"] = "✍️ Setting correct answer…"
            streaming_draft["answer"] = full_answer
            yield self.render(state)
            await asyncio.sleep(0.04)

        # 4. Stream explanation word by word
        if full_exp:
            state["question_status"] = "✍️ Writing explanation…"
            exp_tokens = re.findall(r"\S+|\n+", full_exp)
            for token in exp_tokens:
                if "\n" in token:
                    streaming_draft["explanation"] += token
                else:
                    space = (
                        " "
                        if streaming_draft["explanation"]
                        and not streaming_draft["explanation"].endswith(("\n", " "))
                        else ""
                    )
                    streaming_draft["explanation"] += space + token
                yield self.render(state)
                await asyncio.sleep(0.012)

        # Restore exact full question object
        state["question"] = deepcopy(question_item)

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
            state["question_status"] = (
                "🔄 Generating question…" if feedback is None else "🔄 Updating question…"
            )
            current["busy"] = True
            yield self.render(state)
            response = (
                await self.client.start_question(selected, question_type)
                if feedback is None
                else await self.client.feedback_question(
                    current["thread_id"], feedback.strip(), current["review_id"]
                )
            )
            result = response.get("current_state") or response.get("final_state") or {}
            if result.get("asset_id", state["asset_id"]) != state["asset_id"]:
                raise BackendError("The backend returned a question for a different file", 502)
            current = self._question_result(response)
            if feedback is not None and current.get("result") and feedback != "save":
                async for output in self._stream_single_question_typewriter(
                    state, current, status_prefix="✍️ Writing refined question…"
                ):
                    yield output
            else:
                state["question"] = current
            state["question_status"] = (
                "✅ Question accepted and saved."
                if current["status"] == "completed"
                else "✅ Question ready. Provide feedback, refine automatically, or accept/save."
            )
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
                state["question_status"] += (
                    " Generate a new question before submitting more feedback."
                )
            elif feedback is None:
                current.update(status="failed", review_id=None, error=error.detail)
        except Exception:
            current = state.setdefault("question", {})
            current.update(
                status="failed",
                review_id=None,
                error="Question generation failed. Check the backend logs",
            )
            state["question_status"] = (
                "❌ Question generation failed. Check the backend logs and generate a new question."
            )
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
                events = self.client.summary_events(
                    thread_id=current["thread_id"],
                    feedback=feedback.strip(),
                    review_id=current["review_id"],
                )
            current["status"] = "running"
            current["busy"] = True
            state["summary_status"] = (
                "🔄 Generating summary…" if feedback is None else "🔄 Updating summary…"
            )
            yield self.render(state)
            async for event in events:
                current["thread_id"] = event["thread_id"]
                kind, data, stage = event["event"], event["data"], event.get("stage")
                if data.get("provider_warnings"):
                    current["provider_notice"] = "\n".join(data["provider_warnings"])
                if kind == "stage_started":
                    if stage == "main_point_summarizer":
                        current["preview_outline"] = ""
                    elif stage in {"summarizer_writer", "summarizer_rewriter"}:
                        current["preview_summary"] = ""
                    state["summary_status"] = "🔄 " + {
                        "main_point_summarizer": "Generating main points…",
                        "summarizer_writer": "Writing detailed summary…",
                        "summarizer_structure": "Revising outline and sections…",
                        "summarizer_rewriter": "Revising summary…",
                    }.get(stage, "Finishing summary…")
                elif kind == "text_delta":
                    key = (
                        "preview_outline" if stage == "main_point_summarizer" else "preview_summary"
                    )
                    current[key] = current.get(key, "") + data.get("text", "")
                elif kind == "awaiting_feedback":
                    if (
                        not data.get("review_id")
                        or not isinstance(data.get("summary"), str)
                        or not data["summary"].strip()
                    ):
                        raise BackendError("The backend returned an incomplete summary review", 502)
                    current.update(
                        status="interrupted_for_feedback",
                        review_id=data["review_id"],
                        outline=data.get("table_of_contents", ""),
                        text=data["summary"],
                        notice=data.get("revision_notice", ""),
                    )
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
            state["summary_status"] = (
                "✅ Summary accepted and saved."
                if current.get("status") == "completed"
                else "✅ Summary ready. Provide feedback to revise wording or structure, or accept/save."
            )
            if current.get("notice"):
                state["summary_status"] += "\n\n" + markdown_text(current["notice"])
            if current.get("provider_notice"):
                state["summary_status"] += "\n\n⚠️ " + markdown_text(current["provider_notice"])
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
                state["summary_status"] += (
                    " Click Generate Summary to try again. Update Summary and Accept / Save require a successfully generated summary."
                )
        except Exception:
            current = state.setdefault("summary", {})
            current.update(
                status="failed",
                review_id=None,
                error="Summary generation failed. Check the backend logs",
            )
            current.pop("preview_outline", None)
            current.pop("preview_summary", None)
            state["summary_status"] = (
                "❌ Summary generation failed. Check the backend logs, then click Generate Summary to try again."
            )
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
        for citation in citations:
            if (
                not isinstance(citation, dict)
                or citation.get("asset_id", state["asset_id"]) != state["asset_id"]
            ):
                continue
            label = f"[{citation.get('number', len(rows) + 1)}] {citation.get('source', asset.get('source_name', 'Source'))}"
            places = []
            for key, name in (
                ("page", "Page"),
                ("section", "Section"),
                ("paragraph", "Paragraph"),
                ("table_index", "Table"),
            ):
                if citation.get(key) is not None:
                    places.append(f"{name} {citation[key]}")
            if citation.get("start") is not None:
                places.append(
                    f"{citation['start']:g}s"
                    if isinstance(citation["start"], (int, float))
                    else str(citation["start"])
                )
            if citation.get("source_type") == "video":
                places.append(
                    "Video visual"
                    if citation.get("content_type") == "visual_description"
                    else "Video transcript"
                )
            title = markdown_text(label)
            url = self._source_link(citation, state)
            if url:
                title = f"[{title}]({url})"
            rows.append(
                "- " + title + (" — " + markdown_text(" · ".join(places)) if places else "")
            )
            origins = citation.get("origin_locations", [])
            if isinstance(origins, str):
                try:
                    origins = json.loads(origins)
                except ValueError:
                    origins = []
            for origin in origins if isinstance(origins, list) else []:
                if not isinstance(origin, dict) or origin.get("source_file_id") == citation.get(
                    "source_file_id"
                ):
                    continue
                url = self._source_link(origin, state)
                label = markdown_text(origin.get("source", "Additional original source"))
                rows.append("- " + (f"[{label}]({url})" if url else label))
        return "\n\n**Sources**\n\n" + "\n".join(rows) if rows else ""

    def _cited_answer(self, text, citations, state):
        body, marker, appendix = text.rpartition("\n\nSources:\n")
        lines = appendix.splitlines()
        expected = [
            f"[{item['number']}] "
            for item in citations
            if isinstance(item, dict) and "number" in item
        ]
        if (
            marker
            and expected
            and len(lines) == len(expected)
            and all(line.startswith(prefix) for line, prefix in zip(lines, expected))
        ):
            if state.get("asset", {}).get("source_type") in {"combined", "collection"}:
                return body + self._citations(citations, state)
            original = next(
                (
                    item
                    for item in state.get("asset", {}).get("files", [])
                    if isinstance(item, str) and item.startswith("original.")
                ),
                None,
            )
            rows = []
            for line, citation, prefix in zip(lines, citations, expected):
                label = markdown_text(prefix.strip())
                if original:
                    try:
                        url = self.client.source_url(
                            state["asset_id"],
                            original,
                            page=citation.get("page"),
                            timestamp=citation.get("start"),
                        )
                        label = f"[{label}]({url})"
                    except BackendError:
                        pass
                rows.append("- " + label + " " + line[len(prefix) :])
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
            if start or not current.get("thread_id"):
                effective_asset_id = asset_id or state.get("asset_id")
                await self._ready(effective_asset_id, state)
                current = {"messages": current.get("messages", []), "citations": current.get("citations", [])}
                state["qa"] = current
                start = not bool(current.get("thread_id"))
            elif current.get("status") != "completed":
                raise BackendError("Please wait for the current answer to complete", 409)
            current.update(status="running", pending={"question": question, "text": ""})
            state["qa_status"] = "🔄 Generating a source-grounded answer…"
            yield self.render(state)
            completed = None
            async for event in self.client.qa_events(
                asset_id=state["asset_id"],
                question=question,
                thread_id=None if start else current["thread_id"],
            ):
                current["thread_id"] = event["thread_id"]
                kind, data = event["event"], event["data"]
                if kind == "text_delta":
                    import asyncio
                    import re

                    delta_text = data.get("text", "")
                    tokens = re.findall(r"\S+|\n+", delta_text)
                    for token in tokens:
                        if "\n" in token:
                            current["pending"]["text"] += token
                        else:
                            space = (
                                " "
                                if current["pending"]["text"]
                                and not current["pending"]["text"].endswith(("\n", " "))
                                else ""
                            )
                            current["pending"]["text"] += space + token
                        yield self.render(state)
                        await asyncio.sleep(0.015)
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
            if not isinstance(citations, list) or any(
                not isinstance(item, dict)
                or item.get("asset_id", state["asset_id"]) != state["asset_id"]
                for item in citations
            ):
                raise BackendError("The answer returned invalid source citations", 502)
            current["messages"].extend(
                [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": self._cited_answer(pending["text"], citations, state),
                    },
                ]
            )
            current.pop("pending")
            current.setdefault("citations", []).append(citations)
            current.update(status="completed", answer_status=completed.get("answer_status"))
            state["qa_status"] = "✅ Response complete. Ask a follow-up question."
            if completed.get("answer_status") in {
                "insufficient_evidence",
                "unsupported",
                "insufficient",
            }:
                state["qa_status"] = (
                    "📋 This file does not contain enough evidence to answer the question."
                )
            elif completed.get("answer_status") == "off_topic":
                state["qa_status"] = "📋 This question is outside the selected file."
            if completed.get("provider_warnings"):
                state["qa_status"] += "\n\n⚠️ " + markdown_text(
                    "\n".join(completed["provider_warnings"])
                )
            success = True
        except BackendError as error:
            if current.get("pending"):
                current.pop("pending", None)
                current["status"] = "failed"
            state["qa_status"] = "❌ " + error.detail
        except Exception:
            current.pop("pending", None)
            current["status"] = "failed"
            state["qa_status"] = (
                "❌ The answer could not be completed. Check the backend logs and start a new Q&A session."
            )
        yield self.render(state, clear_message=success)

    async def start_qa(self, asset_id, question, value):
        async for output in self._chat(question, value, asset_id=asset_id, start=True):
            yield output

    async def continue_qa(self, message, value):
        async for output in self._chat(message, value):
            yield output
