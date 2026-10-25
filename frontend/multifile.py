import json
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

import gradio as gr

from frontend.client import BackendError
from frontend.constants import VISUAL_EXTENSIONS, VLM_CHOICES
from frontend.presentation import markdown_text
from frontend.state import empty_state


def file_paths(files):
    return [
        str(item if isinstance(item, (str, Path)) else item.name)
        for item in (files if isinstance(files, list) else [files])
        if item
    ]


def batch_rows(batch):
    rows = []
    labels = {
        "extracting_document_text": "Extracting document text and tables",
        "extracting_document_ocr": "Recognizing scanned document text",
        "detecting_visuals": "Detecting relevant visual content",
        "detecting_visual_content": "Detecting relevant visual content",
        "visual_artifacts_detected": "Visual selection complete",
        "analyzing_visuals": "Analyzing diagrams with VLM",
        "transcribing": "Transcribing audio",
        "merging_multimodal": "Merging multimodal information",
        "reusing_visuals": "Reusing cached visual understanding",
        "indexing": "Building source index",
    }
    for item in batch.get("files", []):
        stage = (item.get("stages") or [{}])[-1]
        error = item.get("error") or {}
        detail = error.get("detail", "") if isinstance(error, dict) else str(error)
        rows.append(
            [
                item.get("source_name", ""),
                str(item.get("source_type", "")).upper(),
                f"{item.get('size_bytes', 0) / 1024:.1f} KB",
                item.get("status", "pending"),
                item.get("vlm_mode", "disabled"),
                detail
                or labels.get(stage.get("name"), str(stage.get("name", "")).replace("_", " ")),
                item.get("id", ""),
            ]
        )
    return rows


class MultifileUI:
    async def refresh_provider_status(self):
        try:
            payload = await self.client.provider_status(refresh=True)
            rows = ["| Task | Provider / model | Status | Details |", "|---|---|---|---|"]
            for item in payload.get("providers", []):
                values = [
                    item.get("task", ""),
                    f"{item.get('provider', '')} / {item.get('model', '')}",
                    item.get("status", "unknown"),
                    item.get("message", ""),
                ]
                rows.append(
                    "| "
                    + " | ".join(
                        markdown_text(str(value)).replace("|", "\\|").replace("\n", " ")
                        for value in values
                    )
                    + " |"
                )
            fallback = payload.get("fallback", {})
            rows.append(
                "\nChecks read provider metadata and do not run inference. Unknown capabilities still require validation during a request."
            )
            rows.append(
                f"\nFallback: {'enabled' if fallback.get('enabled') else 'disabled'}; paid fallback: {'allowed by settings' if fallback.get('allow_paid') else 'disabled'}."
            )
            for name, tool in payload.get("tools", {}).items():
                rows.append(
                    "\n**"
                    + markdown_text(name.replace("_", " ").capitalize())
                    + "**: "
                    + ("enabled" if tool.get("enabled") else "disabled")
                    + ". "
                    + markdown_text(tool.get("message", ""))
                )
                if name == "ocr":
                    rows.append(
                        "Languages: "
                        + markdown_text(tool.get("languages", ""))
                        + "; backend: "
                        + markdown_text(tool.get("backend", ""))
                        + "; Tesseract binary: "
                        + ("found" if tool.get("tesseract_binary_available") else "not found")
                        + "."
                    )
                    rows.append(
                        "PyMuPDF language data: "
                        + (
                            "available"
                            if tool.get("language_data_available")
                            else "missing ("
                            + markdown_text(", ".join(tool.get("missing_languages") or []))
                            + ")"
                        )
                        + "."
                    )
                elif not tool.get("binary_available"):
                    rows.append("LibreOffice binary was not found.")
            return "\n".join(rows), payload
        except BackendError as error:
            return "⚠️ Provider status unavailable: " + markdown_text(error.detail), {
                "status": "unavailable",
                "message": error.detail,
            }

    @staticmethod
    def override_files(files, value):
        paths = file_paths(files)
        current = value if isinstance(value, dict) else {}
        overrides = {key: mode for key, mode in current.items() if key in paths}
        return gr.update(
            choices=[(Path(path).name, path) for path in paths], value=paths[0] if paths else None
        ), overrides

    @staticmethod
    def configure_override(path, preference, overrides):
        allowed = (preference or {}).get("allowed", ["disabled"])
        enabled = bool(
            path
            and Path(path).suffix.lower() in VISUAL_EXTENSIONS
            and (preference or {}).get("available")
        )
        choices = (
            [item for item in VLM_CHOICES if item[1] in allowed] if enabled else VLM_CHOICES[:1]
        )
        current = overrides if isinstance(overrides, dict) else {}
        selected = current.get(path, (preference or {}).get("selected", "disabled"))
        selected = selected if enabled and selected in allowed else "disabled"
        return gr.update(choices=choices, value=selected, interactive=enabled)

    @staticmethod
    def set_override(path, mode, preference, value):
        overrides = deepcopy(value or {})
        enabled = bool(
            path
            and Path(path).suffix.lower() in VISUAL_EXTENSIONS
            and (preference or {}).get("available")
        )
        allowed = (preference or {}).get("allowed", ["disabled"])
        if path:
            overrides[path] = mode if enabled and mode in allowed else "disabled"
        return overrides, "\n".join(
            f"- {Path(key).name}: {item}" for key, item in overrides.items()
        ) or "All files use the main mode."

    @staticmethod
    def pending_files(files):
        rows = []
        for value in file_paths(files):
            path = Path(value)
            try:
                size = f"{path.stat().st_size / 1024:.1f} KB"
            except OSError:
                size = "Unavailable"
            rows.append(
                [
                    path.name,
                    path.suffix.lstrip(".").upper(),
                    size,
                    "pending",
                    "",
                    "Ready to process",
                    "",
                ]
            )
        return rows

    def render_batch(self, state):
        batch = state.get("batch") or {}
        jobs = [
            (item.get("source_name", item["id"]), item["id"])
            for item in batch.get("files", [])
            if item.get("id")
        ]
        if batch.get("id"):
            jobs.insert(0, ("Batch processing", batch["id"]))
        if batch.get("combined_asset_id"):
            jobs.insert(0, ("Combined learning context", batch["combined_asset_id"]))
        selected = state.get("asset_id")
        if selected and not any(value == selected for _, value in jobs):
            jobs.append(
                (state.get("asset", {}).get("source_name") or "Selected Asset ID", selected)
            )
        job_selected = (
            selected
            if any(value == selected for _, value in jobs)
            else (jobs[0][1] if jobs else None)
        )
        return (*self.render(state), batch_rows(batch), gr.update(choices=jobs, value=job_selected))

    async def _apply_batch(self, batch, state):
        state["batch"] = batch
        completed = [item for item in batch.get("files", []) if item.get("status") == "ready"]
        failed = sum(item.get("status") == "failed" for item in batch.get("files", []))
        total = len(batch.get("files", []))
        terminal = batch.get("status") in {"ready", "partial", "failed"}
        state["upload_status"] = (
            f"{'✅' if terminal and completed else '❌' if terminal else '🔄'} {len(completed)} of {total} files ready · {failed} failed."
        )
        if batch.get("warnings"):
            state["upload_status"] += "\n\n" + "\n".join(str(w) for w in batch["warnings"])
        if terminal and completed:
            combined = (
                batch.get("combined_asset_id") if batch.get("combined_status") == "ready" else None
            )
            # The active learning workflows must use the unified collection by
            # default, even when resuming a batch that previously had a source
            # file selected. Individual sources remain available as a fallback
            # only when the merge itself failed.
            selected = combined or completed[0]["id"]
            status = state["upload_status"]
            await self._ready(selected, state)
            await self._load_content(selected, state)
            state["upload_status"] = status + (
                " The combined learning context is ready for questions, summaries and chat."
                if combined
                else " The combined context is unavailable; the first successful file is active. Retry / Resume Failed Files to rebuild the context."
            )

    async def upload_files(
        self, files, value, vlm_mode="disabled", overrides=None, preference=None
    ):
        state = empty_state()
        try:
            paths = file_paths(files)
            if not paths:
                raise BackendError("Select one or more files first", 422)
            # Gradio resolves State when this serialized callback begins. A queued
            # capability response may have repainted the visible dropdown after
            # the user chose a mode; retain that user's last submitted choice.
            if (
                isinstance(preference, dict)
                and preference.get("available")
                and preference.get("visual_file", True)
            ):
                selected = preference.get("selected")
                if selected in preference.get("allowed", []):
                    vlm_mode = selected
            elif isinstance(preference, dict):
                vlm_mode = "disabled"
            state["upload_status"] = f"🔄 Uploading {len(paths)} files…"
            state["batch"] = {
                "files": [
                    {"id": "", "source_name": Path(path).name, "status": "pending"}
                    for path in paths
                ]
            }
            yield self.render_batch(state)
            configs = [
                {"index": index, "vlm_mode": overrides[path]}
                for index, path in enumerate(paths)
                if overrides and path in overrides
            ]
            async for batch in self.client.batch_events(
                paths, vlm_mode=vlm_mode, file_configs=configs
            ):
                await self._apply_batch(batch, state)
                yield self.render_batch(state)
        except BackendError as error:
            state["upload_status"] = "❌ " + error.detail
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Batch frontend processing failed")
            state["upload_status"] = (
                "❌ Processing could not be completed. Resume Processing Status to check submitted files."
            )
        yield self.render_batch(state)

    async def resume_batch(self, value):
        state = self._state(value)
        try:
            batch_id = state.get("batch", {}).get("id")
            if not batch_id:
                raise BackendError("Process files first", 422)
            await self._apply_batch(await self.client.get_batch(batch_id), state)
        except BackendError as error:
            state["upload_status"] = "❌ " + error.detail
        yield self.render_batch(state)

    async def retry_failed_batch(self, value):
        import asyncio
        import time

        state = self._state(value)
        try:
            batch_id = state.get("batch", {}).get("id")
            if not batch_id:
                raise BackendError("Process files first", 422)
            batch = await self.client.resume_failed_batch(batch_id)
            deadline = time.monotonic() + self.client.settings.FRONTEND_UPLOAD_TIMEOUT_SECONDS
            while True:
                await self._apply_batch(batch, state)
                yield self.render_batch(state)
                if batch.get("status") in {"ready", "partial", "failed"}:
                    break
                if time.monotonic() >= deadline:
                    state["upload_status"] = (
                        "Processing continues on the backend. Resume Processing Status to check progress."
                    )
                    break
                await asyncio.sleep(1)
                batch = await self.client.get_batch(batch_id)
        except BackendError as error:
            state["upload_status"] = "❌ " + error.detail
        yield self.render_batch(state)

    def select_asset_batch(self, asset_id, value):
        return self.render_batch(self.select_asset(asset_id, value)[0])

    async def load_text_batch(self, asset_id, value):
        async for result in self.load_text(asset_id, value):
            yield self.render_batch(result[0])

    async def activate_source(self, asset_id, value):
        state = self._state(value)
        if asset_id:
            try:
                await self._ready(asset_id, state)
                await self._load_content(asset_id, state)
                state["upload_status"] = "✅ Learning context selected in all workflow tabs."
            except BackendError as error:
                state["upload_status"] = "❌ " + error.detail
        return self.render_batch(state)

    async def refresh_jobs(self, file_id):
        try:
            if not file_id:
                return (
                    gr.update(choices=[], value=None),
                    {},
                    None,
                    "Select a processed file or batch.",
                )
            payload = await self.client.list_jobs(file_id)
            jobs = payload.get("jobs", [])
            choices = [
                (
                    f"{item.get('job_type', 'Job')} · {item.get('status', '')} · {item.get('timestamp', item.get('created_at', ''))}",
                    item["job_id"],
                )
                for item in jobs
            ]
            selected = choices[0][1] if choices else None
            result = await self.client.get_job(file_id, selected) if selected else {}
            return (
                gr.update(choices=choices, value=selected),
                result,
                None,
                f"{len(jobs)} recorded operations. Select a job to inspect its actual result.",
            )
        except BackendError as error:
            return gr.update(choices=[], value=None), {}, None, "❌ " + error.detail

    @staticmethod
    def _download_path(name, data):
        folder = Path(tempfile.gettempdir()) / "wizora-job-results" / str(uuid.uuid4())
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / name
        target.write_bytes(data)
        return str(target)

    async def inspect_job(self, file_id, job_id):
        try:
            if not file_id or not job_id:
                return {}, None, "Select a file and recorded operation."
            result = await self.client.get_job(file_id, job_id)
            return result, None, f"{result.get('job_type', 'Job')} · {result.get('status', '')}"
        except BackendError as error:
            return {}, None, "❌ " + error.detail

    async def download_job(self, file_id, job_id):
        try:
            if not file_id or not job_id:
                raise BackendError("Select a recorded operation first", 422)
            data = await self.client.download_job(file_id, job_id)
            json.loads(data)
            return self._download_path("job-result.json", data), "✅ JSON download ready."
        except (BackendError, ValueError) as error:
            return None, "❌ " + str(error)

    async def export_jobs(self, value):
        try:
            batch = self._state(value).get("batch", {})
            ids = [item["id"] for item in batch.get("files", []) if item.get("id")]
            ids.extend(item for item in [batch.get("id"), batch.get("combined_asset_id")] if item)
            selected = self._state(value).get("asset_id")
            if selected:
                ids.append(selected)
            source_ids = self._state(value).get("asset", {}).get("source_file_ids", [])
            if isinstance(source_ids, list):
                ids.extend(item for item in source_ids if isinstance(item, str) and item)
            if not ids:
                ids = [selected] if selected else []
            if not ids:
                raise BackendError("Process files first", 422)
            data = await self.client.export_jobs(list(dict.fromkeys(ids)))
            if not data.startswith(b"PK"):
                raise BackendError("The backend returned an invalid ZIP archive", 502)
            return self._download_path(
                "wizora-job-results.zip", data
            ), "✅ ZIP contains recorded results for all files and the batch."
        except BackendError as error:
            return None, "❌ " + error.detail

    async def prepare_zip_download(self, value):
        path, status = await self.export_jobs(value)
        return gr.update(value=path, interactive=bool(path)), status

    async def prepare_browser_zip(self, value):
        from zipfile import BadZipFile

        from frontend.exports import EXPORTS

        path, status = await self.export_jobs(value)
        try:
            entry = EXPORTS.register(path) if path else None
            return (
                gr.update(interactive=bool(entry)),
                entry,
                status,
                gr.update(value=path, interactive=bool(path)),
            )
        except (OSError, ValueError, BadZipFile) as error:
            return (
                gr.update(interactive=False),
                None,
                "❌ " + str(error),
                gr.update(value=None, interactive=False),
            )

    def render_questions(self, state):
        return (*self.render(state), self.question_choices(state))

    @staticmethod
    def question_choices(state):
        bank = state.get("question_bank", [])
        current = state.get("question", {})
        choices = [(f"Question {index}", item["thread_id"]) for index, item in enumerate(bank, 1)]
        return gr.update(choices=choices, value=current.get("thread_id") if choices else None)

    def change_question_type(self, question_type, value):
        state = self._state(value)
        state.update(
            question={},
            question_bank=[],
            question_status=f"📋 Generate new {question_type} questions for the selected source.",
        )
        result = list(self.render_questions(state))
        result[8] = gr.update(value="", visible=question_type == "MCQ")
        return tuple(result)

    def change_difficulty(self, difficulty, value):
        from helper.question_review import DIFFICULTY_RUBRIC

        state = self._state(value)
        state.update(
            question={},
            question_bank=[],
            question_status=(
                f"Generate a new {difficulty} question set. " + DIFFICULTY_RUBRIC[difficulty]
            ),
        )
        return self.render_questions(state)

    def select_question(self, thread_id, value):
        state = self._state(value)
        current = state.get("question", {})
        bank = state.get("question_bank", [])
        for index, item in enumerate(bank):
            if item.get("thread_id") == current.get("thread_id"):
                bank[index] = deepcopy(current)
        selected = next((item for item in bank if item.get("thread_id") == thread_id), None)
        if selected:
            state["question"] = deepcopy(selected)
            state["question_status"] = (
                "✅ Question selected. Feedback, refinement and save apply to this question."
            )
        return self.render_questions(state)

    async def _stream_question_typewriter(
        self, state, bank, target_q, status_prefix="✍️ Writing generated question…"
    ):
        import asyncio
        import re

        draft = deepcopy(target_q.get("result", {}))
        full_q = draft.get("question", "")
        full_options = draft.get("options") or []
        full_answer = str(draft.get("answer", ""))
        full_exp = draft.get("explanation", "")

        state["question_bank"] = bank
        streaming_q = deepcopy(target_q)
        streaming_draft = streaming_q.get("result", {})
        streaming_draft.update(question="", options=[], answer="", explanation="")
        state["question"] = streaming_q
        state["question_status"] = status_prefix
        yield self.render_questions(state)

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
            yield self.render_questions(state)
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
                    yield self.render_questions(state)
                    await asyncio.sleep(0.012)
                curr_opts.append(opt_str)
            streaming_draft["options"] = full_options

        # 3. Stream answer
        if full_answer:
            state["question_status"] = "✍️ Setting correct answer…"
            streaming_draft["answer"] = full_answer
            yield self.render_questions(state)
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
                yield self.render_questions(state)
                await asyncio.sleep(0.012)

        # Restore exact full question object
        state["question"] = deepcopy(target_q)

    async def start_question_set(self, asset_id, question_type, difficulty, count, value):
        state = self._state(value)
        state.update(
            question={},
            question_bank=[],
            question_status="🔄 Identifying concepts and generating questions…",
        )
        yield self.render_questions(state)
        try:
            selected = await self._ready(asset_id, state)
            response = await self.client.start_question(
                selected, question_type, difficulty, int(count)
            )
            sessions = (
                response.get("question_sessions") if "question_sessions" in response else [response]
            )
            if not isinstance(sessions, list):
                raise BackendError("The backend returned invalid question sessions", 502)
            bank = [self._question_result(item) for item in sessions]
            for item in bank:
                result = item["result"]
                if result.get("asset_id", selected) != selected:
                    raise BackendError(
                        "The backend returned a question for a different source", 502
                    )
                if result.get("question_type", question_type) != question_type:
                    raise BackendError("The backend returned a different question type", 502)
            if bank:
                async for output in self._stream_question_typewriter(state, bank, bank[0]):
                    yield output
            else:
                state.update(question_bank=[], question={})
            missing = max(0, int(count) - len(bank))
            state["question_status"] = (
                f"{'⚠️' if missing else '✅'} {len(bank)} of {int(count)} requested questions ready."
            )
            if missing:
                quota = any(
                    item.get("outcome") == "quota_exceeded"
                    for item in response.get("attempt_history", [])
                )
                if quota:
                    state["question_status"] += (
                        " Generation paused because the provider quota was reached. Wait for quota recovery, then retry."
                    )
                else:
                    state["question_status"] += (
                        f" {missing} could not be generated after {response.get('generation_attempts', 'bounded')} attempts. Retry generation or choose another source section."
                    )
            elif bank:
                state["question_status"] += " Select a question to review, refine or save."
            flagged = sum(
                bool(
                    item["result"].get("review_required")
                    or item["result"].get("validation", {}).get("status") == "review_required"
                )
                for item in bank
            )
            if flagged:
                state["question_status"] += (
                    f"\n\n⚠️ {flagged} question(s) require review. Check their source evidence, answer and difficulty before saving."
                )
            if response.get("warnings"):
                state["question_status"] += "\n\n" + "\n".join(response["warnings"])
        except BackendError as error:
            state["question_status"] = "❌ " + error.detail
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Question set generation failed")
            state["question_status"] = "❌ Question generation failed. Check the backend logs."
        yield self.render_questions(state)
