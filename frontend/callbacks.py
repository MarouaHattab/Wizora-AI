import asyncio
import time
from copy import deepcopy
from pathlib import Path

from .components import WORKFLOWS, browser_value, fresh_state, render, session
from .api import BackendError


def error_text(error):
    detail = str(getattr(error, "detail", error))
    if getattr(error, "status_code", None) == 409:
        detail += " Refresh this session to load its current review before submitting again."
    return detail


def question_snapshot(response):
    source = response.get("current_state", response.get("final_state", {}))
    result = {key: source[key] for key in (
        "question", "question_type", "options", "answer", "explanation", "critic_feedback",
        "review_notice", "revision_limit_reached",
    ) if key in source}
    review = response.get("data_for_feedback", {})
    if review.get("revision_notice"):
        result["review_notice"] = review["revision_notice"]
    return {"thread_id": response["thread_id"], "status": response["status"],
            "review_id": response.get("review_id"), "result": result}


class Callbacks:
    def __init__(self, client):
        self.client = client

    def _seed_saved(self, state, saved):
        values = saved.get("sessions", {}) if isinstance(saved, dict) else {}
        if not isinstance(values, dict):
            return
        for asset_id, workflows in list(values.items())[:100]:
            if asset_id not in state["assets"] or not isinstance(workflows, dict):
                continue
            target = state["sessions"].setdefault(asset_id, {})
            for name in WORKFLOWS:
                identifier = workflows.get(name)
                if isinstance(identifier, str) and 0 < len(identifier) <= 128:
                    target.setdefault(name, {"thread_id": identifier})

    async def _library(self, state):
        response = await self.client.get_assets()
        state["assets"] = {item["id"]: item for item in response.get("assets", [])
                           if isinstance(item, dict) and item.get("id") and item.get("status") == "ready"}

    async def _restore(self, state, saved, asset_id):
        state.update(selected_asset=asset_id, extracted="", busy=None)
        if not isinstance(asset_id, str) or asset_id not in state["assets"]:
            state["selected_asset"] = None
            return
        state["job"] = await self.client.get_asset(asset_id)
        state["assets"][asset_id] = state["job"]
        saved_sessions = saved.get("sessions", {}) if isinstance(saved, dict) else {}
        ids = saved_sessions.get(asset_id, {}) if isinstance(saved_sessions, dict) else {}
        if not isinstance(ids, dict):
            ids = {}
        for workflow in WORKFLOWS:
            current = session(state, workflow)
            thread_id = current.get("thread_id") or ids.get(workflow)
            if not isinstance(thread_id, str) or not thread_id:
                continue
            try:
                restored = await self.client.get_session(workflow, thread_id)
                if restored.get("asset_id") != asset_id:
                    raise BackendError("The saved session belongs to another material.", 409)
                current.clear()
                current.update(restored)
                if workflow == "qa" and restored.get("status") == "completed":
                    current["answer_notice"] = self._answer_notice(restored.get("result", {}).get("answer_status"))
            except BackendError as error:
                current.clear()
                if error.status_code != 404:
                    state["notice"] = error_text(error)

    async def load(self, saved):
        state = fresh_state()
        try:
            await self._library(state)
            self._seed_saved(state, saved)
            selected = saved.get("selected_asset") if isinstance(saved, dict) else None
            await self._restore(state, saved, selected)
        except BackendError as error:
            state["notice"] = error_text(error)
        return render(state, self.client)

    async def select(self, asset_id, state, saved):
        state = deepcopy(state or fresh_state())
        state["notice"] = ""
        try:
            await self._restore(state, saved, asset_id)
        except BackendError as error:
            state.update(selected_asset=None, extracted="", busy=None, notice=error_text(error))
        return render(state, self.client)

    async def refresh(self, state, saved):
        state = deepcopy(state or fresh_state())
        state.update(notice="", busy=None)
        try:
            await self._library(state)
            await self._restore(state, saved, state.get("selected_asset"))
        except BackendError as error:
            state["notice"] = error_text(error)
        return render(state, self.client)

    async def upload(self, path, state):
        state = deepcopy(state or fresh_state())
        if state.get("busy"):
            state["notice"] = "Wait for the current request to finish."
            yield render(state, self.client)
            return
        if not path:
            state["notice"] = "Choose a PDF, DOCX, audio, or video file first."
            yield render(state, self.client)
            return
        state.update(busy="upload", notice="Uploading the selected file…")
        yield render(state, self.client)
        try:
            state["job"] = await self.client.upload(Path(path))
            deadline = time.monotonic() + self.client.settings.FRONTEND_TIMEOUT_SECONDS
            while state["job"].get("status") in {"queued", "processing"}:
                state["notice"] = "Processing the upload. Actual stages appear in Processing Status."
                yield render(state, self.client)
                if time.monotonic() >= deadline:
                    raise BackendError("Upload processing is still running. Use Check processing to retrieve its latest status.", 504)
                await asyncio.sleep(min(1, max(.01, deadline - time.monotonic())))
                state["job"] = await self.client.get_upload(state["job"]["id"])
            if state["job"].get("status") != "ready":
                error = state["job"].get("error") or {}
                raise BackendError(error.get("detail", "The file could not be processed."))
            await self._library(state)
            await self._restore(state, browser_value(state), state["job"]["id"])
            response = await self.client.extracted_text(state["selected_asset"])
            state["extracted"] = response.get("extracted_text", "")
            state["notice"] = "Material ready. Generate questions, summarize the content, or start a conversation."
        except BackendError as error:
            state["notice"] = error_text(error)
        finally:
            state["busy"] = None
        yield render(state, self.client)

    async def check_processing(self, state):
        state = deepcopy(state or fresh_state())
        if not state.get("job"):
            state["notice"] = "Upload a file to begin processing."
        else:
            try:
                state["job"] = await self.client.get_upload(state["job"]["id"])
                if state["job"].get("status") == "ready":
                    await self._library(state)
                    state["notice"] = "Processing complete. The material is available in the library."
            except BackendError as error:
                state["notice"] = error_text(error)
        return render(state, self.client)

    async def preview(self, state):
        state = deepcopy(state or fresh_state())
        if not state.get("selected_asset"):
            state["notice"] = "Select a material first."
        else:
            try:
                response = await self.client.extracted_text(state["selected_asset"])
                state["extracted"] = response.get("extracted_text", "")
            except BackendError as error:
                state["notice"] = error_text(error)
        return render(state, self.client)

    def _ready(self, state):
        asset = state.get("assets", {}).get(state.get("selected_asset"), {})
        if not asset or asset.get("status") != "ready":
            raise BackendError("Select a processed material first.")
        if state.get("busy"):
            raise BackendError("Wait for the current request to finish.")

    async def question(self, question_type, feedback, state, action="start"):
        state = deepcopy(state or fresh_state())
        try:
            self._ready(state)
            current = session(state, "question_generation")
            if action != "start" and current.get("status") != "interrupted_for_feedback":
                raise BackendError("Refresh the session to load a question awaiting review.", 409)
            if action == "revise" and not feedback.strip():
                raise BackendError("Describe the revision you want to make.")
            state.update(busy="question", notice="")
            current.update(status="running", error=None)
            yield render(state, self.client)
            if action == "start":
                response = await self.client.start_question(state["selected_asset"], question_type)
            else:
                response = await self.client.feedback_question(
                    current["thread_id"], {"save": "save", "auto": "auto"}.get(action, feedback), current.get("review_id"),
                )
            current.clear()
            current.update(question_snapshot(response))
        except BackendError as error:
            state["notice"] = error_text(error)
            if state.get("busy") == "question":
                session(state, "question_generation").update(status="failed", error=error_text(error))
        finally:
            state["busy"] = None
        yield render(state, self.client)

    async def summary(self, feedback, revision_kind, state, action="start"):
        state = deepcopy(state or fresh_state())
        try:
            self._ready(state)
            current = session(state, "summarization")
            if action != "start" and current.get("status") != "interrupted_for_feedback":
                raise BackendError("Refresh the session to load a summary awaiting review.", 409)
            if action == "revise" and not feedback.strip():
                raise BackendError("Describe the revision you want to make.")
            state.update(busy="summary", notice="")
            current.update(status="running", error=None)
            yield render(state, self.client)
            if action == "start":
                events = self.client.summary_events(asset_id=state["selected_asset"])
            else:
                requested = "save" if action == "save" else feedback.strip()
                if action == "revise" and revision_kind == "Structure":
                    requested = "Reorganize the outline and sections: " + requested
                events = self.client.summary_events(thread_id=current["thread_id"], feedback=requested, review_id=current.get("review_id"))
            async for event in events:
                data = event.get("data", {})
                current["thread_id"] = event["thread_id"]
                kind, stage = event["event"], event.get("stage")
                if kind == "stage_started":
                    if stage == "main_point_summarizer":
                        current["preview_outline"] = ""
                    elif stage in {"summarizer_writer", "summarizer_rewriter"}:
                        current["preview_summary"] = ""
                elif kind == "text_delta":
                    key = "preview_outline" if stage == "main_point_summarizer" else "preview_summary"
                    current[key] = current.get(key, "") + data.get("text", "")
                elif kind == "awaiting_feedback":
                    current.update(status=data["status"], review_id=data.get("review_id"), result={
                        "Main_Points": data.get("table_of_contents", ""), "summary": data.get("summary", ""),
                        "review_notice": data.get("revision_notice", ""), "revision_kind": data.get("revision_kind"),
                    })
                    current.pop("preview_outline", None)
                    current.pop("preview_summary", None)
                elif kind == "completed":
                    current.update(status="completed", review_id=None)
                    current.setdefault("result", {})["summary"] = data.get("summary", "")
                    current.pop("preview_summary", None)
                elif kind == "failed":
                    raise BackendError(data.get("detail", "Summary generation failed."))
                yield render(state, self.client)
        except BackendError as error:
            state["notice"] = error_text(error)
            if state.get("busy") == "summary":
                current = session(state, "summarization")
                current.update(status="failed", error=error_text(error))
                current.pop("preview_outline", None)
                current.pop("preview_summary", None)
        finally:
            state["busy"] = None
        yield render(state, self.client)

    @staticmethod
    def _answer_notice(status):
        if status in {"insufficient_evidence", "unsupported", "insufficient"}:
            return "The material does not contain enough evidence to answer this question."
        if status == "off_topic":
            return "This question is outside the selected material."
        if status == "partial":
            return "Only part of the question is supported. See the answer for missing information."
        return "Answer complete. Ask a follow-up question."

    async def chat(self, question, state):
        state = deepcopy(state or fresh_state())
        try:
            self._ready(state)
            if not question.strip():
                raise BackendError("Enter a question about the selected material.")
            current = session(state, "qa")
            if current.get("status") in {"running", "failed", "interrupted_for_feedback"}:
                raise BackendError("Refresh the conversation or start a new conversation before continuing.", 409)
            state.update(busy="qa", notice="")
            current.update(status="running", error=None, pending={"question": question.strip(), "text": ""})
            yield render(state, self.client)
            async for event in self.client.qa_events(asset_id=state["selected_asset"], question=question.strip(), thread_id=current.get("thread_id")):
                current["thread_id"] = event["thread_id"]
                data, kind = event.get("data", {}), event["event"]
                if kind == "text_delta":
                    current["pending"]["text"] += data.get("text", "")
                elif kind == "completed":
                    pending = current.pop("pending")
                    if not pending.get("text", "").strip():
                        raise BackendError("The backend completed without a usable answer. Refresh the conversation.", 502)
                    result = current.setdefault("result", {})
                    result.setdefault("messages", []).extend([
                        {"role": "user", "content": pending["question"]},
                        {"role": "assistant", "content": pending["text"]},
                    ])
                    result.update(answer_citations=data.get("citations", []), answer_status=data.get("answer_status"))
                    current.update(status="completed", answer_notice=self._answer_notice(data.get("answer_status")))
                elif kind == "failed":
                    raise BackendError(data.get("detail", "The answer could not be completed."))
                elif kind == "awaiting_feedback":
                    raise BackendError("The conversation did not finish. Refresh it or start a new conversation.", 409)
                yield render(state, self.client)
        except BackendError as error:
            state["notice"] = error_text(error)
            if state.get("busy") == "qa":
                current = session(state, "qa")
                current.pop("pending", None)
                current.update(status="failed", error=error_text(error))
        finally:
            state["busy"] = None
        yield render(state, self.client)

    async def new_chat(self, state):
        state = deepcopy(state or fresh_state())
        if state.get("busy"):
            state["notice"] = "Wait for the current request to finish."
        else:
            session(state, "qa").clear()
            state["notice"] = "A new conversation will begin with your next question."
        return render(state, self.client)

    async def resume(self, workflow, thread_id, state):
        state = deepcopy(state or fresh_state())
        try:
            self._ready(state)
            value = await self.client.get_session(workflow, thread_id.strip())
            if value.get("asset_id") != state.get("selected_asset"):
                raise BackendError("This session belongs to another material. Select that material first.", 409)
            current = session(state, workflow)
            current.clear()
            current.update(value)
            state["notice"] = "Session restored from the backend."
        except BackendError as error:
            state["notice"] = error_text(error)
        return render(state, self.client)
