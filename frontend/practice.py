import logging
from uuid import uuid4

import gradio as gr

from frontend.client import BackendError
from frontend.presentation import markdown_text
from helper.quiz import answer_question, move_question, start_quiz, student_question, submit_quiz

logger = logging.getLogger(__name__)


class PracticeUI:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def _asset(workflow):
        return workflow.get("asset_id", "") if isinstance(workflow, dict) else ""

    @staticmethod
    def _cursor(quiz):
        return f"{quiz['attempt_id']}:{quiz['index']}:{int(quiz['submitted'])}"

    @staticmethod
    def render(quiz=None, notice=""):
        if not quiz:
            return (
                None,
                notice or "Save questions for the active source, then start a practice test.",
                "",
                gr.update(choices=[], value=None, interactive=False),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",
            )
        prompt = student_question(quiz)
        index, total = quiz["index"], len(quiz["questions"])
        submitted = quiz["submitted"]
        status = f"Question {index + 1} of {total} · {len(quiz['answers'])} answered."
        status += (
            " Review your answers below." if submitted else " Submit the test to see your score."
        )
        review = score_text = ""
        if submitted:
            question = quiz["questions"][index]
            review = (
                "**Your answer:** "
                + markdown_text(quiz["answers"].get(index, "Unanswered"))
                + "\n\n**Correct answer:** "
                + markdown_text(question["answer"])
                + "\n\n"
                + markdown_text(question["explanation"])
            )
            score = quiz["score"]
            score_text = (
                f"**Score: {score['correct']} / {score['total']} ({score['percentage']:.1f}%)**"
                f"\n\nCorrect: {score['correct']} · Incorrect: {score['incorrect']}"
                f" · Unanswered: {score['unanswered']}"
            )
        return (
            quiz,
            notice or status,
            prompt["question"],
            gr.update(
                choices=prompt["choices"],
                value=quiz["answers"].get(index),
                interactive=not submitted,
            ),
            review,
            score_text,
            gr.update(interactive=index > 0),
            gr.update(interactive=index < total - 1),
            gr.update(interactive=not submitted),
            PracticeUI._cursor(quiz),
        )

    def _active(self, workflow, quiz):
        if not quiz:
            raise ValueError("Start a practice test with saved questions first.")
        if quiz.get("asset_id") != self._asset(workflow):
            raise ValueError(
                "The active source changed. Start a new practice test for this source."
            )
        return quiz

    async def start(self, workflow):
        try:
            asset_id = self._asset(workflow)
            if not asset_id:
                raise ValueError("Upload or select a source before starting a practice test.")
            payload = await self.client.saved_questions(asset_id)
            if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
                raise BackendError("The backend returned an invalid saved-question library", 502)
            records = [
                item
                for item in payload["questions"]
                if isinstance(item, dict) and item.get("asset_id") == asset_id
            ]
            quiz = start_quiz(records)
            quiz["asset_id"] = asset_id
            quiz["attempt_id"] = uuid4().hex
            return self.render(quiz)
        except (BackendError, ValueError) as error:
            return self.render(notice=str(error))
        except Exception:
            logger.exception("Saved-question practice could not start")
            return self.render(
                notice="Saved questions could not be loaded. Check the backend and try again."
            )

    def sync_source(self, workflow, quiz):
        if quiz and quiz.get("asset_id") != self._asset(workflow):
            return self.render(
                notice="The active source changed. Start a new practice test for this source."
            )
        return tuple(gr.skip() for _ in range(10))

    def _action(self, answer, workflow, quiz, action, cursor=None):
        # Stateful inputs are read when Gradio executes an event. The hidden
        # non-State cursor was captured in the browser with the displayed
        # answer, so a queued old event cannot edit a newer question/attempt.
        if quiz and cursor is not None and cursor != self._cursor(quiz):
            return tuple(gr.skip() for _ in range(10))
        try:
            self._active(workflow, quiz)
        except ValueError as error:
            return self.render(notice=str(error))
        try:
            # Navigation/submit also save the displayed selection, including a
            # click queued before Radio.input has finished. Reviews are frozen.
            updated = quiz if quiz["submitted"] else answer_question(quiz, answer)
            if action == "submit":
                updated = submit_quiz(updated)
            elif action in {"next", "previous"}:
                updated = move_question(updated, action)
            return self.render(updated)
        except ValueError as error:
            return self.render(quiz, str(error))

    def choose(self, answer, workflow, quiz, cursor=None):
        result = list(self._action(answer, workflow, quiz, "answer", cursor))
        if result[0] is not None:
            # The user may already have selected a newer answer while this
            # callback was queued. Save state without overwriting local input.
            result[3] = gr.skip()
        return tuple(result)

    def previous(self, answer, workflow, quiz, cursor=None):
        return self._action(answer, workflow, quiz, "previous", cursor)

    def next(self, answer, workflow, quiz, cursor=None):
        return self._action(answer, workflow, quiz, "next", cursor)

    def submit(self, answer, workflow, quiz, cursor=None):
        return self._action(answer, workflow, quiz, "submit", cursor)
