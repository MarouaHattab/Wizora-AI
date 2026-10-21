import asyncio
import sqlite3
import time
import uuid
from fastapi import APIRouter, HTTPException
from typing import Literal
from langgraph.types import Command
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from controllers.GraphState import QuestionGenState
from chains.contracts import (
    ChainInputError, InsufficientSourceError, ProviderExecutionError,
    StructuredOutputError, TextOutputError,
)
from chains.schemes.QuestionSchem import NonemptyText, Question
from helper import get_settings
from helper.failures import StorageUnavailable
from helper.session_review import acquire_session, accept_feedback, review_revision, session_status, validate_feedback
from helper.asset_http import ReviewFeedbackRequest, SingleAssetRequest, WorkflowRequest, read_asset_text, validate_session_asset
from helper.question_review import REVIEW_MESSAGE, validate_question_draft
from helper.workflow_history import critic_feedback_value, normalize_review_state, revision_delta
from helper.workflow_metrics import metrics
from helper.workflow_runtime import note_snapshot, require_graph

router = APIRouter()
settings = get_settings()
_MANUAL_EDIT_NOTICE = "Manual edits saved to this draft. Accept / Save to add it to your library."


class QuestionGenerationRequest(SingleAssetRequest):
    question_type: Literal["T/F", "MCQ"]


class QuestionEditRequest(WorkflowRequest):
    thread_id: str
    review_id: NonemptyText
    question: Question


def question_response(thread_id, snapshot):
    status = session_status(snapshot)
    current_graph_state = normalize_review_state(snapshot.values)
    if status == "completed":
        return {"thread_id": thread_id, "status": status, "final_state": current_graph_state}
    if status != "interrupted_for_feedback":
        raise HTTPException(status_code=500, detail="Question generation did not reach a review or completion")
    response = {
        "thread_id": thread_id,
        "status": status,
        "review_id": review_revision(snapshot),
        "data_for_feedback": {
            "generated_question": current_graph_state.get("question"),
            "options": current_graph_state.get("options"),
            "answer": current_graph_state.get("answer"),
            "explanation": current_graph_state.get("explanation"),
            "message": REVIEW_MESSAGE,
        },
        "current_state": current_graph_state,
    }
    critic = critic_feedback_value(current_graph_state)
    if critic is not None:
        response["data_for_feedback"]["critic_verdict"] = critic.verdict
        response["data_for_feedback"]["critic_feedback"] = critic.feedback
    if current_graph_state.get("review_notice"):
        response["data_for_feedback"]["revision_notice"] = current_graph_state["review_notice"]
    if current_graph_state.get("revision_limit_reached"):
        response["data_for_feedback"]["revision_limit_reached"] = True
    history = current_graph_state.get("history", [])
    if history and isinstance(history[-1], dict) and history[-1].get("status") == "unchanged":
        response["data_for_feedback"].setdefault("revision_notice", history[-1].get("reason"))
    return response


def _question_graph():
    return require_graph("question_generation")


async def _record_failure(thread_id, graph, started, exc, request_id):
    snapshot = None
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        snapshot = None
    note_snapshot(
        "question_generation", thread_id, snapshot,
        started=started, error=exc, request_id=request_id,
    )


def _storage_error(exc):
    if isinstance(exc, (StorageUnavailable, sqlite3.OperationalError)):
        return HTTPException(status_code=503, detail="Workflow storage is unavailable")
    return None


@router.post("/edit_question")
async def edit_question(request: QuestionEditRequest):
    """Persist a manual draft edit without invoking a model or accepting it."""
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    graph = _question_graph()
    lock = acquire_session("question_generation", thread_id)
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    edit_committed = False
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            snapshot = await graph.aget_state(config)
            pending_edit = (
                session_status(snapshot) in {"running", "failed"}
                and snapshot.next == ("human_feedback",)
                and snapshot.values.get("feedback_review_id") == request.review_id
                and snapshot.values.get("review_notice") == _MANUAL_EDIT_NOTICE
                and snapshot.metadata.get("source") == "update"
            )
            if pending_edit:
                await validate_session_asset(snapshot.values)
                if validate_question_draft(snapshot.values) != request.question:
                    raise HTTPException(status_code=409, detail="An interrupted edit must be retried with the same question")
                # Only the pure interrupt node remains. Complete it without
                # repeating the edit or invoking a generation/revision model.
                edit_committed = True
                await graph.ainvoke(None, config=config)
                snapshot = await graph.aget_state(config)
                note_snapshot("question_generation", thread_id, snapshot, started=started, request_id=request_id)
                return question_response(thread_id, snapshot)
            validate_feedback("question_generation", thread_id, snapshot, "manual edit", request.review_id)
            await validate_session_asset(snapshot.values)
            try:
                question = validate_question_draft(snapshot.values, request.question)
            except (ChainInputError, StructuredOutputError):
                raise HTTPException(status_code=422, detail="Manual edits must preserve the question type and valid answer choices") from None
            audit_state = {
                **snapshot.values, "human_feedback_text": "Manual edit",
                "feedback_review_id": review_revision(snapshot), "critic_feedback": None,
            }
            history = revision_delta(audit_state, settings, draft=question, status="revised", reason="Edited by user.")
            delta = {
                **question.model_dump(), **history,
                "revision_requirements": snapshot.values.get("revision_requirements", []),
                "feedback": "", "human_feedback_text": "", "feedback_review_id": request.review_id,
                "critic_feedback": None, "workflow_action": None,
                "auto_revision_count": 0, "review_model_calls": 0,
                "revision_limit_reached": False,
                "review_notice": _MANUAL_EDIT_NOTICE,
            }
            # Re-enter the existing review node from its unconditional predecessor.
            # Updating as human_feedback would route using its previous action.
            await graph.aupdate_state(config, delta, as_node="question_gen")
            edit_committed = True
            await graph.ainvoke(None, config=config)
            snapshot = await graph.aget_state(config)
            note_snapshot("question_generation", thread_id, snapshot, started=started, request_id=request_id)
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except TimeoutError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=504, detail="Saving question edits timed out. Reload the session before retrying.") from None
    except Exception as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        storage = _storage_error(exc)
        if storage is not None:
            if edit_committed:
                storage.detail = "Question edits were stored, but review could not reopen. Retry the same edits."
            raise storage from None
        raise HTTPException(status_code=500, detail="Unable to save question edits") from None
    finally:
        lock.release()


@router.post("/start_session")
async def start_question_generation_session(
    request: QuestionGenerationRequest,
):
    thread_id = str(uuid.uuid4())
    context = await read_asset_text(request.asset_id, for_question=True)

    initial_state = QuestionGenState(
        asset_id=request.asset_id, context=context, question_type=request.question_type,
        question="", options=None, answer="", explanation="", history=[],
        feedback="", human_feedback_text="", critic_feedback=None,
        feedback_review_id=None, revision_requirements=[], state_version=4,
        auto_revision_count=0, review_model_calls=0, revision_limit_reached=False,
    )
    config = {"configurable": {"thread_id": thread_id}}
    request_id = uuid.uuid4().hex[:12]
    graph = _question_graph()
    lock = acquire_session("question_generation", thread_id)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            await graph.ainvoke(initial_state, config=config)
            snapshot = await graph.aget_state(config)
            note_snapshot("question_generation", thread_id, snapshot, started=started, request_id=request_id)
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail={"status": "insufficient_source", "reason": exc.reason}) from None
    except ChainInputError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail="Invalid question generation input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException) as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=502, detail="The question provider returned an invalid structured result") from None
    except ProviderExecutionError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=502, detail="The question provider failed") from None
    except TimeoutError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=504, detail="Question generation timed out") from None
    except Exception as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        storage = _storage_error(exc)
        if storage is not None:
            raise storage from None
        raise HTTPException(status_code=500, detail="Error invoking question generation graph") from None
    finally:
        lock.release()


@router.post("/provide_feedback")
async def provide_feedback_to_question_generation(
    request: ReviewFeedbackRequest,
):
    thread_id, feedback, review_id = request.thread_id, request.feedback, request.review_id
    config = {"configurable": {"thread_id": thread_id}}
    request_id = uuid.uuid4().hex[:12]
    graph = _question_graph()
    lock = acquire_session("question_generation", thread_id)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            snapshot = await graph.aget_state(config)
            await validate_session_asset(snapshot.values)
            accept_feedback("question_generation", thread_id, snapshot, feedback, review_id)
            metrics.increment("workflow_resume", workflow="question_generation")
            await graph.ainvoke(Command(resume={
                "feedback": feedback, "review_id": review_revision(snapshot),
            }), config=config)
            snapshot = await graph.aget_state(config)
            note_snapshot("question_generation", thread_id, snapshot, started=started, request_id=request_id)
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail={"status": "insufficient_source", "reason": exc.reason}) from None
    except ChainInputError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail="Invalid question revision input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException) as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=502, detail="The question provider returned an invalid structured result") from None
    except ProviderExecutionError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=502, detail="The question provider failed") from None
    except TimeoutError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=504, detail="Question revision timed out") from None
    except Exception as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        storage = _storage_error(exc)
        if storage is not None:
            raise storage from None
        raise HTTPException(status_code=500, detail="Error resuming question generation graph") from None
    finally:
        lock.release()
