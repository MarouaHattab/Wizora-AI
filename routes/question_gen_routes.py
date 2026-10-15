import asyncio
import sqlite3
import time
import uuid
from fastapi import APIRouter, HTTPException, Body
from typing import Literal
from langgraph.types import Command
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from controllers.GraphState import QuestionGenState
from chains.contracts import (
    ChainInputError, InsufficientSourceError, ProviderExecutionError,
    StructuredOutputError, TextOutputError,
)
from helper import get_settings
from helper.failures import StorageUnavailable
from helper.session_review import acquire_session, accept_feedback, review_revision, session_status
from helper.asset_http import read_asset_text
from helper.question_review import REVIEW_MESSAGE
from helper.workflow_history import critic_feedback_value, normalize_review_state
from helper.workflow_metrics import metrics
from helper.workflow_runtime import note_snapshot, require_graph

router = APIRouter()
settings = get_settings()

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


@router.post("/start_session")
async def start_question_generation_session(
    asset_id: str = Body(..., embed=True),
    question_type: Literal["T/F", "MCQ"] = Body(..., embed=True)
):
    thread_id = str(uuid.uuid4())
    context = await read_asset_text(asset_id, for_question=True)

    initial_state = QuestionGenState(
        asset_id=asset_id, context=context, question_type=question_type,
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
    thread_id: str = Body(..., embed=True),
    feedback: str = Body(..., embed=True),
    review_id: str | None = Body(None, embed=True),
):
    config = {"configurable": {"thread_id": thread_id}}
    request_id = uuid.uuid4().hex[:12]
    graph = _question_graph()
    lock = acquire_session("question_generation", thread_id)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            snapshot = await graph.aget_state(config)
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
