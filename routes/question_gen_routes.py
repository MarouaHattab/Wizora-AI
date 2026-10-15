import asyncio
import uuid
from fastapi import APIRouter, HTTPException, Body
from typing import Literal
from langgraph.types import Command
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from controllers.GraphState import QuestionGenState
from controllers.QuestionGenGraphController import qg_graph
from chains.contracts import (
    ChainInputError, InsufficientSourceError, ProviderExecutionError,
    StructuredOutputError, TextOutputError,
)
from helper import get_settings
from helper.session_review import acquire_session, accept_feedback, review_revision, session_status
from helper.asset_http import read_asset_text
from helper.workflow_history import normalize_review_state

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
            "message": "Provide feedback or type 'auto' to automatically generate a new question or 'save' to finish",
        },
        "current_state": current_graph_state,
    }
    history = current_graph_state.get("history", [])
    if history and isinstance(history[-1], dict) and history[-1].get("status") == "unchanged":
        response["data_for_feedback"]["revision_notice"] = history[-1].get("reason")
    return response


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
    )
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("question_generation", thread_id)
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            await qg_graph.ainvoke(initial_state, config=config)
            return question_response(thread_id, await qg_graph.aget_state(config))
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        raise HTTPException(status_code=422, detail={"status": "insufficient_source", "reason": exc.reason}) from None
    except ChainInputError:
        raise HTTPException(status_code=422, detail="Invalid question generation input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException):
        raise HTTPException(status_code=502, detail="The question provider returned an invalid structured result") from None
    except ProviderExecutionError:
        raise HTTPException(status_code=502, detail="The question provider failed") from None
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Question generation timed out") from None
    except Exception:
        raise HTTPException(status_code=500, detail="Error invoking question generation graph")
    finally:
        lock.release()


@router.post("/provide_feedback")
async def provide_feedback_to_question_generation(
    thread_id: str = Body(..., embed=True),
    feedback: str = Body(..., embed=True),
    review_id: str | None = Body(None, embed=True),
):
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("question_generation", thread_id)
    try:
        async with asyncio.timeout(settings.GRAPH_TIMEOUT_SECONDS):
            snapshot = await qg_graph.aget_state(config)
            accept_feedback("question_generation", thread_id, snapshot, feedback, review_id)
            await qg_graph.ainvoke(Command(resume={
                "feedback": feedback, "review_id": review_revision(snapshot),
            }), config=config)
            return question_response(thread_id, await qg_graph.aget_state(config))
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        raise HTTPException(status_code=422, detail={"status": "insufficient_source", "reason": exc.reason}) from None
    except ChainInputError:
        raise HTTPException(status_code=422, detail="Invalid question revision input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException):
        raise HTTPException(status_code=502, detail="The question provider returned an invalid structured result") from None
    except ProviderExecutionError:
        raise HTTPException(status_code=502, detail="The question provider failed") from None
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Question revision timed out") from None
    except Exception:
        raise HTTPException(status_code=500, detail="Error resuming question generation graph")
    finally:
        lock.release()
