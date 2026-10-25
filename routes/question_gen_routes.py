import asyncio
import sqlite3
import time
import uuid
from functools import wraps
from typing import Literal

import anyio
from fastapi import APIRouter, HTTPException
from langchain_core.exceptions import OutputParserException
from langgraph.types import Command
from pydantic import Field, ValidationError

from chains import (
    ChainInputError,
    InsufficientSourceError,
    ProviderExecutionError,
    StructuredOutputError,
    TextOutputError,
)
from controllers import QuestionGenState
from helper import get_settings
from helper.asset_http import (
    ReviewFeedbackRequest,
    SingleAssetRequest,
    WorkflowRequest,
    read_asset_text,
    validate_session_asset,
)
from helper.failures import StorageUnavailable
from helper.ingestion_errors import ProcessingError
from helper.provider_status import (
    begin_provider_audit,
    effective_model_info,
    end_provider_audit,
    provider_audit,
)
from helper.question_output import question_json
from helper.question_review import (
    REVIEW_MESSAGE,
    parse_review_action,
    question_assessment,
    validate_question_draft,
)
from helper.session_review import (
    accept_feedback,
    acquire_session,
    review_revision,
    session_status,
    validate_feedback,
)
from helper.sse import public_error
from helper.workflow_history import critic_feedback_value, normalize_review_state, revision_delta
from helper.workflow_metrics import metrics
from helper.workflow_runtime import note_snapshot, require_graph
from schemas import NonemptyText, Question

router = APIRouter()
settings = get_settings()
_MANUAL_EDIT_NOTICE = "Manual edits saved to this draft. Accept / Save to add it to your library."


class QuestionGenerationRequest(SingleAssetRequest):
    question_type: Literal["T/F", "MCQ"]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_count: int = Field(default=1, ge=1, le=20, strict=True)


class QuestionEditRequest(WorkflowRequest):
    thread_id: str
    review_id: NonemptyText
    question: Question


def question_response(thread_id, snapshot):
    status = session_status(snapshot)
    current_graph_state = normalize_review_state(snapshot.values)
    public_questions = [question_json(current_graph_state, current_graph_state)]
    if status == "completed":
        return {
            "thread_id": thread_id,
            "status": status,
            "final_state": current_graph_state,
            "questions": public_questions,
        }
    if status != "interrupted_for_feedback":
        raise HTTPException(
            status_code=500, detail="Question generation did not reach a review or completion"
        )
    response = {
        "thread_id": thread_id,
        "status": status,
        "questions": public_questions,
        "review_id": review_revision(snapshot),
        "data_for_feedback": {
            "generated_question": current_graph_state.get("question"),
            "options": current_graph_state.get("options"),
            "answer": current_graph_state.get("answer"),
            "explanation": current_graph_state.get("explanation"),
            "message": REVIEW_MESSAGE,
            "source_scope": current_graph_state.get("source_scope", ""),
            "context_note": current_graph_state.get("context_note", ""),
            "source_citations": current_graph_state.get("source_citations", []),
            "difficulty": current_graph_state.get("difficulty", "medium"),
            "validation": public_questions[0]["validation"],
            "review_required": public_questions[0]["review_required"],
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


def _start_question_job(request, job_type=None, action=None):
    from helper.asset_catalog import read_job
    from helper.job_results import start_job

    try:
        asset = read_job(request.asset_id)
    except ProcessingError:
        return None  # The normal source validation returns its precise public error.
    return start_job(
        request.asset_id,
        asset.get("source_name", request.asset_id),
        asset.get("source_type", "unknown"),
        job_type or ("questions_mcq" if request.question_type == "MCQ" else "questions_true_false"),
        processing_config={
            "question_type": request.question_type,
            "difficulty": request.difficulty,
            "question_count": request.question_count,
            "action": action,
        },
        source_file_ids=asset.get("source_file_ids"),
        model_info=effective_model_info("QUESTION_GENERATER", settings)
        if job_type is None
        else None,
    )


def _complete_question_job(job_id, response, model_info=None):
    if not job_id:
        return
    from helper.job_results import complete_job

    sessions = response.get("question_sessions", [response])
    details = [
        {
            "thread_id": item.get("thread_id"),
            "review_id": item.get("review_id"),
            "status": item.get("status"),
            "questions": item.get("questions", []),
            "generation_diagnostics": item.get("current_state", item.get("final_state", {})).get(
                "generation_diagnostics", {}
            ),
        }
        for item in sessions
    ]
    state = sessions[0].get("current_state", sessions[0].get("final_state", {})) if sessions else {}
    result = {
        "questions": response.get("questions", []),
        "question_sessions": details,
        "requested_question_count": response.get("requested_question_count", 1),
        "generated_question_count": response.get("generated_question_count", len(details)),
        "content_understanding": state.get("content_understanding", {}),
        "revision_history": state.get("history", []),
        "review_notice": state.get("review_notice"),
    }
    result.update(
        {
            key: response[key]
            for key in (
                "generation_status",
                "missing_question_count",
                "generation_attempts",
                "max_generation_attempts",
                "attempt_history",
                "generation_budget",
            )
            if key in response
        }
    )
    if model_info is not None:
        result["provider_execution"] = model_info
    if response.get("generated_question_count") == 0:
        from helper.job_results import fail_job

        fail_job(
            job_id,
            {
                "category": "question_generation_exhausted",
                "detail": "No valid questions were generated within the bounded request budget.",
            },
            warnings=response.get("warnings", []),
            results=result,
            model_info=model_info,
        )
    else:
        complete_job(job_id, result, warnings=response.get("warnings", []), model_info=model_info)


def _fail_question_job(job_id, error, model_info=None):
    if not job_id:
        return
    from helper.job_results import fail_job

    detail = (
        error.detail
        if isinstance(error, HTTPException)
        else "Question processing could not be completed."
    )
    if not isinstance(detail, str):
        detail = (
            detail.get("reason", "Question processing could not be completed.")
            if isinstance(detail, dict)
            else "Question processing failed."
        )
    executions = provider_audit()
    if model_info is None:
        model_info = (
            {**executions[-1], "executions": executions}
            if executions
            else effective_model_info("QUESTION_GENERATER", settings)
        )
    fail_job(
        job_id,
        {"category": type(error).__name__, "detail": detail},
        results={"provider_executions": executions},
        model_info=model_info,
    )


def _question_provider_audit(function):
    @wraps(function)
    async def audited(request):
        token = begin_provider_audit()
        try:
            return await function(request)
        finally:
            end_provider_audit(token)

    return audited


def _review_model_info(prefixes, purpose):
    infos = [effective_model_info(prefix, settings) for prefix in prefixes]
    executions = provider_audit()
    selected = executions or infos

    def joined(key):
        return " + ".join(dict.fromkeys(str(item[key]) for item in selected if item.get(key)))

    return {
        "provider": joined("provider"),
        "model": joined("model"),
        "requested_provider": " + ".join(
            dict.fromkeys(str(item["requested_provider"]) for item in infos)
        ),
        "requested_model": " + ".join(
            dict.fromkeys(str(item["requested_model"]) for item in infos)
        ),
        "fallback_used": any(item.get("fallback_used", False) for item in selected),
        "execution_verified": bool(executions)
        and all(item.get("execution_verified", False) for item in executions),
        "executions": executions,
        "purpose": purpose,
    }


def _review_job(operation):
    def decorate(function):
        @wraps(function)
        async def audited(request):
            job_id = None
            try:
                snapshot = await _question_graph().aget_state(
                    {"configurable": {"thread_id": request.thread_id}}
                )
                state = snapshot.values or {}
                action = (
                    parse_review_action(getattr(request, "feedback", ""))
                    if operation == "question_refinement"
                    else "manual_edit"
                )
                if state.get("asset_id"):
                    descriptor = QuestionGenerationRequest(
                        asset_id=state["asset_id"],
                        question_type=state["question_type"],
                        difficulty=state.get("difficulty", "medium"),
                    )
                    job_id = await anyio.to_thread.run_sync(
                        _start_question_job, descriptor, operation, action
                    )
                response = await function(request)
                model_info = None
                final = response.get("current_state", response.get("final_state", {}))
                if operation == "question_refinement" and action != "accept":
                    prefixes = ["QUESTION_REFINER"] if action == "refine" else ["QUESTION_REWRITER"]
                    if action == "refine" and final.get("review_model_calls", 0) > 1:
                        prefixes.append("QUESTION_REWRITER")
                    model_info = _review_model_info(
                        prefixes,
                        "critic_and_revision"
                        if len(prefixes) > 1
                        else "critic"
                        if action == "refine"
                        else "revision",
                    )
                    response["model_info"] = model_info
                await anyio.to_thread.run_sync(_complete_question_job, job_id, response, model_info)
                if job_id:
                    response["job_id"] = job_id
                return response
            except Exception as error:
                failed_info = None
                if job_id and operation == "question_refinement" and action != "accept":
                    prefixes = ["QUESTION_REFINER"] if action == "refine" else ["QUESTION_REWRITER"]
                    failed_info = _review_model_info(
                        prefixes, "critic" if action == "refine" else "revision"
                    )
                await anyio.to_thread.run_sync(_fail_question_job, job_id, error, failed_info)
                storage = _storage_error(error)
                if storage is not None:
                    raise storage from None
                raise

        return _question_provider_audit(audited)

    return decorate


def _question_graph():
    return require_graph("question_generation")


async def _record_failure(thread_id, graph, started, exc, request_id):
    snapshot = None
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        snapshot = None
    note_snapshot(
        "question_generation",
        thread_id,
        snapshot,
        started=started,
        error=exc,
        request_id=request_id,
    )


def _storage_error(exc):
    if isinstance(exc, (StorageUnavailable, sqlite3.OperationalError)):
        return HTTPException(status_code=503, detail="Workflow storage is unavailable")
    return None


@router.post("/edit_question")
@_review_job("question_manual_edit")
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
                    raise HTTPException(
                        status_code=409,
                        detail="An interrupted edit must be retried with the same question",
                    )
                # Only the pure interrupt node remains. Complete it without
                # repeating the edit or invoking a generation/revision model.
                edit_committed = True
                await graph.ainvoke(None, config=config)
                snapshot = await graph.aget_state(config)
                note_snapshot(
                    "question_generation",
                    thread_id,
                    snapshot,
                    started=started,
                    request_id=request_id,
                )
                return question_response(thread_id, snapshot)
            validate_feedback(
                "question_generation", thread_id, snapshot, "manual edit", request.review_id
            )
            await validate_session_asset(snapshot.values)
            try:
                question = validate_question_draft(snapshot.values, request.question)
            except (ChainInputError, StructuredOutputError):
                raise HTTPException(
                    status_code=422,
                    detail="Manual edits must preserve the question type and valid answer choices",
                ) from None
            audit_state = {
                **snapshot.values,
                "human_feedback_text": "Manual edit",
                "feedback_review_id": review_revision(snapshot),
                "critic_feedback": None,
            }
            history = revision_delta(
                audit_state, settings, draft=question, status="revised", reason="Edited by user."
            )
            delta = {
                **question.model_dump(),
                **history,
                "validation": question_assessment(
                    question,
                    snapshot.values.get("source_context") or snapshot.values.get("context", ""),
                ),
                "review_required": True,
                "revision_requirements": snapshot.values.get("revision_requirements", []),
                "feedback": "",
                "human_feedback_text": "",
                "feedback_review_id": request.review_id,
                "critic_feedback": None,
                "workflow_action": None,
                "auto_revision_count": 0,
                "review_model_calls": 0,
                "revision_limit_reached": False,
                "review_notice": _MANUAL_EDIT_NOTICE,
            }
            # Re-enter the existing review node from its unconditional predecessor.
            # Updating as human_feedback would route using its previous action.
            await graph.aupdate_state(config, delta, as_node="question_gen")
            edit_committed = True
            await graph.ainvoke(None, config=config)
            snapshot = await graph.aget_state(config)
            note_snapshot(
                "question_generation", thread_id, snapshot, started=started, request_id=request_id
            )
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except TimeoutError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(
            status_code=504,
            detail="Saving question edits timed out. Reload the session before retrying.",
        ) from None
    except Exception as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        storage = _storage_error(exc)
        if storage is not None:
            if edit_committed:
                storage.detail = (
                    "Question edits were stored, but review could not reopen. Retry the same edits."
                )
            raise storage from None
        raise HTTPException(status_code=500, detail="Unable to save question edits") from None
    finally:
        lock.release()


async def _generate_question_session(
    request: QuestionGenerationRequest,
    *,
    additional_previous=(),
):
    thread_id = str(uuid.uuid4())
    context = await read_asset_text(request.asset_id, for_question=True)

    from helper.session_store import completed_sessions

    graph = _question_graph()
    saved = await anyio.to_thread.run_sync(
        completed_sessions, "question_generation", request.asset_id
    )
    previous_questions = []
    for item in saved:
        snapshot = await graph.aget_state({"configurable": {"thread_id": item["thread_id"]}})
        if (
            session_status(snapshot) == "completed"
            and snapshot.values.get("asset_id") == request.asset_id
        ):
            question = snapshot.values.get("question")
            if isinstance(question, str) and question.strip():
                previous_questions.append(question)
    previous_questions.extend(additional_previous)

    initial_state = QuestionGenState(
        asset_id=request.asset_id,
        context=context,
        question_type=request.question_type,
        difficulty=request.difficulty,
        require_assessed_difficulty=True,
        previous_questions=previous_questions,
        question="",
        options=None,
        answer="",
        explanation="",
        history=[],
        feedback="",
        human_feedback_text="",
        critic_feedback=None,
        feedback_review_id=None,
        revision_requirements=[],
        state_version=4,
        auto_revision_count=0,
        review_model_calls=0,
        revision_limit_reached=False,
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
            note_snapshot(
                "question_generation", thread_id, snapshot, started=started, request_id=request_id
            )
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(
            status_code=422,
            detail={
                "status": "insufficient_source",
                "reason": exc.reason,
                "generation_diagnostics": getattr(exc, "generation_diagnostics", {}),
            },
        ) from None
    except ChainInputError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail="Invalid question generation input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException) as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(
            status_code=502,
            detail={
                "status": "invalid_structure",
                "reason": "The question provider returned an invalid structured result",
                "generation_diagnostics": getattr(exc, "generation_diagnostics", {}),
            },
        ) from None
    except ProviderExecutionError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        failure = public_error(exc)
        quota = failure["category"] == "quota"
        raise HTTPException(
            status_code=429 if quota else 502,
            detail={
                "status": "quota_exceeded" if quota else "provider_unavailable",
                "reason": failure["detail"] if quota else str(exc),
                "generation_diagnostics": getattr(exc, "generation_diagnostics", {}),
            },
        ) from None
    except TimeoutError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=504, detail="Question generation timed out") from None
    except Exception as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        storage = _storage_error(exc)
        if storage is not None:
            raise storage from None
        raise HTTPException(
            status_code=500, detail="Error invoking question generation graph"
        ) from None
    finally:
        lock.release()


@router.post("/start_session")
@_question_provider_audit
async def start_question_generation_session(request: QuestionGenerationRequest):
    """Generate independent review sessions; retain the legacy single-item response."""
    job_id = await anyio.to_thread.run_sync(_start_question_job, request)
    from helper.content_understanding import question_near_duplicate

    sessions, previous, warnings, attempts = [], [], [], []
    configured_retries = getattr(settings, "QUESTION_BATCH_RETRY_LIMIT", None)
    retry_limit = (
        min(3, request.question_count) if configured_retries is None else configured_retries
    )
    max_attempts = request.question_count + retry_limit
    overall_seconds = min(
        settings.GRAPH_TIMEOUT_SECONDS * request.question_count,
        settings.FRONTEND_TIMEOUT_SECONDS * 0.9,
    )
    configured_seconds = getattr(settings, "QUESTION_OVERALL_TIMEOUT_SECONDS", None)
    if configured_seconds is not None:
        overall_seconds = min(overall_seconds, configured_seconds)
    budget = {
        "candidate_attempt_limit": getattr(settings, "QUESTION_CANDIDATE_ATTEMPT_LIMIT", None),
        "candidate_default_minimum": 3,
        "batch_retry_limit": retry_limit,
        "max_session_attempts": max_attempts,
        "overall_timeout_seconds": overall_seconds,
        "per_session_timeout_seconds": settings.GRAPH_TIMEOUT_SECONDS,
    }
    deadline = asyncio.get_running_loop().time() + overall_seconds
    try:
        while len(sessions) < request.question_count and len(attempts) < max_attempts:
            attempt = {
                "attempt": len(attempts) + 1,
                "outcome": "failed",
                "generated_count": len(sessions),
                "remaining_count": request.question_count - len(sessions),
            }
            attempts.append(attempt)
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                response = await asyncio.wait_for(
                    _generate_question_session(request, additional_previous=list(previous)),
                    remaining,
                )
            except TimeoutError:
                attempt.update(outcome="deadline_reached", error_category="timeout")
                warnings.append(
                    f"The request deadline was reached after {len(sessions)} of {request.question_count} questions. Completed drafts remain reviewable."
                )
                break
            except HTTPException as error:
                if (
                    error.status_code == 429
                    and isinstance(error.detail, dict)
                    and error.detail.get("status") == "quota_exceeded"
                ):
                    attempt.update(outcome="quota_exceeded", error_category="http_429")
                    warnings.append(error.detail["reason"])
                    break  # Repeating a quota failure cannot fill the remaining count.
                source_exhausted = (
                    error.status_code == 422
                    and isinstance(error.detail, dict)
                    and error.detail.get("status") == "insufficient_source"
                )
                if error.status_code not in {502, 504} and not source_exhausted:
                    raise
                attempt.update(
                    outcome="invalid_result"
                    if error.status_code == 502
                    else "source_exhausted"
                    if source_exhausted
                    else "timeout",
                    error_category=f"http_{error.status_code}",
                )
                if isinstance(error.detail, dict):
                    if error.detail.get("generation_diagnostics"):
                        attempt["generation_diagnostics"] = error.detail["generation_diagnostics"]
                    if error.detail.get("status") == "provider_unavailable":
                        attempt["outcome"] = "provider_unavailable"
                        reason = error.detail.get("reason")
                        if isinstance(reason, str) and reason not in warnings:
                            warnings.append(reason)
                    elif error.detail.get("status") == "insufficient_source":
                        reason = error.detail.get("reason")
                        if isinstance(reason, str) and reason not in warnings:
                            warnings.append(f"ℹ️ Material notice: {reason}")
                continue
            state = response.get("current_state", response.get("final_state", {}))
            try:
                candidate = validate_question_draft(
                    {"question_type": request.question_type, "difficulty": state.get("difficulty", request.difficulty)},
                    {key: state[key] for key in Question.model_fields if key in state},
                )
                if state.get("asset_id", request.asset_id) != request.asset_id:
                    raise StructuredOutputError("The generated source does not match the request")
            except (ChainInputError, StructuredOutputError):
                attempt["outcome"] = "invalid_contract"
                continue
            if any(question_near_duplicate(candidate.question, old) for old in previous):
                attempt["outcome"] = "duplicate_rejected"
                continue
            sessions.append(response)
            understanding = state.get("content_understanding", {})
            warnings.extend(
                message
                for message in understanding.get("classification_warnings", [])
                if message not in warnings
            )
            previous.append(candidate.question)
            attempt.update(
                outcome="retained",
                generated_count=len(sessions),
                remaining_count=request.question_count - len(sessions),
            )
        missing = request.question_count - len(sessions)
        if missing:
            warnings.append(
                f"Partial result: generated {len(sessions)} of {request.question_count} requested questions; {missing} still missing after {len(attempts)} bounded attempts. Existing drafts remain reviewable."
            )
        result = {
            **(sessions[0] if sessions else {"thread_id": None, "status": "failed"}),
            "question_sessions": sessions,
            "questions": [
                question for session in sessions for question in session.get("questions", [])
            ],
            "requested_question_count": request.question_count,
            "generated_question_count": len(sessions),
            "generation_status": "partial" if missing else "complete",
            "missing_question_count": missing,
            "generation_attempts": len(attempts),
            "max_generation_attempts": max_attempts,
            "attempt_history": attempts,
            "generation_budget": budget,
            "warnings": warnings,
        }
        model_info = effective_model_info("QUESTION_GENERATER", settings)
        result["model_info"] = model_info
        await anyio.to_thread.run_sync(_complete_question_job, job_id, result, model_info)
        if job_id:
            result["job_id"] = job_id
        return result
    except Exception as error:
        await anyio.to_thread.run_sync(_fail_question_job, job_id, error)
        raise


@router.post("/provide_feedback")
@_review_job("question_refinement")
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
            await graph.ainvoke(
                Command(
                    resume={
                        "feedback": feedback,
                        "review_id": review_revision(snapshot),
                    }
                ),
                config=config,
            )
            snapshot = await graph.aget_state(config)
            note_snapshot(
                "question_generation", thread_id, snapshot, started=started, request_id=request_id
            )
            return question_response(thread_id, snapshot)
    except HTTPException:
        raise
    except InsufficientSourceError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(
            status_code=422, detail={"status": "insufficient_source", "reason": exc.reason}
        ) from None
    except ChainInputError as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(status_code=422, detail="Invalid question revision input") from None
    except (StructuredOutputError, TextOutputError, ValidationError, OutputParserException) as exc:
        await _record_failure(thread_id, graph, started, exc, request_id)
        raise HTTPException(
            status_code=502, detail="The question provider returned an invalid structured result"
        ) from None
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
        raise HTTPException(
            status_code=500, detail="Error resuming question generation graph"
        ) from None
    finally:
        lock.release()
