import logging

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from chains import (
    ChainInputError,
    InsufficientSourceError,
    ProviderExecutionError,
    StructuredOutputError,
    get_task_chain,
)
from helper import get_settings
from helper.content_context import prepare_source
from helper.content_understanding import (
    configured_understanding,
    grounding_issue,
    originality_issue,
    project_understanding,
    question_near_duplicate,
    understand_content,
)
from helper.question_review import (
    DIFFICULTY_RUBRIC,
    REVIEW_MESSAGE,
    accepted_feedback,
    invalid_critic_feedback,
    needs_revision,
    parse_review_action,
    question_assessment,
    refine_budget_exhausted,
    structural_feedback,
    validate_question_draft,
)
from helper.workflow_history import (
    critic_feedback_value,
    human_feedback_value,
    revision_context,
    revision_delta,
)
from helper.workflow_metrics import metrics

from .GraphState import QuestionGenState

settings = get_settings()
logger = logging.getLogger("wizora.question")


def current_question(state):
    return validate_question_draft(state)


def _generation_inputs(state, context=None, previous=None):
    if not state["context"].strip():
        raise InsufficientSourceError("The source contains no educational material.")
    return {
        "context": context or state["context"],
        "question_type": state["question_type"],
        "difficulty": state.get("difficulty", "medium"),
        "previous_questions": previous
        if previous is not None
        else state.get("previous_questions", []),
    }


def _generation_delta(result, state, override_difficulty=None):
    if result.status == "insufficient_source":
        raise InsufficientSourceError(result.reason)
    eval_state = dict(state)
    target_difficulty = override_difficulty or getattr(result.question, "difficulty", None)
    if target_difficulty in {"easy", "medium", "hard"}:
        eval_state["difficulty"] = target_difficulty
        if hasattr(result.question, "difficulty"):
            result.question.difficulty = target_difficulty
    question = validate_question_draft(eval_state, result.question)
    return {
        **question.model_dump(),
        "workflow_action": None,
        "auto_revision_count": 0,
        "review_model_calls": 0,
        "revision_limit_reached": False,
        "review_notice": None,
    }


def _generation_attempts(state, prepared):
    previous = state.get("previous_questions") or []
    parts = prepared["source_chunks"]
    order = [(len(previous) + offset) % len(parts) for offset in range(len(parts))]
    if state.get("difficulty") == "hard":
        # Multi-premise questions need connected explanatory evidence. Advancing
        # blindly by question count can strand every retry in short frame/speech
        # fragments. Prefer coherent explanations, while retaining every part as
        # a fallback and excluding all previous stems in the model and reviewer.
        understanding = prepared.get("content_understanding", {})
        strengths = {
            index: sum(
                len(block["text"].encode("utf-8"))
                for block in project_understanding(part["context"], understanding)[
                    "explanatory_content"
                ]
            )
            for index, part in enumerate(parts)
        }
        order.sort(key=lambda index: strengths[index], reverse=True)
    for attempt in range(_candidate_attempt_limit(prepared)):
        index = order[attempt % len(order)]
        yield index, parts[index], _generation_inputs(state, parts[index]["context"], previous)


def _candidate_attempt_limit(prepared):
    configured = getattr(settings, "QUESTION_CANDIDATE_ATTEMPT_LIMIT", None)
    return configured if configured is not None else max(3, len(prepared["source_chunks"]))


def _candidate_issue(state, result, understanding):
    if result.status != "generated":
        return "Choose another educational concept supported by the source. " + (
            result.reason or ""
        )
    eval_state = dict(state)
    if hasattr(result.question, "difficulty") and result.question.difficulty in {"easy", "medium", "hard"}:
        eval_state["difficulty"] = result.question.difficulty
    question = validate_question_draft(eval_state, result.question)
    return originality_issue(
        question.question, understanding, state.get("previous_questions") or []
    ) or grounding_issue(question, understanding)


def _candidate_review(state, result, part):
    if result.status != "generated" or any(
        question_near_duplicate(result.question.question, old)
        for old in state.get("previous_questions") or []
    ):
        return None
    eval_state = dict(state)
    if hasattr(result.question, "difficulty") and result.question.difficulty in {"easy", "medium", "hard"}:
        eval_state["difficulty"] = result.question.difficulty
    question = validate_question_draft(eval_state, result.question)
    return {
        "context": part["context"],
        "previous_questions": state.get("previous_questions") or [],
        **question.model_dump(),
    }


def _candidate_assessment(state, result, prepared, part, critic):
    return question_assessment(
        result.question,
        part["context"],
        critic,
        understanding=prepared["content_understanding"],
        previous=state.get("previous_questions") or [],
    )


def _record_rejection(attempt, feedback, failed_checks):
    # Persist the bounded reason/checks, never the candidate's raw question object.
    attempt.update(rejection_reason=feedback[:1200], failed_checks=list(failed_checks))
    logger.info(
        "question candidate rejected attempt=%s source_part=%s checks=%s",
        attempt.get("attempt"),
        attempt.get("source_part"),
        ",".join(failed_checks),
    )


def _review_rejection_feedback(state, result, prepared, part, critic):
    validation = _candidate_assessment(state, result, prepared, part, critic)
    failed = [name for name, status in validation["checks"].items() if status == "failed"]
    if (
        state.get("require_assessed_difficulty")
        and validation["checks"]["difficulty"] != "passed"
        and "difficulty" not in failed
    ):
        failed.append("difficulty")
    blocking = (
        [issue for issue in critic.issues if issue.severity != "low"] if critic is not None else []
    )
    if not failed and blocking:
        failed = ["critic_review"]
    # A positive critic summary must not override a local failed check. Its
    # actionable issue hints are retained separately from local warnings.
    warnings = validation["warnings"]
    hints = " | ".join(
        f"{issue.code[:48]}: {issue.correction_hint[:220]}" for issue in blocking[:3]
    )
    difficulty = state.get("difficulty", "medium")
    parts = [
        f"Candidate rejected. Requested difficulty: {difficulty}; assessed: {validation['assessed_difficulty']}; cognitive level: {validation['cognitive_level']}.",
        "Required rubric: " + DIFFICULTY_RUBRIC[difficulty],
        "Failed checks: " + ", ".join(failed or ["source_validation"])[:120],
    ]
    if warnings:
        parts.append("Local validation warnings: " + " | ".join(warnings)[:300])
    if hints:
        parts.append("Critic correction hints: " + hints[:360])
    elif critic is not None and not critic.acceptable:
        parts.append("Critic correction: " + critic.feedback[:250])
    return "\n".join(parts)[:1200], failed


def _source_rejection_checks(state, result, understanding):
    if result.status != "generated":
        return ["source_support"]
    if originality_issue(
        result.question.question, understanding, state.get("previous_questions") or []
    ):
        return ["originality"]
    return ["factual_support"]


def _prepared_generation_delta(state, result, prepared, index, part, critic):
    if (
        critic is not None
        and not critic.acceptable
        and any(issue.severity != "low" for issue in critic.issues)
    ):
        return None
    validation = _candidate_assessment(state, result, prepared, part, critic)

    # Check if fundamental factual/source validity checks failed
    other_failed = [
        name for name, status in validation["checks"].items()
        if status == "failed" and name != "difficulty"
    ]
    if other_failed:
        return None

    # Adaptive Difficulty handling:
    difficulty_check = validation["checks"].get("difficulty")
    adapted_difficulty = None
    adaptive_note = ""

    if state.get("require_assessed_difficulty") and difficulty_check != "passed":
        assessed = validation.get("assessed_difficulty")
        if assessed in {"easy", "medium"} and state.get("difficulty") == "hard":
            adapted_difficulty = assessed
            adaptive_note = (
                f" (Note: Adapted to '{assessed}' difficulty because the source passage "
                f"lacks multiple interacting premises for '{state.get('difficulty')}')."
            )
            validation["warnings"].append(
                f"Difficulty adapted from {state.get('difficulty')} to {assessed}: source evidence supports {assessed} demand."
            )
            validation["checks"]["difficulty"] = "passed"
        else:
            return None
    elif difficulty_check == "failed":
        assessed = validation.get("assessed_difficulty")
        if assessed in {"easy", "medium", "hard"}:
            adapted_difficulty = assessed
            validation["checks"]["difficulty"] = "passed"
        else:
            return None

    count = len(prepared["source_chunks"])
    note = prepared["context_note"]
    if count > 1:
        note += f" This question uses source part {index + 1} of {count}; other parts are retained for subsequent questions."
    if adaptive_note:
        note += adaptive_note

    delta = _generation_delta(result, state, override_difficulty=adapted_difficulty)
    return {
        **prepared,
        **delta,
        "source_context": part["context"],
        "source_citations": part["citations"],
        "context_note": note,
        "critic_feedback": critic,
        "previous_questions": state.get("previous_questions") or [],
        "validation": validation,
        "review_required": True if adapted_difficulty else validation["status"] != "validated",
    }


def _generation_diagnostics(history, repairs, max_attempts=None):
    return {
        "attempts": len(history),
        "structured_repair_attempts": repairs,
        "attempt_history": list(history),
        "max_candidate_attempts": max_attempts,
    }


def _prepared_understanding(state, prepared, config):
    context = "\n\n".join(part["context"] for part in prepared["source_chunks"])
    if state.get("asset_id"):
        return configured_understanding(
            context, settings, chain_factory=get_task_chain, config=config
        )
    return understand_content(context)  # Legacy inline callers never dispatch another provider.


def _generation_exhausted(prepared, history, repairs):
    if history and all(item["outcome"] == "invalid_structure" for item in history):
        error = StructuredOutputError(
            "All bounded candidate attempts returned invalid structured outputs"
        )
    else:
        covered = len({item["source_part"] for item in history})
        reasons = [item.get("rejection_reason") for item in history if item.get("rejection_reason")]
        detail_hint = f" ({reasons[-1]})" if reasons else ""
        error = InsufficientSourceError(
            f"The selected material is insufficient to generate a validated question after {len(history)} attempt(s) across {covered} source part(s).{detail_hint} Try selecting another source section or choosing an easier difficulty."
        )
    error.generation_diagnostics = _generation_diagnostics(
        history, repairs, _candidate_attempt_limit(prepared)
    )
    return error


def _review_generated_candidate(state, review, config):
    for attempt in range(2 if state.get("require_assessed_difficulty") else 1):
        try:
            return get_task_chain("question_review", settings).invoke(review, config=config)
        except StructuredOutputError:
            return None  # A malformed assessment remains unverified; do not label it passed.
        except ProviderExecutionError:
            if state.get("require_assessed_difficulty"):
                if attempt == 0:
                    continue  # Keep this candidate instead of spending quota regenerating it.
                raise
    return None


async def _review_generated_candidate_stream(state, review, config):
    for attempt in range(2 if state.get("require_assessed_difficulty") else 1):
        try:
            return await get_task_chain("question_review", settings).ainvoke(review, config=config)
        except StructuredOutputError:
            return None
        except ProviderExecutionError:
            if state.get("require_assessed_difficulty"):
                if attempt == 0:
                    continue
                raise
    return None


def QuestionGenerator(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = prepare_source(state, settings)
    understanding = _prepared_understanding(state, prepared, config)
    prepared["content_understanding"] = understanding
    feedback, history, repairs = "", [], 0
    for index, part, inputs in _generation_attempts(state, prepared):
        inputs.update(
            content_understanding=project_understanding(part["context"], understanding),
            validation_feedback=feedback,
        )
        attempt = {
            "attempt": len(history) + 1,
            "source_part": index + 1,
            "outcome": "candidate_rejected",
        }
        history.append(attempt)
        try:
            result = get_task_chain("question_generation", settings).invoke(inputs, config=config)
        except StructuredOutputError as rejected:
            attempt["repair_attempts"] = getattr(rejected, "repair_attempts", 0)
            repairs += attempt["repair_attempts"]
            attempt["outcome"] = "invalid_structure"
            feedback = "The prior candidate remained malformed after bounded schema repair. Return a complete schema-valid question with the same requested type and difficulty."
            _record_rejection(attempt, feedback, ["structure"])
            continue
        except ProviderExecutionError as rejected:
            attempt.update(
                outcome="provider_unavailable",
                repair_attempts=getattr(rejected, "repair_attempts", 0),
            )
            repairs += attempt["repair_attempts"]
            rejected.generation_diagnostics = _generation_diagnostics(
                history, repairs, _candidate_attempt_limit(prepared)
            )
            raise
        repairs += getattr(result, "_repair_attempts", 0)
        feedback = _candidate_issue(state, result, understanding) or ""
        if feedback:
            _record_rejection(
                attempt, feedback, _source_rejection_checks(state, result, understanding)
            )
            continue
        review = _candidate_review(state, result, part)
        if review is None:
            feedback = "The candidate repeats a previous question; choose another supported original assessment."
            _record_rejection(attempt, feedback, ["originality"])
            continue
        critic = _review_generated_candidate(state, review, config)
        delta = _prepared_generation_delta(state, result, prepared, index, part, critic)
        if delta is not None:
            attempt["outcome"] = "validated" if not delta["review_required"] else "review_required"
            delta["generation_diagnostics"] = _generation_diagnostics(
                history, repairs, _candidate_attempt_limit(prepared)
            )
            return delta
        feedback, failed_checks = _review_rejection_feedback(state, result, prepared, part, critic)
        _record_rejection(attempt, feedback, failed_checks)
    raise _generation_exhausted(prepared, history, repairs)


async def QuestionGeneratorStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    import anyio

    prepared = await anyio.to_thread.run_sync(prepare_source, state, settings)
    understanding = await anyio.to_thread.run_sync(_prepared_understanding, state, prepared, config)
    prepared["content_understanding"] = understanding
    feedback, history, repairs = "", [], 0
    for index, part, inputs in _generation_attempts(state, prepared):
        inputs.update(
            content_understanding=project_understanding(part["context"], understanding),
            validation_feedback=feedback,
        )
        attempt = {
            "attempt": len(history) + 1,
            "source_part": index + 1,
            "outcome": "candidate_rejected",
        }
        history.append(attempt)
        try:
            result = await get_task_chain("question_generation", settings).ainvoke(
                inputs, config=config
            )
        except StructuredOutputError as rejected:
            attempt["repair_attempts"] = getattr(rejected, "repair_attempts", 0)
            repairs += attempt["repair_attempts"]
            attempt["outcome"] = "invalid_structure"
            feedback = "The prior candidate remained malformed after bounded schema repair. Return a complete schema-valid question with the same requested type and difficulty."
            _record_rejection(attempt, feedback, ["structure"])
            continue
        except ProviderExecutionError as rejected:
            attempt.update(
                outcome="provider_unavailable",
                repair_attempts=getattr(rejected, "repair_attempts", 0),
            )
            repairs += attempt["repair_attempts"]
            rejected.generation_diagnostics = _generation_diagnostics(
                history, repairs, _candidate_attempt_limit(prepared)
            )
            raise
        repairs += getattr(result, "_repair_attempts", 0)
        feedback = _candidate_issue(state, result, understanding) or ""
        if feedback:
            _record_rejection(
                attempt, feedback, _source_rejection_checks(state, result, understanding)
            )
            continue
        review = _candidate_review(state, result, part)
        if review is None:
            feedback = "The candidate repeats a previous question; choose another supported original assessment."
            _record_rejection(attempt, feedback, ["originality"])
            continue
        critic = await _review_generated_candidate_stream(state, review, config)
        delta = _prepared_generation_delta(state, result, prepared, index, part, critic)
        if delta is not None:
            attempt["outcome"] = "validated" if not delta["review_required"] else "review_required"
            delta["generation_diagnostics"] = _generation_diagnostics(
                history, repairs, _candidate_attempt_limit(prepared)
            )
            return delta
        feedback, failed_checks = _review_rejection_feedback(state, result, prepared, part, critic)
        _record_rejection(attempt, feedback, failed_checks)
    raise _generation_exhausted(prepared, history, repairs)


def _human_feedback_delta(value):
    if isinstance(value, str):
        feedback, review_id = value, None
    elif isinstance(value, dict) and isinstance(value.get("feedback"), str):
        feedback, review_id = value["feedback"], value.get("review_id")
        if review_id is not None and not isinstance(review_id, str):
            raise ValueError("A feedback review_id must be text")
    else:
        raise ValueError("Human feedback must be text or a feedback review object")
    return {
        "feedback": feedback,
        "human_feedback_text": feedback,
        "critic_feedback": None,
        "feedback_review_id": review_id,
        "state_version": 4,
        "workflow_action": parse_review_action(feedback),
        "auto_revision_count": 0,
        "review_model_calls": 0,
        "revision_limit_reached": False,
        "review_notice": None,
    }


def HumanFeedback(state: QuestionGenState) -> dict:
    critic = critic_feedback_value(state)
    payload = {
        "generated_question": state["question"],
        "gernerated_question": state["question"],
        "options": state["options"],
        "answer": state["answer"],
        "explanation": state["explanation"],
        "message": REVIEW_MESSAGE,
        "source_scope": state.get("source_scope", ""),
        "context_note": state.get("context_note", ""),
        "source_citations": state.get("source_citations", []),
        "difficulty": state.get("difficulty", "medium"),
        "validation": state.get("validation"),
        "review_required": state.get("review_required", True),
    }
    if critic is not None:
        payload["critic_verdict"] = critic.verdict
        payload["critic_feedback"] = critic.feedback
    if state.get("review_notice"):
        payload["revision_notice"] = state["review_notice"]
    return _human_feedback_delta(interrupt(payload))


async def HumanFeedbackStream(state: QuestionGenState) -> dict:
    return HumanFeedback(state)


def Router(state: QuestionGenState) -> str:
    return parse_review_action(human_feedback_value(state))


def _review_inputs(state):
    return {
        "context": state.get("source_context") or state["context"],
        "previous_questions": state.get("previous_questions") or [],
        **current_question(state).model_dump(),
    }


def _critic_delta(state, critic, *, model_calls=0, notice=None, limit=False):
    calls = (
        state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0
    ) + model_calls
    delta = {
        "critic_feedback": critic,
        "workflow_action": "refine",
        "review_model_calls": calls,
        "revision_limit_reached": limit,
        "review_notice": notice,
    }
    validation = question_assessment(
        current_question(state),
        state.get("source_context") or state["context"],
        critic,
        understanding=state.get("content_understanding"),
        previous=state.get("previous_questions", ()),
    )
    delta.update(validation=validation, review_required=validation["status"] != "validated")
    if accepted_feedback(critic) or limit:
        reason = notice or (
            "The critic accepted the draft; no rewrite was required."
            if accepted_feedback(critic)
            else "Automatic refinement reached its configured limit."
        )
        delta.update(
            revision_delta(
                state,
                settings,
                draft=current_question(state),
                status="unchanged",
                reason=reason,
                critic=critic,
            )
        )
        delta["review_notice"] = reason
    return delta


def _refiner_precheck(state):
    structural = structural_feedback(state)
    if structural is not None:
        return _critic_delta(
            state, structural, limit=refine_budget_exhausted(state, settings, extra_calls=1)
        )
    if refine_budget_exhausted(state, settings, extra_calls=1):
        return _critic_delta(
            state,
            critic_feedback_value(state) or structural,
            notice="Automatic refinement reached its configured model-call limit.",
            limit=True,
        )
    return None


def _invalid_critic_delta(state):
    return _critic_delta(
        state,
        invalid_critic_feedback(),
        model_calls=1,
        notice="The critic result could not be validated, so the draft was not rewritten.",
        limit=True,
    )


def QuestionRefiner(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _refiner_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = get_task_chain("question_review", settings).invoke(
            _review_inputs(state), config=config
        )
    except (StructuredOutputError, ProviderExecutionError):
        return _invalid_critic_delta(state)
    return _critic_delta(state, response, model_calls=1)


async def QuestionRefinerStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _refiner_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = await get_task_chain("question_review", settings).ainvoke(
            _review_inputs(state), config=config
        )
    except (StructuredOutputError, ProviderExecutionError):
        return _invalid_critic_delta(state)
    return _critic_delta(state, response, model_calls=1)


def route_after_critique(state: QuestionGenState) -> str:
    limited = bool(
        state.get("revision_limit_reached")
        or refine_budget_exhausted(state, settings, extra_calls=1)
    )
    if limited:
        decision = "review"
    else:
        critic = critic_feedback_value(state)
        if critic is None:
            raise StructuredOutputError("Automatic refinement requires a validated critic decision")
        decision = "rewrite" if needs_revision(critic) else "review"
    metrics.increment("graph_route", workflow="question_generation", decision=decision)
    if limited:
        metrics.increment("reflection_stopped", workflow="question_generation")
    logger.info("question review decision=%s limit=%s", decision, limited)
    return decision


def _revision_inputs(state):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    if action not in {"refine", "revise"}:
        action = "revise"
    return {
        "context": state.get("source_context") or state["context"],
        "question_type": state["question_type"],
        "difficulty": state.get("difficulty", "medium"),
        "current_question": current_question(state),
        "action": action,
        "feedback": {
            "action": action,
            "human_feedback": human_feedback_value(state),
            "critic_feedback": critic_feedback_value(state),
        },
        "history": revision_context(state, settings),
    }


def _revision_delta(state, response):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    calls = (
        state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0
    ) + 1
    revisions = (
        state.get("auto_revision_count", 0)
        if type(state.get("auto_revision_count", 0)) is int
        else 0
    )
    if action == "refine":
        revisions += 1
    question = validate_question_draft(state, response.question)
    understanding = state.get("content_understanding") or understand_content(state["context"])
    rejection = originality_issue(
        question.question, understanding, state.get("previous_questions") or []
    ) or grounding_issue(question, understanding)
    if rejection:
        raise StructuredOutputError(rejection)
    if any(
        question_near_duplicate(question.question, old)
        for old in state.get("previous_questions") or []
    ):
        raise StructuredOutputError("The revision repeats an existing saved question")
    notice = None
    limit = action == "refine" and refine_budget_exhausted(
        {**state, "auto_revision_count": revisions, "review_model_calls": calls},
        settings,
    )
    if limit:
        notice = "Automatic refinement reached its configured limit."
    return {
        **question.model_dump(),
        "validation": question_assessment(
            question,
            state.get("source_context") or state["context"],
            None,
            understanding=understanding,
            previous=state.get("previous_questions", ()),
        ),
        "review_required": True,
        "workflow_action": action if action in {"refine", "revise"} else "revise",
        "review_model_calls": calls,
        "auto_revision_count": revisions,
        "revision_limit_reached": limit,
        "review_notice": notice,
        **revision_delta(
            state,
            settings,
            draft=question,
            status=response.status,
            reason=response.reason or notice,
            critic=critic_feedback_value(state),
        ),
    }


def _rewrite_precheck(state):
    if (
        state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    ) != "refine":
        return None
    if not refine_budget_exhausted(state, settings, extra_calls=1):
        return None
    return _critic_delta(
        state,
        critic_feedback_value(state),
        notice="Automatic refinement reached its configured limit.",
        limit=True,
    )


def _unrevised_delta(state, notice):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    if action not in {"refine", "revise"}:
        action = "revise"
    calls = (
        state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0
    ) + 1
    revisions = (
        state.get("auto_revision_count", 0)
        if type(state.get("auto_revision_count", 0)) is int
        else 0
    )
    if action == "refine":
        revisions += 1
    return {
        "workflow_action": action,
        "review_model_calls": calls,
        "auto_revision_count": revisions,
        "revision_limit_reached": action == "refine",
        "review_notice": notice,
        **revision_delta(
            state,
            settings,
            draft=current_question(state),
            status="unchanged",
            reason=notice,
            critic=critic_feedback_value(state),
        ),
    }


def _rewrite(state, response):
    try:
        return _revision_delta(state, response)
    except (ChainInputError, StructuredOutputError):
        return _unrevised_delta(
            state, "The revised question failed validation, so the current question was kept."
        )


def QuestionRewriter(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _rewrite_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = get_task_chain("question_revision", settings).invoke(
            _revision_inputs(state), config=config
        )
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unrevised_delta(
            state, "The revision could not be validated, so the current question was kept."
        )
    return _rewrite(state, response)


async def QuestionRewriterStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _rewrite_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = await get_task_chain("question_revision", settings).ainvoke(
            _revision_inputs(state), config=config
        )
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unrevised_delta(
            state, "The revision could not be validated, so the current question was kept."
        )
    return _rewrite(state, response)


graph = StateGraph(QuestionGenState)
graph.add_node("question_gen", RunnableLambda(QuestionGenerator, afunc=QuestionGeneratorStream))
graph.add_node("question_refiner", RunnableLambda(QuestionRefiner, afunc=QuestionRefinerStream))
graph.add_node("question_rewriter", RunnableLambda(QuestionRewriter, afunc=QuestionRewriterStream))
graph.add_node("human_feedback", RunnableLambda(HumanFeedback, afunc=HumanFeedbackStream))

graph.add_edge(START, "question_gen")
graph.add_edge("question_gen", "human_feedback")
graph.add_conditional_edges(
    "human_feedback",
    Router,
    {"accept": END, "refine": "question_refiner", "revise": "question_rewriter"},
)
graph.add_conditional_edges(
    "question_refiner",
    route_after_critique,
    {"review": "human_feedback", "rewrite": "question_rewriter"},
)
graph.add_edge("question_rewriter", "human_feedback")


def compile_question_graph(checkpointer):
    return graph.compile(checkpointer=checkpointer)
