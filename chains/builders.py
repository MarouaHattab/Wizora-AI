import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field, ValidationError, create_model
from typing_extensions import NotRequired, TypedDict

from chains.ContentClassificationChain import (
    ContentClassificationPrompt,
    ContentClassificationResult,
)
from chains.contracts import (
    ChainInputError,
    ProviderExecutionError,
    StructuredOutputError,
    requested_question,
    structured_output,
    text_output,
)
from chains.GradeQuestionChain import GradeQuestionPrompt
from chains.QuestionAnswerChain import QuestionAnswerPrompt, prepare_question_answer_inputs
from chains.QuestionGenChain import question_generation_prompt
from chains.QuestionRefinerChain import QuestionRefinerPrompt
from chains.QuestionRewriterChain import QuestionRewriterPrompt
from chains.RAGChains import (
    EvidenceAssessmentInput,
    EvidenceAssessmentPrompt,
    GroundedAnswerInput,
    GroundedAnswerPrompt,
    GroundingCheckInput,
    GroundingCheckPrompt,
    QueryRewriteInput,
    QueryRewritePrompt,
    evidence_assessment_inputs,
    evidence_assessment_result,
    grounded_answer_inputs,
    grounded_answer_result,
    grounding_check_inputs,
    grounding_check_result,
    query_rewrite_inputs,
    query_rewrite_result,
)
from chains.SummarizerGenChain import SummarizerGenPrompt
from chains.SummarizerMainPointChain import SummarizerMainPointPrompt
from chains.SummarizerRewriterChain import SummarizerRewriterPrompt
from chains.SummarizerStructureChain import SummarizerStructurePrompt
from helper.provider_status import ProviderUnavailableError
from schemas import (
    EvidenceDecision,
    Feedback,
    GenerationResult,
    GradeQuestion,
    GroundedAnswer,
    GroundingAssessment,
    MCQGenerationResult,
    MCQQuestion,
    MCQRevisionResult,
    QueryRewrite,
    Question,
    QuestionAssessment,
    ReviewIssue,
    RevisionResult,
    SummaryStructureRevision,
    TrueFalseGenerationResult,
    TrueFalseQuestion,
    TrueFalseRevisionResult,
)

logger = logging.getLogger("wizora.chains")


def _log_structured_failure(schema, error):
    causes, current, codes = set(), error, []
    while current is not None and id(current) not in causes:
        causes.add(id(current))
        if isinstance(current, ValidationError):
            for issue in current.errors(include_input=False, include_context=False):
                message = issue.get("msg", "")
                code = (
                    "missing_unchanged_reason"
                    if "Unchanged results require a" in message
                    else "unexpected_revised_reason"
                    if "Revised results require a null reason" in message
                    else issue.get("type", "validation_error")
                )
                codes.append(code)
        current = current.__cause__ or current.__context__
    logger.warning(
        "Structured output rejected schema=%s issue_codes=%s exception_type=%s",
        schema.__name__ if schema is not None else "text",
        ",".join(dict.fromkeys(codes)) or "output_parser",
        type(error).__name__,
    )


class GenerationInput(TypedDict):
    context: str
    question_type: Literal["MCQ", "T/F"]
    previous_questions: NotRequired[list[str]]
    difficulty: NotRequired[Literal["easy", "medium", "hard"]]
    content_understanding: NotRequired[dict]
    validation_feedback: NotRequired[str]


class ReviewInput(GenerationInput):
    question: str
    options: list[str] | None
    answer: str
    explanation: str


class RevisionInput(GenerationInput):
    current_question: Question | dict
    feedback: str | dict
    history: list[dict]
    action: NotRequired[Literal["refine", "revise"]]


class MainPointsInput(TypedDict):
    context: str


class SummaryInput(MainPointsInput):
    table_of_contents: str


class SummaryRevisionInput(MainPointsInput):
    original_summary: str
    user_feedback: str | dict


class SummaryStructureInput(SummaryRevisionInput):
    outline: str


class TopicGradeInput(TypedDict):
    docs: str
    question: str


class AnswerInput(TypedDict):
    documents: list[Document]
    question: str
    history: list[BaseMessage]
    history_messages: NotRequired[int]
    history_characters: NotRequired[int]


def _mapping(value):
    if not isinstance(value, Mapping):
        raise ChainInputError("Task inputs must be a mapping")
    return value


def _text(inputs, name, *, nonblank=False):
    value = inputs.get(name)
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise ChainInputError(f"Task input {name} must be {'nonblank ' if nonblank else ''}text")
    return value


def _question_type(inputs):
    value = inputs.get("question_type")
    if value not in ("MCQ", "T/F"):
        raise ChainInputError("Task input question_type must be MCQ or T/F")
    return value


def _json(value):
    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError("Unsupported task input")

    try:
        return json.dumps(value, default=encode, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ChainInputError("Task input must contain JSON-compatible data") from exc


def _feedback(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return _json(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ChainInputError("Feedback must be nonblank text or a structured mapping")


def _generation_inputs(value):
    inputs = _mapping(value)
    previous = inputs.get("previous_questions", [])
    if not isinstance(previous, list) or any(not isinstance(item, str) for item in previous):
        raise ChainInputError("Previous questions must be a list of text stems")
    # A bounded recent list guides diversity; controllers check every saved stem.
    selected, size = [], 0
    for question in reversed(previous):
        measured = len(question.encode("utf-8"))
        if size + measured > 2000:
            break
        selected.append(question)
        size += measured
    difficulty = inputs.get("difficulty", "medium")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ChainInputError("Task input difficulty must be easy, medium or hard")
    from helper.content_understanding import understand_content, understanding_prompt

    context = _text(inputs, "context")
    understanding = inputs.get("content_understanding")
    if not isinstance(understanding, dict):
        understanding = understand_content(context)
    feedback = inputs.get("validation_feedback", "")
    if not isinstance(feedback, str):
        raise ChainInputError("Validation feedback must be text")
    return {
        "context": context,
        "question_type": _question_type(inputs),
        "difficulty": difficulty,
        "content_understanding": understanding_prompt(understanding),
        "validation_feedback": feedback[:1200],
        "previous_questions": _json(list(reversed(selected))),
    }


def _review_inputs(value):
    inputs = _mapping(value)
    result = _generation_inputs(inputs)
    options = inputs.get("options")
    if "options" not in inputs or (
        options is not None
        and (not isinstance(options, list) or any(not isinstance(item, str) for item in options))
    ):
        raise ChainInputError("Review options must be a text list or null")
    return {
        **result,
        **{name: _text(inputs, name) for name in ("question", "answer", "explanation")},
        "options": _json(options),
    }


def _revision_inputs(value):
    inputs = _mapping(value)
    result = _generation_inputs(inputs)
    try:
        current = structured_output(Question, inputs.get("current_question"))
    except StructuredOutputError as exc:
        raise ChainInputError("Revision requires a valid current question") from exc
    if current.question_type != result["question_type"]:
        raise ChainInputError("Current question type must match the requested type")
    if current.difficulty != result["difficulty"]:
        raise ChainInputError("Current question difficulty must match the requested difficulty")
    history = inputs.get("history")
    if not isinstance(history, list) or any(not isinstance(item, Mapping) for item in history):
        raise ChainInputError("Revision history must be a list of records")
    action = inputs.get("action", "revise")
    if action not in {"refine", "revise"}:
        raise ChainInputError("Revision action must be refine or revise")
    return {
        **result,
        "action": action,
        "current_question": _json(current),
        "feedback": _feedback(inputs.get("feedback")),
        "history": _json(history),
        "_current_question": current,
    }


def _main_points_inputs(value):
    inputs = _mapping(value)
    return {"context": _text(inputs, "context")}


def _summary_inputs(value):
    inputs = _mapping(value)
    return {
        **_main_points_inputs(inputs),
        "table_of_contents": _text(inputs, "table_of_contents", nonblank=True),
    }


def _summary_revision_inputs(value):
    inputs = _mapping(value)
    return {
        **_main_points_inputs(inputs),
        "original_summary": _text(inputs, "original_summary", nonblank=True),
        "user_feedback": _feedback(inputs.get("user_feedback")),
    }


def _summary_structure_inputs(value):
    inputs = _mapping(value)
    return {
        **_main_points_inputs(inputs),
        "outline": _text(inputs, "outline", nonblank=True),
        "summary": _text(inputs, "original_summary", nonblank=True),
        "feedback": _feedback(inputs.get("user_feedback")),
    }


def _topic_grade_inputs(value):
    inputs = _mapping(value)
    return {"docs": _text(inputs, "docs"), "question": _text(inputs, "question", nonblank=True)}


def _answer_inputs(value):
    inputs = _mapping(value)
    documents, history = inputs.get("documents"), inputs.get("history")
    if not isinstance(documents, list) or any(not isinstance(item, Document) for item in documents):
        raise ChainInputError("Answer documents must be a list of Documents")
    if not isinstance(history, list) or any(not isinstance(item, BaseMessage) for item in history):
        raise ChainInputError("Answer history must be a list of LangChain messages")
    budgets = {}
    for name, default in (("history_messages", 12), ("history_characters", 12000)):
        budget = inputs.get(name, default)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ChainInputError(f"Task input {name} must be a positive integer")
        budgets[name] = budget
    try:
        return prepare_question_answer_inputs(
            documents, _text(inputs, "question", nonblank=True), history, **budgets
        )
    except (TypeError, ValueError) as exc:
        raise ChainInputError("Invalid answer evidence or history") from exc


def formatted_prompt_size(value, schema=None):
    schema_text = (
        json.dumps(schema.model_json_schema(), ensure_ascii=False) if schema is not None else ""
    )
    text = value.to_string() if hasattr(value, "to_string") else str(value)
    messages = value.to_messages() if hasattr(value, "to_messages") else []
    return len((text + "\n" + schema_text).encode("utf-8")) + 16 * (len(messages) + 2)


def revision_source_limit(settings, workflow):
    prefixes = (
        ("QUESTION_GENERATER", "QUESTION_REFINER", "QUESTION_REWRITER")
        if workflow == "question_generation"
        else ("SUMMARIZER_MAINPOINT", "SUMMARIZER_WRITER", "SUMMARIZER_REWRITER")
    )

    def setting(name, default=None):
        return (
            settings.get(name, default)
            if isinstance(settings, Mapping)
            else getattr(settings, name, default)
        )

    configured = int(setting("GENERATION_CONTEXT_CHARACTERS", 24000))
    budgets = []
    for prefix in prefixes:
        provider = setting(prefix + "_PROVIDER")
        if provider is not None:
            budgets.append(
                min(configured, int(setting("OLLAMA_NUM_CTX", 16384)) - 2048)
                if provider == "OLLAMA"
                else configured
            )
    if not budgets:
        return None  # Pure callers may provide only character/chunk settings.
    if workflow == "question_generation":
        base = QuestionRewriterPrompt.invoke(
            {
                "context": "",
                "question_type": "MCQ",
                "difficulty": "medium",
                "action": "revise",
                "current_question": "{}",
                "feedback": "{}",
                "history": "[]",
            }
        )
        overhead = max(
            formatted_prompt_size(base, MCQRevisionResult),
            formatted_prompt_size(base, TrueFalseRevisionResult),
        )
        copies = 2
    else:
        base = SummarizerStructurePrompt.invoke(
            {"context": "", "outline": "", "summary": "", "feedback": "{}"}
        )
        overhead = max(
            formatted_prompt_size(base, SummaryStructureRevision),
            formatted_prompt_size(
                SummarizerRewriterPrompt.invoke(
                    {"context": "", "original_summary": "", "user_feedback": "{}"}
                )
            ),
        )
        copies = 3
    return (min(budgets) - overhead - 1024) // copies


def _fit_optional_history(prompt, schema=None, *, field="history"):
    def fit(value, config: RunnableConfig):
        budget = (config or {}).get("configurable", {}).get("generation_input_budget")
        if budget is None:
            return value
        if type(budget) is not int or budget <= 0:
            raise ChainInputError("The generation input budget must be a positive integer")
        result = dict(value)
        try:
            payload = json.loads(result[field])
        except (KeyError, TypeError, ValueError):
            return result
        history = (
            payload
            if field == "history"
            else payload.get("prior")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(history, list):
            return result
        history = list(history)
        while formatted_prompt_size(prompt.invoke(result), schema) > budget and history:
            first = history[0]
            if (
                isinstance(first, Mapping)
                and isinstance(first.get("retained_requirements"), list)
                and first["retained_requirements"]
            ):
                requirements = first["retained_requirements"][1:]
                if requirements:
                    history[0] = {**first, "retained_requirements": requirements}
                else:
                    history.pop(0)
            else:
                history.pop(0)
            result[field] = _json(history if field == "history" else {**payload, "prior": history})
        # Mandatory source, current draft and feedback remain exact. The final
        # model boundary raises explicitly if those alone still cannot fit.
        return result

    return RunnableLambda(fit, name="fit_optional_revision_history")


def _model_boundary(llm, schema=None):
    try:
        model = llm.with_structured_output(schema) if schema is not None else llm
    except ProviderUnavailableError as exc:
        raise ProviderExecutionError(str(exc)) from exc
    except Exception as exc:
        raise ProviderExecutionError(
            "The configured model could not prepare the requested task"
        ) from exc

    def validate_prompt(value, config):
        budget = (config or {}).get("configurable", {}).get("generation_input_budget")
        if budget is None:
            return
        if type(budget) is not int or budget <= 0:
            raise ChainInputError("The generation input budget must be a positive integer")
        measured = formatted_prompt_size(value, schema)
        if measured > budget:
            raise ChainInputError(
                "The formatted task, source, and history exceed the configured model context budget; no input was truncated"
            )

    def completed_result(result):
        metadata = getattr(result, "response_metadata", {})
        if isinstance(metadata, Mapping):
            for name in ("finish_reason", "done_reason", "stop_reason"):
                reason = metadata.get(name)
                reason = str(getattr(reason, "name", reason)).lower()
                if reason in {"length", "max_tokens", "max_token", "2"} or reason.endswith(
                    ".max_tokens"
                ):
                    raise ProviderExecutionError(
                        "The provider reached its output limit before completing the result"
                    )
        return result

    def invoke(value, config: RunnableConfig):
        validate_prompt(value, config)
        try:
            return completed_result(model.invoke(value, config=config))
        except (OutputParserException, ValidationError) as exc:
            _log_structured_failure(schema, exc)
            raise StructuredOutputError("The model returned an invalid structured result") from exc
        except ProviderUnavailableError as exc:
            raise ProviderExecutionError(str(exc)) from exc
        except Exception as exc:
            raise ProviderExecutionError("The configured model request failed") from exc

    async def ainvoke(value, config: RunnableConfig):
        validate_prompt(value, config)
        try:
            return completed_result(await model.ainvoke(value, config=config))
        except (OutputParserException, ValidationError) as exc:
            _log_structured_failure(schema, exc)
            raise StructuredOutputError("The model returned an invalid structured result") from exc
        except ProviderUnavailableError as exc:
            raise ProviderExecutionError(str(exc)) from exc
        except Exception as exc:
            raise ProviderExecutionError("The configured model request failed") from exc

    return RunnableLambda(invoke, afunc=ainvoke, name="model_execution")


def _generation_result(value):
    result = structured_output(GenerationResult, value["_result"])
    if result.question is not None:
        requested_question(result.question, value["question_type"])
        if result.question.difficulty != value["difficulty"]:
            raise StructuredOutputError("Generated question difficulty does not match the request")
    return result


def _revision_result(value):
    result = structured_output(RevisionResult, value["_result"])
    requested_question(result.question, value["question_type"])
    if result.question.difficulty != value["difficulty"]:
        raise StructuredOutputError("Revised question difficulty does not match the request")
    if result.status == "unchanged" and result.question != value["_current_question"]:
        raise StructuredOutputError("Unchanged revision must preserve the exact current question")
    return result


@lru_cache(maxsize=6)
def generation_schema(question_type, difficulty):
    base_question, base_result = (
        (TrueFalseQuestion, TrueFalseGenerationResult)
        if question_type == "T/F"
        else (MCQQuestion, MCQGenerationResult)
    )
    question = create_model(
        base_question.__name__ + difficulty.title(),
        __base__=base_question,
        difficulty=(
            Literal[difficulty],
            Field(description="Exactly the requested difficulty; required."),
        ),
    )
    return create_model(
        base_result.__name__,
        __base__=base_result,
        question=(
            question | None,
            Field(
                description="Complete requested question for generated; null for insufficient_source."
            ),
        ),
    )


def build_question_generation_chain(llm) -> Runnable:
    prompt = RunnableLambda(
        lambda value: question_generation_prompt(
            value["question_type"], value["difficulty"]
        ).invoke(value)
    )
    branches = [
        (
            lambda value, kind=kind, level=level: value["question_type"] == kind
            and value["difficulty"] == level,
            prompt | _model_boundary(llm, generation_schema(kind, level)),
        )
        for kind in ("T/F", "MCQ")
        for level in ("easy", "medium", "hard")
    ]
    generate = RunnableBranch(*branches[:-1], branches[-1][1])
    return (
        RunnableLambda(_generation_inputs)
        | RunnablePassthrough.assign(_result=generate)
        | RunnableLambda(_generation_result)
    ).with_types(input_type=GenerationInput, output_type=GenerationResult)


def build_question_review_chain(llm) -> Runnable:
    def verdict(value):
        assessment = structured_output(QuestionAssessment, value)
        issues = []
        for name in (
            "factual_support",
            "option_correctness",
            "unique_answer",
            "explanation_consistency",
            "originality",
            "difficulty",
            "ambiguity",
            "distractor_quality",
            "evidence_availability",
        ):
            if getattr(assessment, name) != "failed":
                continue
            field = (
                "question"
                if name in {"difficulty", "originality", "ambiguity"}
                else "options"
                if name in {"option_correctness", "unique_answer", "distractor_quality"}
                else "explanation"
            )
            kind = (
                "structural"
                if name == "difficulty"
                else "ambiguity"
                if name == "ambiguity"
                else "distractor"
                if name == "distractor_quality"
                else "factual"
            )
            issues.append(
                ReviewIssue(
                    code="difficulty_mismatch" if name == "difficulty" else name,
                    kind=kind,
                    field=field,
                    severity="high",
                    correction_hint=assessment.reasoning,
                )
            )
        return Feedback(
            acceptable=not issues,
            issues=issues,
            feedback=assessment.reasoning,
            assessment=assessment,
        )

    return (
        RunnableLambda(_review_inputs)
        | QuestionRefinerPrompt
        | _model_boundary(llm, QuestionAssessment)
        | RunnableLambda(verdict)
    ).with_types(input_type=ReviewInput, output_type=Feedback)


def build_content_classification_chain(llm) -> Runnable:
    def prepare(value):
        blocks = _mapping(value).get("blocks")
        if not isinstance(blocks, list) or not 1 <= len(blocks) <= 16:
            raise ChainInputError(
                "Educational classification requires one to sixteen source blocks"
            )
        identifiers = []
        for block in blocks:
            block = _mapping(block)
            identifiers.append(_text(block, "block_id", nonblank=True))
            _text(block, "text", nonblank=True)
        if len(set(identifiers)) != len(identifiers):
            raise ChainInputError("Educational block identifiers must be unique")
        return {"blocks_json": _json(blocks), "_block_ids": identifiers}

    def validate(value):
        result = structured_output(ContentClassificationResult, value["_result"])
        if sorted(item.block_id for item in result.decisions) != sorted(value["_block_ids"]):
            raise StructuredOutputError(
                "Classification must preserve every source block identifier exactly once"
            )
        return result

    return (
        RunnableLambda(prepare)
        | RunnablePassthrough.assign(
            _result=ContentClassificationPrompt | _model_boundary(llm, ContentClassificationResult)
        )
        | RunnableLambda(validate)
    )


def build_question_revision_chain(llm) -> Runnable:
    revise = RunnableBranch(
        (
            lambda value: value["question_type"] == "T/F",
            _fit_optional_history(QuestionRewriterPrompt, TrueFalseRevisionResult)
            | QuestionRewriterPrompt
            | _model_boundary(llm, TrueFalseRevisionResult),
        ),
        _fit_optional_history(QuestionRewriterPrompt, MCQRevisionResult)
        | QuestionRewriterPrompt
        | _model_boundary(llm, MCQRevisionResult),
    )
    return (
        RunnableLambda(_revision_inputs)
        | RunnablePassthrough.assign(_result=revise)
        | RunnableLambda(_revision_result)
    ).with_types(input_type=RevisionInput, output_type=RevisionResult)


def _text_chain(llm, prepare, prompt, input_type, *, optional_history=None):
    prepared = RunnableLambda(prepare)
    if optional_history is not None:
        prepared = prepared | _fit_optional_history(prompt, field=optional_history)
    return (prepared | prompt | _model_boundary(llm) | RunnableLambda(text_output)).with_types(
        input_type=input_type, output_type=str
    )


def build_main_points_chain(llm) -> Runnable:
    return _text_chain(llm, _main_points_inputs, SummarizerMainPointPrompt, MainPointsInput)


def build_summary_chain(llm) -> Runnable:
    return _text_chain(llm, _summary_inputs, SummarizerGenPrompt, SummaryInput)


def build_summary_revision_chain(llm) -> Runnable:
    return _text_chain(
        llm,
        _summary_revision_inputs,
        SummarizerRewriterPrompt,
        SummaryRevisionInput,
        optional_history="user_feedback",
    )


def build_summary_structure_chain(llm) -> Runnable:
    return (
        RunnableLambda(_summary_structure_inputs)
        | SummarizerStructurePrompt
        | _model_boundary(llm, SummaryStructureRevision)
        | RunnableLambda(lambda value: structured_output(SummaryStructureRevision, value))
    ).with_types(input_type=SummaryStructureInput, output_type=SummaryStructureRevision)


def build_topic_grade_chain(llm) -> Runnable:
    return (
        RunnableLambda(_topic_grade_inputs)
        | GradeQuestionPrompt
        | _model_boundary(llm, GradeQuestion)
        | RunnableLambda(lambda value: structured_output(GradeQuestion, value))
    ).with_types(input_type=TopicGradeInput, output_type=GradeQuestion)


def build_answer_chain(llm) -> Runnable:
    return _text_chain(llm, _answer_inputs, QuestionAnswerPrompt, AnswerInput)


def _rag_chain(llm, prepare, prompt, schema, validate, input_type):
    from .runtime import token_counter_for

    counter = token_counter_for(llm)
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)

    def prompt_budget(value, config: RunnableConfig):
        configurable = config.get("configurable", {})
        if "rag_input_budget" not in configurable:
            return value
        budget = configurable["rag_input_budget"]
        if type(budget) is not int or budget <= 0:
            raise ChainInputError("RAG input budget must be a positive integer")
        text = value.to_string() + "\n" + schema_text
        try:
            size = counter(text) if counter is not None else len(text.encode("utf-8"))
        except Exception as exc:
            raise ChainInputError("The local RAG tokenizer could not count the prompt") from exc
        # Include chat framing and structured-output envelope overhead as well as source text.
        if type(size) is not int or size < 0 or size + 16 * (len(value.to_messages()) + 2) > budget:
            raise ChainInputError("The formatted RAG task exceeds its configured input budget")
        return value

    return (
        RunnableLambda(prepare)
        | RunnablePassthrough.assign(
            _result=prompt | RunnableLambda(prompt_budget) | _model_boundary(llm, schema)
        )
        | RunnableLambda(validate)
    ).with_types(input_type=input_type, output_type=schema)


def build_query_rewrite_chain(llm) -> Runnable:
    return _rag_chain(
        llm,
        query_rewrite_inputs,
        QueryRewritePrompt,
        QueryRewrite,
        query_rewrite_result,
        QueryRewriteInput,
    )


def build_evidence_assessment_chain(llm) -> Runnable:
    return _rag_chain(
        llm,
        evidence_assessment_inputs,
        EvidenceAssessmentPrompt,
        EvidenceDecision,
        evidence_assessment_result,
        EvidenceAssessmentInput,
    )


def build_grounded_answer_chain(llm) -> Runnable:
    return _rag_chain(
        llm,
        grounded_answer_inputs,
        GroundedAnswerPrompt,
        GroundedAnswer,
        grounded_answer_result,
        GroundedAnswerInput,
    )


def build_grounding_check_chain(llm) -> Runnable:
    return _rag_chain(
        llm,
        grounding_check_inputs,
        GroundingCheckPrompt,
        GroundingAssessment,
        grounding_check_result,
        GroundingCheckInput,
    )


TASK_BUILDERS = {
    "content_classification": build_content_classification_chain,
    "question_generation": build_question_generation_chain,
    "question_review": build_question_review_chain,
    "question_revision": build_question_revision_chain,
    "main_points": build_main_points_chain,
    "summary": build_summary_chain,
    "summary_revision": build_summary_revision_chain,
    "summary_structure_revision": build_summary_structure_chain,
    "topic_grade": build_topic_grade_chain,
    "answer": build_answer_chain,
    "query_rewrite": build_query_rewrite_chain,
    "evidence_assessment": build_evidence_assessment_chain,
    "grounded_answer": build_grounded_answer_chain,
    "grounding_check": build_grounding_check_chain,
}


def build_task_chain(task: str, llm) -> Runnable:
    try:
        builder = TASK_BUILDERS[task]
    except (KeyError, TypeError):
        raise ChainInputError("Unknown LLM task") from None
    chain = builder(llm)
    if task == "question_generation":
        chain = _question_schema_repair(chain)
    if task in {
        "question_revision",
        "query_rewrite",
        "evidence_assessment",
        "grounded_answer",
        "grounding_check",
    }:
        retried = chain.with_retry(
            retry_if_exception_type=(StructuredOutputError,),
            stop_after_attempt=2,
            wait_exponential_jitter=False,
        )
        if task == "question_revision":
            # Automatic refinement accounts for one provider call per graph node.
            chain = RunnableBranch(
                (
                    lambda value: isinstance(value, Mapping) and value.get("action") == "refine",
                    chain,
                ),
                retried,
            )
        else:
            chain = retried
    return chain.with_config(run_name=task)


def _question_schema_repair(chain):
    def repair_input(value, error):
        issues, cause, visited = [], error, set()
        while cause is not None and id(cause) not in visited:
            visited.add(id(cause))
            if isinstance(cause, ValidationError):
                issues.extend(
                    ".".join(str(part) for part in issue["loc"]) + ": " + issue["type"]
                    for issue in cause.errors(include_input=False, include_context=False)
                )
            cause = cause.__cause__ or cause.__context__
        note = (
            "SCHEMA_REPAIR: The previous structured response was rejected. Return a fresh complete result using the same SOURCE, "
            "REQUESTED_TYPE and REQUESTED_DIFFICULTY. Do not omit fields or add extra fields. "
            "T/F options must be null and answer the quoted string True or False; MCQ needs four distinct labelled options and answer A/B/C/D. "
            "For generated use reason null; for insufficient_source use question null and a nonblank reason. "
            "Rejected constraints: "
            + "; ".join(issues[:8] or ["question type, difficulty, or structured result contract"])
        )
        return {
            **value,
            "validation_feedback": note[:1000] + "\n" + value.get("validation_feedback", "")[:180],
        }

    def invoke(value, config: RunnableConfig):
        try:
            return chain.invoke(value, config=config)
        except StructuredOutputError as error:
            try:
                repaired = chain.invoke(repair_input(value, error), config=config)
            except (StructuredOutputError, ProviderExecutionError) as rejected:
                rejected.repair_attempts = 1
                raise
            repaired._repair_attempts = 1
            return repaired

    async def ainvoke(value, config: RunnableConfig):
        try:
            return await chain.ainvoke(value, config=config)
        except StructuredOutputError as error:
            try:
                repaired = await chain.ainvoke(repair_input(value, error), config=config)
            except (StructuredOutputError, ProviderExecutionError) as rejected:
                rejected.repair_attempts = 1
                raise
            repaired._repair_attempts = 1
            return repaired

    return RunnableLambda(invoke, afunc=ainvoke, name="bounded_question_schema_repair")
