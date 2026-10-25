import hashlib
import json
import re
import unicodedata

from chains import StructuredOutputError, requested_question, structured_output
from schemas import Feedback, Question, ReviewIssue

REVIEW_MESSAGE = (
    "Provide feedback, type 'auto' or 'refine' to automatically refine the current question, "
    "or type 'save' to finish"
)
ACCEPT_ACTIONS = frozenset({"save", "accept"})
REFINE_ACTIONS = frozenset({"auto", "refine"})

DIFFICULTY_RUBRIC = {
    "easy": "Recall one explicit source fact or definition.",
    "medium": "Apply or explain a supported relationship in a new context.",
    "hard": "Combine at least two distinct source-supported premises that interact in an inference or multiple reasoning steps; a long recall statement does not qualify.",
}
_ASSESSMENT_CHECKS = (
    "factual_support",
    "option_correctness",
    "unique_answer",
    "explanation_consistency",
    "originality",
    "difficulty",
    "ambiguity",
    "distractor_quality",
    "evidence_availability",
)


def _normalized_evidence(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _educational_source(context):
    # Record boundaries may split one sentence. Provenance labels are not part
    # of that sentence and must not break an otherwise exact supporting quote.
    return re.sub(r"(?m)^SOURCE_LOCATOR: [^\r\n]*\r?\nSOURCE_TEXT:\r?\n", "", context)


def _normalized_quote(value):
    value = value.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}
    if len(value) > 1 and value[0] in pairs and value[-1] == pairs[value[0]]:
        value = value[1:-1]
    return _normalized_evidence(value)


def _assessment_fingerprints(question, context):
    return {
        "draft_fingerprint": hashlib.sha256(
            json.dumps(question.model_dump(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "source_fingerprint": hashlib.sha256(context.encode()).hexdigest(),
    }


def assessment_matches(value, question, context):
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "1.1"
        and all(key in value.get("checks", {}) for key in _ASSESSMENT_CHECKS)
        and all(
            value.get(key) == fingerprint
            for key, fingerprint in _assessment_fingerprints(question, context).items()
        )
    )


def question_assessment(question, context, critic=None, *, understanding=None, previous=()):
    from helper.content_understanding import grounding_issue, originality_issue, understand_content

    if isinstance(critic, dict):
        try:
            critic = Feedback.model_validate(critic)
        except (TypeError, ValueError):
            critic = None
    details = critic.assessment if isinstance(critic, Feedback) else None
    checks = {
        key: getattr(details, key) if details is not None else "uncertain"
        for key in _ASSESSMENT_CHECKS
    }
    checks["structure"] = "passed"  # Question is already schema validated.
    warnings, evidence = [], []
    if details is None:
        warnings.append(
            "The critic did not provide a complete evidence and difficulty assessment; human review is required."
        )
    else:
        source = _normalized_evidence(_educational_source(context))
        evidence = list(dict.fromkeys(details.supporting_evidence))
        if not evidence or any(_normalized_quote(item) not in source for item in evidence):
            if checks["difficulty"] != "failed":
                checks["difficulty"] = "uncertain"
            if checks["factual_support"] != "failed":
                checks["factual_support"] = "uncertain"
            if checks["evidence_availability"] != "failed":
                checks["evidence_availability"] = "uncertain"
            warnings.append(
                "Supporting excerpts were missing or could not be located in the selected source part."
            )
        levels = {
            "easy": {"recall"},
            "medium": {"application", "explanation"},
            "hard": {"inference", "multi_step"},
        }
        if details.assessed_difficulty == "uncertain" or details.cognitive_level == "uncertain":
            if checks["difficulty"] != "failed":
                checks["difficulty"] = "uncertain"
        elif (
            details.assessed_difficulty != question.difficulty
            or details.cognitive_level not in levels[question.difficulty]
        ):
            checks["difficulty"] = "failed"
            warnings.append(
                "The assessed cognitive demand does not meet the requested difficulty rubric."
            )
        distinct_evidence = []
        for item in sorted({_normalized_quote(item) for item in evidence}, key=len, reverse=True):
            if not any(item in retained for retained in distinct_evidence):
                distinct_evidence.append(item)
        if question.difficulty == "hard" and len(distinct_evidence) < 2:
            checks["difficulty"] = "uncertain" if checks["difficulty"] != "failed" else "failed"
            warnings.append(
                "A hard question requires at least two distinct source-supported interacting premises."
            )
        if question.question_type == "MCQ":
            options = details.option_evaluations
            if sorted(item.label for item in options) != list("ABCD"):
                for key in ("option_correctness", "unique_answer"):
                    if checks[key] != "failed":
                        checks[key] = "uncertain"
                warnings.append("All four MCQ options need an independent correctness assessment.")
            elif any(item.correctness == "uncertain" for item in options):
                for key in ("option_correctness", "unique_answer"):
                    if checks[key] != "failed":
                        checks[key] = "uncertain"
            elif [item.label for item in options if item.correctness == "correct"] != [
                question.answer
            ]:
                checks["option_correctness"] = checks["unique_answer"] = "failed"
                warnings.append(
                    "The option assessment does not identify exactly the selected answer as correct."
                )
    if question.question_type == "T/F":
        checks["option_correctness"] = "not_applicable"
        checks["distractor_quality"] = "not_applicable"
        exact_assertion = _normalized_evidence(question.question).rstrip(".?!؟")
        source_sentences = {
            _normalized_evidence(item).rstrip(".?!؟")
            for item in re.split(r"(?<=[.!?؟])\s+|\n+", context)
        }
        if question.answer == "False" and exact_assertion in source_sentences:
            checks["factual_support"] = "failed"
            warnings.append(
                "The False answer contradicts the identical assertion present in the source."
            )
    if not context.strip():
        checks["evidence_availability"] = checks["factual_support"] = "failed"
        warnings.append(
            "No usable source context is available for the question's factual premises."
        )
    quality_hints = {
        "ambiguity": "Make the stem and its conditions unambiguous.",
        "distractor_quality": "Use distinct, plausible, parallel incorrect MCQ alternatives.",
        "evidence_availability": "Every necessary premise must be present in the supplied source and supported by locatable evidence.",
    }
    for name, hint in quality_hints.items():
        if name == "distractor_quality" and question.question_type == "T/F":
            continue
        if checks[name] == "not_applicable":
            checks[name] = "uncertain"
        if checks[name] in {"failed", "uncertain"}:
            warnings.append(f"The {name} assessment is {checks[name]}. {hint}")
    understanding = understanding or understand_content(context)
    for key, issue in (
        ("originality", originality_issue(question.question, understanding, previous)),
        ("factual_support", grounding_issue(question, understanding)),
    ):
        if issue:
            checks[key] = "failed"
            warnings.append(issue)
    if isinstance(critic, Feedback) and not critic.acceptable:
        warnings.append(critic.feedback)
        if not any(value in {"failed", "uncertain"} for value in checks.values()):
            checks["factual_support"] = "uncertain"
    applicable = [
        value
        for key, value in checks.items()
        if not (
            key in {"option_correctness", "distractor_quality"} and question.question_type == "T/F"
        )
    ]
    confident = all(value == "passed" for value in applicable)
    return {
        "schema_version": "1.1",
        "status": "validated" if confident else "review_required",
        "checks": checks,
        "assessment_gaps": [key for key, value in checks.items() if value == "uncertain"],
        "requested_difficulty": question.difficulty,
        "assessed_difficulty": details.assessed_difficulty if details else "uncertain",
        "cognitive_level": details.cognitive_level if details else "uncertain",
        "supporting_evidence": evidence,
        "reasoning": details.reasoning if details else None,
        "option_evaluations": [item.model_dump() for item in details.option_evaluations]
        if details
        else [],
        "rubric": dict(DIFFICULTY_RUBRIC),
        "warnings": list(dict.fromkeys(warnings)),
        "limitation": "Source-grounded model assessment with local checks; this does not guarantee perfect accuracy.",
        **_assessment_fingerprints(question, context),
    }


def parse_review_action(feedback):
    key = feedback.strip().casefold() if isinstance(feedback, str) else ""
    if key in ACCEPT_ACTIONS:
        return "accept"
    if key in REFINE_ACTIONS:
        return "refine"
    return "revise"


def structural_feedback(state):
    requested = state.get("question_type")
    try:
        question = structured_output(
            Question,
            {
                key: state.get(key)
                for key in ("question", "question_type", "options", "answer", "explanation")
            },
        )
        requested_question(question, requested)
    except (StructuredOutputError, TypeError, ValueError) as exc:
        hint = str(exc) or "Restore a schema-valid question of the requested type."
        return Feedback(
            acceptable=False,
            verdict="needs_revision",
            issues=[
                ReviewIssue(
                    code="invalid_structure",
                    kind="structural",
                    field="question",
                    severity="high",
                    correction_hint=hint[:2000],
                )
            ],
            feedback="The draft failed deterministic structural validation.",
        )
    return None


def validate_question_draft(state, question=None):
    draft = question
    if draft is None:
        draft = {
            key: state.get(key)
            for key in ("question", "question_type", "options", "answer", "explanation")
        }
        draft["difficulty"] = state.get("difficulty", "medium")
    if hasattr(draft, "model_dump"):
        draft = draft.model_dump()
    question = structured_output(Question, draft)
    requested_question(question, state["question_type"])
    if question.difficulty != state.get("difficulty", "medium"):
        raise StructuredOutputError(
            "Generated question difficulty does not match the requested difficulty"
        )
    return question


def refine_budget_exhausted(state, settings, *, extra_calls=0):
    revisions = state.get("auto_revision_count", 0)
    calls = state.get("review_model_calls", 0)
    if (
        type(revisions) is not int
        or type(calls) is not int
        or type(extra_calls) is not int
        or extra_calls < 0
    ):
        return True
    return (
        revisions >= settings.AUTO_REVISION_LIMIT
        or calls + extra_calls > settings.REFINE_MODEL_CALL_LIMIT
    )


def accepted_feedback(critic):
    if critic is None:
        return False
    return critic.acceptable is True and critic.verdict == "acceptable"


def needs_revision(critic):
    if critic is None or critic.verdict != "needs_revision":
        return False
    return any(issue.severity != "low" for issue in critic.issues)


def invalid_critic_feedback():
    return Feedback(
        acceptable=False,
        verdict="needs_revision",
        issues=[
            ReviewIssue(
                code="invalid_critic_output",
                kind="structural",
                field="question",
                severity="high",
                correction_hint="Repeat the review. The previous critic result was not a valid decision.",
            )
        ],
        feedback="The critic result could not be validated, so the draft was not rewritten.",
    )
