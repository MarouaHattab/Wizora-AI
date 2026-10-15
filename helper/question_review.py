from chains.contracts import StructuredOutputError, requested_question, structured_output
from chains.schemes.QuestionSchem import Feedback, Question, ReviewIssue


REVIEW_MESSAGE = (
    "Provide feedback, type 'auto' or 'refine' to automatically refine the current question, "
    "or type 'save' to finish"
)
ACCEPT_ACTIONS = frozenset({"save", "accept"})
REFINE_ACTIONS = frozenset({"auto", "refine"})


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
        question = structured_output(Question, {
            key: state.get(key) for key in ("question", "question_type", "options", "answer", "explanation")
        })
        requested_question(question, requested)
    except (StructuredOutputError, TypeError, ValueError) as exc:
        hint = str(exc) or "Restore a schema-valid question of the requested type."
        return Feedback(
            acceptable=False,
            verdict="needs_revision",
            issues=[ReviewIssue(
                code="invalid_structure", kind="structural", field="question",
                severity="high", correction_hint=hint[:2000],
            )],
            feedback="The draft failed deterministic structural validation.",
        )
    return None


def validate_question_draft(state, question=None):
    draft = question
    if draft is None:
        draft = {key: state.get(key) for key in ("question", "question_type", "options", "answer", "explanation")}
    if hasattr(draft, "model_dump"):
        draft = draft.model_dump()
    question = structured_output(Question, draft)
    return requested_question(question, state["question_type"])


def refine_budget_exhausted(state, settings, *, extra_calls=0):
    """Stop before another automatic revision or model call would exceed its budget."""
    revisions = state.get("auto_revision_count", 0)
    calls = state.get("review_model_calls", 0)
    if type(revisions) is not int or type(calls) is not int or type(extra_calls) is not int or extra_calls < 0:
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
    """Rewrite only for validated medium or high findings."""
    if critic is None or critic.verdict != "needs_revision":
        return False
    return any(issue.severity != "low" for issue in critic.issues)


def invalid_critic_feedback():
    return Feedback(
        acceptable=False,
        verdict="needs_revision",
        issues=[ReviewIssue(
            code="invalid_critic_output",
            kind="structural",
            field="question",
            severity="high",
            correction_hint="Repeat the review. The previous critic result was not a valid decision.",
        )],
        feedback="The critic result could not be validated, so the draft was not rewritten.",
    )
