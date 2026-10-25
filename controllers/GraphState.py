from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, BaseMessage
from typing_extensions import TypedDict

from helper.workflow_history import merge_messages
from schemas import EvidenceDecision, Feedback, GroundedAnswer, GroundingAssessment


class RevisionRecord(TypedDict):
    id: str
    review_id: str
    asset_id: str | None
    human_feedback: str
    critic_feedback: dict | None
    status: Literal["revised", "unchanged"]
    reason: str | None
    draft_hash: str
    draft_preview: str


class RevisionRequirement(TypedDict, total=False):
    review_id: str
    feedback: str
    feedback_hash: str
    truncated: bool


class ReviewState(TypedDict, total=False):
    asset_id: str
    feedback: str
    human_feedback_text: str
    critic_feedback: Feedback | None
    feedback_review_id: str | None
    history: list[RevisionRecord]
    revision_requirements: list[RevisionRequirement]
    state_version: Literal[4]
    workflow_action: Literal["accept", "refine", "revise"]
    auto_revision_count: int
    review_model_calls: int
    revision_limit_reached: bool
    review_notice: str | None
    revision_kind: Literal["wording", "structural", "unsupported"] | None
    source_chunks: list[dict]
    source_citations: list[dict]
    source_scope: str
    context_note: str


class QuestionGenState(ReviewState):
    context: str
    question_type: Literal["T/F", "MCQ"]
    question: str
    options: list[str] | None
    answer: str
    explanation: str
    source_context: str
    previous_questions: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    content_understanding: dict
    validation: dict
    review_required: bool
    generation_diagnostics: dict
    require_assessed_difficulty: bool


class SummaryGenState(ReviewState):
    context: str
    summary: str
    Main_Points: str
    source_outlines: list[str]
    summary_sections: list[str]


class QuestionAnswerRequiredState(TypedDict):
    asset_id: str
    messages: Annotated[list[AnyMessage], merge_messages]
    relevant_text: list[Document]
    on_topic: Literal["Yes", "No", ""]
    context: str


class QuestionAnswerState(QuestionAnswerRequiredState, total=False):
    video_targeted_attempted: bool
    video_processing_notice: str
    history_trimmed: int
    state_version: Literal[4]
    conversation_history: list[BaseMessage]
    original_question: str
    retrieval_query: str
    index_version: str
    retrieval_pass: int
    retrieved_ids: list[str]
    retrieval_scores: dict[str, float]
    evidence_changed: bool
    retrieval_issue: str | None
    evidence_decision: EvidenceDecision | None
    answer_draft: GroundedAnswer | None
    grounding_assessment: GroundingAssessment | None
    answer_status: Literal["supported", "partial", "insufficient", "off_topic"]
    answer_citations: list[dict]
    grounding_status: Literal["model_assessed", "not_applicable", "rejected"]
    rag_diagnostics: dict
