from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, BaseMessage

from chains.schemes.QuestionSchem import Feedback
from chains.schemes.RAGSchem import EvidenceDecision, GroundedAnswer, GroundingAssessment
from helper.workflow_history import merge_messages


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


class RevisionRequirement(TypedDict):
    review_id: str
    feedback: str
    feedback_hash: NotRequired[str]
    truncated: NotRequired[bool]


class ReviewState(TypedDict, total=False):
    asset_id: str
    feedback: str
    human_feedback_text: str
    critic_feedback: Feedback | None
    feedback_review_id: str | None
    history: list[RevisionRecord]
    revision_requirements: list[RevisionRequirement]
    state_version: Literal[4]


class QuestionGenState(ReviewState):
    context: str
    question_type: Literal["T/F", "MCQ"]
    question: str
    options: list[str] | None
    answer: str
    explanation: str


class SummaryGenState(ReviewState):
    context: str
    summary: str
    Main_Points: str
    old_summary: NotRequired[str]


class QuestionAnswerState(TypedDict):
    asset_id: str
    messages: Annotated[list[AnyMessage], merge_messages]
    relevant_text: list[Document]
    on_topic: Literal["Yes", "No", ""]
    context: str
    history_trimmed: NotRequired[int]
    state_version: NotRequired[Literal[4]]
    conversation_history: NotRequired[list[BaseMessage]]
    original_question: NotRequired[str]
    retrieval_query: NotRequired[str]
    index_version: NotRequired[str]
    retrieval_pass: NotRequired[int]
    retrieved_ids: NotRequired[list[str]]
    retrieval_scores: NotRequired[dict[str, float]]
    evidence_changed: NotRequired[bool]
    retrieval_issue: NotRequired[str | None]
    evidence_decision: NotRequired[EvidenceDecision | None]
    answer_draft: NotRequired[GroundedAnswer | None]
    grounding_assessment: NotRequired[GroundingAssessment | None]
    answer_status: NotRequired[Literal["supported", "partial", "insufficient", "off_topic"]]
    answer_citations: NotRequired[list[dict]]
    grounding_status: NotRequired[Literal["model_assessed", "not_applicable", "rejected"]]
    rag_diagnostics: NotRequired[dict]
