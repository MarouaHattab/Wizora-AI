import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, model_validator

NonemptyText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")


class Question(StructuredResult):
    question: NonemptyText = Field(
        description="One unambiguous question or True/False statement supported by the source."
    )
    question_type: Literal["T/F", "MCQ"] = Field(
        description="Must match the requested question type."
    )
    options: list[NonemptyText] | None = Field(
        description="MCQ: an array of exactly four distinct choices labelled A. through D. in order. T/F: JSON null."
    )
    answer: NonemptyText = Field(
        description="MCQ: exactly A, B, C, or D. T/F: exactly True or False."
    )
    explanation: NonemptyText = Field(
        description="Explain the answer using the supplied source, preserving its qualifications."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Exactly match the requested difficulty: easy, medium, or hard.",
    )

    @model_validator(mode="after")
    def validate_question(self):
        if self.question_type == "T/F":
            if self.options is not None or self.answer not in {"True", "False"}:
                raise ValueError("T/F requires null options and an answer of True or False")
            return self
        if self.options is None or len(self.options) != 4:
            raise ValueError("MCQ requires exactly four options")
        contents = []
        for label, option in zip("ABCD", self.options):
            match = re.fullmatch(r"([A-D])\. (\S.*)", option)
            if match is None or match.group(1) != label:
                raise ValueError("MCQ options must be labelled A. through D. in order")
            contents.append(
                unicodedata.normalize("NFKC", " ".join(match.group(2).split())).casefold()
            )
        if len(set(contents)) != 4:
            raise ValueError("MCQ option texts must be distinct")
        if len(self.answer) != 1 or self.answer not in "ABCD":
            raise ValueError("MCQ answer must reference an option label A through D")
        return self

    def format(self) -> str:
        return f"Question: {self.question}\nOptions: {self.options}\nAnswer: {self.answer}\nExplanation: {self.explanation}\n"


class GenerationResult(StructuredResult):
    _repair_attempts: int = PrivateAttr(default=0)
    status: Literal["generated", "insufficient_source"] = Field(
        description="generated requires a complete question and reason null; insufficient_source requires question null and a nonblank reason."
    )
    question: Question | None = Field(
        description="Complete question object only for generated; JSON null only for insufficient_source."
    )
    reason: NonemptyText | None = Field(
        description="For generated, MUST be JSON null. For insufficient_source, REQUIRED nonblank explanation of the missing educational evidence; NEVER null."
    )

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "generated":
            if self.question is None or self.reason is not None:
                raise ValueError("Generated results require a question and null reason")
        elif self.question is not None or self.reason is None:
            raise ValueError("Insufficient source requires a reason and null question")
        return self


class TrueFalseQuestion(Question):
    question: NonemptyText = Field(
        description="A declarative statement that can be judged True or False; never an interrogative question."
    )
    question_type: Literal["T/F"] = Field(description="Exactly the string T/F.")
    options: None = Field(
        description="Required explicit JSON null for T/F, never an empty array or omitted field."
    )
    answer: Literal["True", "False"] = Field(
        description="Exactly the JSON string True or False (quoted text), never a JSON boolean true or false."
    )


class MCQQuestion(Question):
    question_type: Literal["MCQ"]
    options: list[NonemptyText] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]


class TrueFalseGenerationResult(GenerationResult):
    question: TrueFalseQuestion | None = Field(
        description="For generated, a complete T/F question with options null and answer a quoted True/False string; for insufficient_source, JSON null."
    )


class MCQGenerationResult(GenerationResult):
    question: MCQQuestion | None = Field(
        description="For generated, a complete MCQ question; for insufficient_source, JSON null."
    )


class RevisionResult(StructuredResult):
    status: Literal["revised", "unchanged"] = Field(
        description="revised if any question field changed; unchanged only when every field exactly matches the current question."
    )
    question: Question = Field(
        description="The complete resulting question. For unchanged, copy every current field exactly."
    )
    reason: NonemptyText | None = Field(
        description="For unchanged, REQUIRED nonblank explanation of why no change is needed or possible, including an already-concise draft. NEVER null for unchanged. For revised, MUST be JSON null."
    )

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "revised" and self.reason is not None:
            raise ValueError("Revised results require a null reason")
        if self.status == "unchanged" and self.reason is None:
            raise ValueError(
                "Unchanged results require a nonblank reason, including when the draft already satisfies the request"
            )
        return self


class TrueFalseRevisionResult(RevisionResult):
    question: TrueFalseQuestion


class MCQRevisionResult(RevisionResult):
    question: MCQQuestion


class ReviewIssue(StructuredResult):
    code: NonemptyText | None = Field(
        default=None, description="Stable issue code such as unsupported_answer or ambiguous_stem."
    )
    kind: Literal["structural", "factual", "ambiguity", "distractor"]
    field: Literal["question_type", "question", "options", "answer", "explanation"]
    severity: Literal["low", "medium", "high"] = "medium"
    correction_hint: NonemptyText

    @model_validator(mode="after")
    def default_code(self):
        if self.code is None:
            self.code = self.kind
        return self


CheckStatus = Literal["passed", "failed", "uncertain", "not_applicable"]


class OptionEvaluation(StructuredResult):
    label: Literal["A", "B", "C", "D"]
    correctness: Literal["correct", "incorrect", "uncertain"]
    rationale: NonemptyText


class QuestionAssessment(StructuredResult):
    factual_support: CheckStatus
    option_correctness: CheckStatus
    unique_answer: CheckStatus
    explanation_consistency: CheckStatus
    originality: CheckStatus
    difficulty: CheckStatus
    ambiguity: CheckStatus = Field(
        default="uncertain",
        description="passed only for a clear stem with explicit conditions and one unambiguous interpretation.",
    )
    distractor_quality: CheckStatus = Field(
        default="uncertain",
        description="MCQ: distinct, plausible, parallel incorrect alternatives; T/F: not_applicable.",
    )
    evidence_availability: CheckStatus = Field(
        default="uncertain",
        description="Whether SOURCE contains every premise needed to settle the question and explanation; quoted excerpts must be locatable.",
    )
    assessed_difficulty: Literal["easy", "medium", "hard", "uncertain"]
    cognitive_level: Literal[
        "recall", "application", "explanation", "inference", "multi_step", "uncertain"
    ]
    supporting_evidence: list[NonemptyText] = Field(
        max_length=8,
        description="Exact short excerpts from SOURCE supporting the answer and explanation; at least two distinct interacting premises for hard.",
    )
    reasoning: NonemptyText
    option_evaluations: list[OptionEvaluation] = Field(
        default_factory=list,
        max_length=4,
        description="MCQ: evaluate all four options independently; T/F: empty array.",
    )


class Feedback(StructuredResult):
    acceptable: bool
    verdict: Literal["acceptable", "needs_revision"] | None = None
    issues: list[ReviewIssue]
    feedback: NonemptyText
    assessment: QuestionAssessment | None = Field(
        default=None,
        description="Explicit source evidence and cognitive-demand checks. Missing legacy assessment requires human review.",
    )

    @model_validator(mode="after")
    def validate_review(self):
        blocking = [issue for issue in self.issues if issue.severity != "low"]
        if self.acceptable:
            if blocking:
                raise ValueError("Acceptable reviews cannot include unresolved blocking issues")
        elif not self.issues:
            raise ValueError("A revision verdict requires actionable findings")
        verdict = "acceptable" if self.acceptable else "needs_revision"
        if self.verdict is None:
            self.verdict = verdict
        if self.verdict != verdict:
            raise ValueError("verdict must match acceptable")
        return self

    def format(self) -> str:
        return self.model_dump_json()


class GradeQuestion(StructuredResult):
    score: Literal["Yes", "No"] = Field(
        description="Topic relevance only: Yes if related to the source topic, otherwise No. Not an answerability judgment."
    )
