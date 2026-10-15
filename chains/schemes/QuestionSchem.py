import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonemptyText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")


class Question(StructuredResult):
    question: NonemptyText = Field(description="One unambiguous question or True/False statement supported by the source.")
    question_type: Literal["T/F", "MCQ"] = Field(description="Must match the requested question type.")
    options: list[NonemptyText] | None = Field(
        description="MCQ: an array of exactly four distinct choices labelled A. through D. in order. T/F: JSON null."
    )
    answer: NonemptyText = Field(description="MCQ: exactly A, B, C, or D. T/F: exactly True or False.")
    explanation: NonemptyText = Field(description="Explain the answer using the supplied source, preserving its qualifications.")

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
            contents.append(unicodedata.normalize("NFKC", " ".join(match.group(2).split())).casefold())
        if len(set(contents)) != 4:
            raise ValueError("MCQ option texts must be distinct")
        if len(self.answer) != 1 or self.answer not in "ABCD":
            raise ValueError("MCQ answer must reference an option label A through D")
        return self

    def format(self) -> str:
        return f"Question: {self.question}\nOptions: {self.options}\nAnswer: {self.answer}\nExplanation: {self.explanation}\n"


class GenerationResult(StructuredResult):
    status: Literal["generated", "insufficient_source"]
    question: Question | None
    reason: NonemptyText | None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "generated":
            if self.question is None or self.reason is not None:
                raise ValueError("Generated results require a question and null reason")
        elif self.question is not None or self.reason is None:
            raise ValueError("Insufficient source requires a reason and null question")
        return self


class RevisionResult(StructuredResult):
    status: Literal["revised", "unchanged"]
    question: Question
    reason: NonemptyText | None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "revised" and self.reason is not None:
            raise ValueError("Revised results require a null reason")
        if self.status == "unchanged" and self.reason is None:
            raise ValueError("Unchanged results require a reason describing the conflicting request")
        return self


class ReviewIssue(StructuredResult):
    code: NonemptyText | None = Field(default=None, description="Stable issue code such as unsupported_answer or ambiguous_stem.")
    kind: Literal["structural", "factual", "ambiguity", "distractor"]
    field: Literal["question_type", "question", "options", "answer", "explanation"]
    severity: Literal["low", "medium", "high"] = "medium"
    correction_hint: NonemptyText

    @model_validator(mode="after")
    def default_code(self):
        if self.code is None:
            return self.model_copy(update={"code": self.kind})
        return self


class Feedback(StructuredResult):
    acceptable: bool
    verdict: Literal["acceptable", "needs_revision"] | None = None
    issues: list[ReviewIssue]
    feedback: NonemptyText

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
            return self.model_copy(update={"verdict": verdict})
        if self.verdict != verdict:
            raise ValueError("verdict must match acceptable")
        return self

    def format(self) -> str:
        return self.model_dump_json()


class GradeQuestion(StructuredResult):
    score: Literal["Yes", "No"] = Field(description="Topic relevance only: Yes if related to the source topic, otherwise No. Not an answerability judgment.")
