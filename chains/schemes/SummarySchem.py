from typing import Literal

from pydantic import Field, model_validator

from .QuestionSchem import NonemptyText, StructuredResult


class SummaryStructureRevision(StructuredResult):
    outline: NonemptyText = Field(description="Updated numbered outline synchronized with the revised summary.")
    summary: NonemptyText = Field(description="Revised summary organized by the updated outline and grounded in SOURCE.")
    limitation: NonemptyText | None = Field(default=None, description="Null unless a requested change is missing from SOURCE.")


class SummaryRevisionKind(StructuredResult):
    kind: Literal["wording", "structural", "unsupported"]
