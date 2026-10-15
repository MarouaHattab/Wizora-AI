import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .QuestionSchem import StructuredResult


RAGText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)]
ChunkId = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128)]


def _unique_ids(values):
    if len(values) != len(set(values)):
        raise ValueError("Chunk identifiers must be unique")
    return values


class QueryRewrite(StructuredResult):
    query: RAGText = Field(description="A standalone retrieval question preserving the current request, with references resolved only from the supplied user history.")


class EvidenceDecision(StructuredResult):
    status: Literal["supported", "partial", "insufficient", "off_topic"]
    supporting_chunk_ids: list[ChunkId] = Field(max_length=32)
    missing_information: RAGText | None = Field(description="Null for full support; otherwise describe requested information that the evidence does not establish.")

    @model_validator(mode="after")
    def validate_decision(self):
        _unique_ids(self.supporting_chunk_ids)
        if self.status in {"supported", "partial"}:
            if not self.supporting_chunk_ids:
                raise ValueError("Supported and partial decisions require evidence identifiers")
        elif self.supporting_chunk_ids:
            raise ValueError("Insufficient and off-topic decisions cannot assert supporting evidence")
        if (self.status == "supported") != (self.missing_information is None):
            raise ValueError("Only supported decisions require null missing information")
        return self


class GroundedClaim(StructuredResult):
    text: RAGText = Field(description="One factual answer claim, including its source qualifications or conflict, without citation markup or a Sources section.")
    chunk_ids: list[ChunkId] = Field(min_length=1, max_length=32, description="Exact supplied chunk identifiers supporting this entire claim.")

    @model_validator(mode="after")
    def validate_identifiers(self):
        _unique_ids(self.chunk_ids)
        if re.search(r"\[(?:\d+|E[^\]]*|(?:chunk|source)[^\]]*)\]|\bSources\s*:", self.text, re.IGNORECASE):
            raise ValueError("Citation markup is owned by the application renderer")
        return self


class GroundedAnswer(StructuredResult):
    claims: list[GroundedClaim] = Field(min_length=1, max_length=12)


class ClaimVerdict(StructuredResult):
    claim_index: int = Field(ge=0)
    supported: bool
    reason: RAGText = Field(description="Concise explanation of entailment or the unsupported element; valid citation membership alone is insufficient.")


class GroundingAssessment(StructuredResult):
    claims: list[ClaimVerdict] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_indices(self):
        indices = [claim.claim_index for claim in self.claims]
        if len(indices) != len(set(indices)):
            raise ValueError("Each claim requires exactly one verdict")
        return self
