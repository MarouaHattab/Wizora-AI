from importlib import import_module

_EXPORTS = {
    "NonemptyText": ("schemas.question", "NonemptyText"),
    "StructuredResult": ("schemas.question", "StructuredResult"),
    "Question": ("schemas.question", "Question"),
    "GenerationResult": ("schemas.question", "GenerationResult"),
    "TrueFalseQuestion": ("schemas.question", "TrueFalseQuestion"),
    "MCQQuestion": ("schemas.question", "MCQQuestion"),
    "TrueFalseGenerationResult": ("schemas.question", "TrueFalseGenerationResult"),
    "MCQGenerationResult": ("schemas.question", "MCQGenerationResult"),
    "RevisionResult": ("schemas.question", "RevisionResult"),
    "TrueFalseRevisionResult": ("schemas.question", "TrueFalseRevisionResult"),
    "MCQRevisionResult": ("schemas.question", "MCQRevisionResult"),
    "ReviewIssue": ("schemas.question", "ReviewIssue"),
    "CheckStatus": ("schemas.question", "CheckStatus"),
    "OptionEvaluation": ("schemas.question", "OptionEvaluation"),
    "QuestionAssessment": ("schemas.question", "QuestionAssessment"),
    "Feedback": ("schemas.question", "Feedback"),
    "GradeQuestion": ("schemas.question", "GradeQuestion"),
    "QueryRewrite": ("schemas.rag", "QueryRewrite"),
    "EvidenceDecision": ("schemas.rag", "EvidenceDecision"),
    "GroundedClaim": ("schemas.rag", "GroundedClaim"),
    "GroundedAnswer": ("schemas.rag", "GroundedAnswer"),
    "ClaimVerdict": ("schemas.rag", "ClaimVerdict"),
    "GroundingAssessment": ("schemas.rag", "GroundingAssessment"),
    "SummaryStructureRevision": ("schemas.summary", "SummaryStructureRevision"),
    "SummaryRevisionKind": ("schemas.summary", "SummaryRevisionKind"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
