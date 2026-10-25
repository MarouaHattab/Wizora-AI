from importlib import import_module

_EXPORTS = {
    "QuestionGenPrompt": ("chains.QuestionGenChain", "QuestionGenPrompt"),
    "QuestionRefinerPrompt": ("chains.QuestionRefinerChain", "QuestionRefinerPrompt"),
    "QuestionRewriterPrompt": ("chains.QuestionRewriterChain", "QuestionRewriterPrompt"),
    "SummarizerGenPrompt": ("chains.SummarizerGenChain", "SummarizerGenPrompt"),
    "SummarizerRewriterPrompt": ("chains.SummarizerRewriterChain", "SummarizerRewriterPrompt"),
    "SummarizerMainPointPrompt": ("chains.SummarizerMainPointChain", "SummarizerMainPointPrompt"),
    "GradeQuestionPrompt": ("chains.GradeQuestionChain", "GradeQuestionPrompt"),
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
    "ChainInputError": ("chains.contracts", "ChainInputError"),
    "StructuredOutputError": ("chains.contracts", "StructuredOutputError"),
    "TextOutputError": ("chains.contracts", "TextOutputError"),
    "ProviderExecutionError": ("chains.contracts", "ProviderExecutionError"),
    "InsufficientSourceError": ("chains.contracts", "InsufficientSourceError"),
    "text_fragment": ("chains.contracts", "text_fragment"),
    "text_output": ("chains.contracts", "text_output"),
    "structured_output": ("chains.contracts", "structured_output"),
    "requested_question": ("chains.contracts", "requested_question"),
    "build_task_chain": ("chains.builders", "build_task_chain"),
    "revision_source_limit": ("chains.builders", "revision_source_limit"),
    "get_task_chain": ("chains.runtime", "get_task_chain"),
    "get_task_token_counter": ("chains.runtime", "get_task_token_counter"),
    "prepare_conversation_history": ("chains.QuestionAnswerChain", "prepare_conversation_history"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
