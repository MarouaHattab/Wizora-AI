from importlib import import_module

_EXPORTS = {
    "QuestionGenState": ("controllers.GraphState", "QuestionGenState"),
    "SummaryGenState": ("controllers.GraphState", "SummaryGenState"),
    "QuestionAnswerState": ("controllers.GraphState", "QuestionAnswerState"),
    "compile_question_graph": ("controllers.QuestionGenGraphController", "compile_question_graph"),
    "compile_summary_graph": ("controllers.SummarizerGenGraphController", "compile_summary_graph"),
    "compile_qa_graph": ("controllers.QuestionAnswerGraphController", "compile_qa_graph"),
    "message_text": ("controllers.QuestionAnswerGraphController", "message_text"),
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
