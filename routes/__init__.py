from importlib import import_module

_EXPORTS = {
    "batch_processing_router": ("routes.batch_processing_routes", "router"),
    "file_processing_router": ("routes.file_processing_routes", "router"),
    "job_results_router": ("routes.job_results_routes", "router"),
    "provider_status_router": ("routes.provider_status_routes", "router"),
    "qa_router": ("routes.qa_routes", "router"),
    "question_gen_router": ("routes.question_gen_routes", "router"),
    "session_router": ("routes.session_routes", "router"),
    "summarizer_router": ("routes.summarizer_routes", "router"),
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
