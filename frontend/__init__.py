from importlib import import_module

_EXPORTS = {
    "BackendClient": ("frontend.client", "BackendClient"),
    "BackendError": ("frontend.client", "BackendError"),
    "FrontendSettings": ("frontend.client", "FrontendSettings"),
    "get_frontend_settings": ("frontend.client", "get_frontend_settings"),
    "GradioUI": ("frontend.controller", "GradioUI"),
    "PracticeUI": ("frontend.practice", "PracticeUI"),
    "SUPPORTED_EXTENSIONS": ("frontend.constants", "SUPPORTED_EXTENSIONS"),
    "VISUAL_EXTENSIONS": ("frontend.constants", "VISUAL_EXTENSIONS"),
    "VLM_CHOICES": ("frontend.constants", "VLM_CHOICES"),
    "VLM_DESCRIPTIONS": ("frontend.constants", "VLM_DESCRIPTIONS"),
    "asset_details": ("frontend.presentation", "asset_details"),
    "markdown_text": ("frontend.presentation", "markdown_text"),
    "timestamp_text": ("frontend.presentation", "timestamp_text"),
    "upload_progress": ("frontend.presentation", "upload_progress"),
    "empty_state": ("frontend.state", "empty_state"),
    "CUSTOM_CSS": ("frontend.styles", "CUSTOM_CSS"),
    "VLM_DROPDOWN_JS": ("frontend.styles", "VLM_DROPDOWN_JS"),
    "create_theme": ("frontend.styles", "create_theme"),
    "SAVE_ZIP_JS": ("frontend.exports", "SAVE_ZIP_JS"),
    "ExportBlocks": ("frontend.exports", "ExportBlocks"),
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
