from importlib import import_module

_EXPORTS = {
    "extract_text_from_pdf": ("tools.pdf_extractor_tool", "extract_text_from_pdf"),
    "extract_pdf_hybrid": ("tools.pdf_extractor_tool", "extract_pdf_hybrid"),
    "extract_docx_documents": ("tools.docx_extractor_tool", "extract_docx_documents"),
    "extract_docx_hybrid": ("tools.docx_extractor_tool", "extract_docx_hybrid"),
    "extract_text_documents": ("tools.text_extractor_tool", "extract_text_documents"),
    "process_audio": ("tools.media_processing", "process_audio"),
    "process_video": ("tools.media_processing", "process_video"),
    "transcribe_documents": ("tools.transcript_tool", "transcribe_documents"),
    "analyze_artifacts": ("tools.visual_understanding", "analyze_artifacts"),
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
