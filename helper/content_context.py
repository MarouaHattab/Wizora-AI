"""Bounded, source-labelled generation inputs, with explicit coverage and no tail slicing.

The conservative budget uses UTF-8 bytes: for multilingual inputs a character
limit alone is not a safe upper bound on the number of provider tokens.
"""
import json
import re
import unicodedata

from langchain_core.documents import Document

from chains.contracts import ChainInputError


LOCATOR_KEYS = (
    "source", "source_type", "record_id", "page", "element_id", "paragraph",
    "section", "table_index", "start", "end", "timestamp_precision",
    "content_type", "frame_index", "frame_reference", "line_start", "line_end",
)


def _scope(documents):
    kinds = {document.metadata.get("source_type", "text") for document in documents}
    notices = []
    if "video" in kinds:
        visuals = any(document.metadata.get("content_type") == "visual_description" for document in documents)
        speech = any(document.metadata.get("source_type") == "video" and document.metadata.get("content_type") != "visual_description" for document in documents)
        notices.append(
            "Video evidence includes " + ("transcribed speech and " if speech else "") +
            "sampled frame descriptions; events between sampled frames may be absent."
            if visuals else "Video evidence covers transcribed speech only; visual content was not analyzed."
        )
    if "audio" in kinds:
        notices.append("Audio evidence is transcribed speech; transcription may contain recognition errors.")
    if "pdf" in kinds:
        notices.append("PDF evidence uses extracted page text and preserves original page numbers.")
        if any(document.metadata.get("scanned_page_count", 0) for document in documents):
            notices.append("Some PDF pages require OCR; check extraction warnings for unread pages.")
    if "docx" in kinds:
        notices.append("DOCX evidence uses paragraphs and tables in document order; references identify elements rather than rendered pages.")
        if any(document.metadata.get("image_count", 0) for document in documents):
            notices.append("Embedded images are available as attachments; their visual meaning is not included in text evidence.")
    if "text" in kinds:
        notices.append("Text evidence preserves the extracted source content and available line references.")
    return " ".join(notices)


def _block(text, locator):
    return "SOURCE_LOCATOR: " + json.dumps(locator, ensure_ascii=False, separators=(",", ":")) + "\nSOURCE_TEXT:\n" + text


def prepare_documents(documents, settings, *, workflow=None):
    documents = list(documents)
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ChainInputError("The source contains no educational material")
    maximum = int(getattr(settings, "GENERATION_CONTEXT_CHARACTERS", 24000))
    target = min(maximum, int(getattr(settings, "GENERATION_CHUNK_CHARACTERS", 12000)))
    from chains.builders import revision_source_limit
    workflows = (workflow,) if workflow is not None else ("question_generation", "summarization")
    for selected_workflow in workflows:
        revision_limit = revision_source_limit(settings, selected_workflow)
        if revision_limit is not None:
            target = min(target, revision_limit)
    limit = int(getattr(settings, "GENERATION_MAX_CHUNKS", 64))
    if min(target, limit) <= 0:
        raise ChainInputError("The configured generation model has insufficient room for source context")
    chunks, blocks, citations, size = [], [], [], 0

    def flush():
        nonlocal blocks, citations, size
        if blocks:
            chunks.append({"context": "\n\n".join(blocks), "citations": citations})
            blocks, citations, size = [], [], 0
            if len(chunks) > limit:
                raise ChainInputError(f"The source requires more than {limit} source parts; increase GENERATION_MAX_CHUNKS or upload smaller sections. No source tail was discarded.")

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise ChainInputError("Generation requires source document records")
        text = document.page_content
        locator = {key: document.metadata[key] for key in LOCATOR_KEYS if key in document.metadata}
        locator["source_record_index"] = index
        position = 0
        while position < len(text):
            # Binary search the maximal exact source slice that fits with its locator.
            low, high, best = position + 1, len(text), position
            while low <= high:
                middle = (low + high) // 2
                candidate_locator = {**locator, "source_start": position, "source_end": middle}
                if len(_block(text[position:middle], candidate_locator).encode("utf-8")) <= target:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == position:
                raise ChainInputError("A source locator exceeds the generation context budget")
            end = best
            # Prefer a natural boundary; retain it in the first slice so nothing is lost.
            if end < len(text):
                boundary = max(text.rfind("\n", position, end), text.rfind(". ", position, end))
                if boundary > position + (end - position) // 2:
                    end = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
            reference = {**locator, "source_start": position, "source_end": end}
            block = _block(text[position:end], reference)
            block_size = len(block.encode("utf-8"))
            if blocks and size + 2 + block_size > target:
                flush()
            blocks.append(block)
            citations.append(reference)
            size += block_size + (2 if len(blocks) > 1 else 0)
            position = end
    flush()
    return {
        "source_chunks": chunks,
        "source_citations": [citation for chunk in chunks for citation in chunk["citations"]],
        "source_scope": _scope(documents),
        "context_note": f"All {len(documents)} source records are retained in {len(chunks)} bounded source parts; no extracted text was truncated.",
    }


def prepare_source(state, settings):
    from helper.asset_storage import load_asset_documents
    documents = load_asset_documents(state["asset_id"], context=state["context"]) if state.get("asset_id") else [
        Document(page_content=state["context"], metadata={"source_type": "text"})
    ]
    return prepare_documents(documents, settings, workflow="question_generation" if "question_type" in state else "summarization")


def duplicate_question(question, previous):
    def normalize(value):
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))
    key = normalize(question)
    return any(isinstance(item, str) and normalize(item) == key for item in previous)


def map_config(config, count):
    if count < 2:
        return config
    result = dict(config or {})
    result["metadata"] = {**result.get("metadata", {}), "content_map_private": True}
    return result


def join_sections(sections):
    if len(sections) == 1:
        return sections[0]
    return "\n\n".join(f"### Source part {index} of {len(sections)}\n\n{section}"
                       for index, section in enumerate(sections, 1))
