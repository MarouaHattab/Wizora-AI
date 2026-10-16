import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import PureWindowsPath

from langchain_core.documents import Document

from chains.contracts import ChainInputError, StructuredOutputError
from .asset_index import RetrievalHit
from .asset_paths import validate_asset_id
from .asset_storage import validate_source_metadata
from .ingestion_errors import InvalidAssetError, InvalidDocumentError


def packed_evidence_text(documents, asset_id: str, index_version: str) -> str:
    """Serialize packed evidence the same way grounded RAG prompts receive it."""
    from chains.RAGChains import evidence_records

    return json.dumps(evidence_records(documents, asset_id, index_version), ensure_ascii=False)


_TOKENS = re.compile(r"\w+(?:['’]\w+)?|[-+]?\d+(?:[.,]\d+)*%?", re.UNICODE)
_NUMBERS = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:%|\b)")
_OPERATORS = re.compile(r"[=<>!≤≥≠+*/^∝≈±∈∉∧∨¬→↔-]+")
_QUALIFIERS = frozenset({
    "not", "no", "never", "neither", "nor", "without", "cannot", "can't", "isn't",
    "aren't", "doesn't", "don't", "won't", "may", "might", "can", "could", "must",
    "should", "only", "except", "unless", "some", "all", "always", "sometimes",
    "usually", "often", "rarely", "approximately", "about", "less", "greater",
    "before", "after", "but", "however", "although", "whereas", "if", "when",
    "under", "least", "most", "minimum", "maximum", "up", "down", "increase",
    "decrease", "optional", "required", "possible", "impossible", "unproven",
})
_ARTICLES = frozenset({"a", "an", "the"})
_CITATION_MARKER = re.compile(r"\[[^\]\r\n]*\]|【|】|(?im:^\s*(?:sources?|references?)\s*:)")


def validate_evidence(documents, *, asset_id: str, index_version: str) -> None:
    """Reject mixed scopes, duplicate identifiers, and malformed provenance."""
    try:
        validate_asset_id(asset_id)
    except InvalidAssetError as exc:
        raise ChainInputError("Invalid evidence asset identifier") from exc
    if not isinstance(index_version, str) or not index_version or len(index_version) > 128:
        raise ChainInputError("An evidence index version is required")
    if not isinstance(documents, (list, tuple)):
        raise ChainInputError("Evidence must be a list of indexed documents")
    identifiers = set()
    for document in documents:
        if not isinstance(document, Document) or not document.page_content.strip():
            raise ChainInputError("Evidence must contain nonblank indexed documents")
        metadata = document.metadata
        identifier = metadata.get("chunk_id")
        if (
            not isinstance(identifier, str) or not identifier or len(identifier) > 128
            or document.id not in (None, identifier)
            or metadata.get("asset_id") != asset_id
            or metadata.get("index_version") != index_version
        ):
            raise ChainInputError("Evidence does not match the active asset and index")
        if identifier in identifiers:
            raise ChainInputError("Evidence chunk identifiers must be unique")
        identifiers.add(identifier)
        content_hash = metadata.get("content_hash")
        if content_hash is not None and content_hash != hashlib.sha256(document.page_content.encode("utf-8")).hexdigest():
            raise ChainInputError("Evidence content does not match its indexed hash")
        try:
            validate_source_metadata(metadata)
        except InvalidDocumentError as exc:
            raise ChainInputError("Evidence source metadata is invalid") from exc
        page = metadata.get("page")
        if page is not None and (type(page) is not int or page < 1):
            raise ChainInputError("Evidence page numbers must be positive integers")
        for key in ("start", "end"):
            value = metadata.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ChainInputError("Evidence timestamps must be finite and nonnegative")
        if metadata.get("start") is not None and metadata.get("end") is not None and metadata["end"] < metadata["start"]:
            raise ChainInputError("Evidence timestamp ranges must be ordered")


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _near_duplicate(left, right, threshold):
    left, right = _normalized(left), _normalized(right)
    if left == right:
        return True
    if tuple(_NUMBERS.findall(left)) != tuple(_NUMBERS.findall(right)):
        return False
    if _OPERATORS.findall(left) != _OPERATORS.findall(right):
        return False
    left_tokens, right_tokens = _TOKENS.findall(left), _TOKENS.findall(right)
    if [word for word in left_tokens if word in _QUALIFIERS] != [word for word in right_tokens if word in _QUALIFIERS]:
        return False
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
    if matcher.ratio() < threshold:
        return False
    # A high lexical overlap cannot establish semantic equivalence. Drop only
    # punctuation/article variations; changed nouns, qualifiers, units, numbers,
    # negation, or newly added sentences remain separate evidence.
    for tag, i, j, x, y in matcher.get_opcodes():
        if tag != "equal" and any(word not in _ARTICLES for word in left_tokens[i:j] + right_tokens[x:y]):
            return False
    return True


def select_evidence(
    hits, *, asset_id: str, index_version: str, max_documents: int,
    max_characters: int, max_tokens: int | None = None, token_count=None,
    near_duplicate_threshold: float = 0.95,
) -> list[Document]:
    """Pack whole, integrity-checked chunks by affinity within serialized budgets."""
    if type(max_documents) is not int or max_documents < 1 or type(max_characters) is not int or max_characters < 1:
        raise ChainInputError("Evidence document and character budgets must be positive")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ChainInputError("The evidence token budget must be positive")
    if token_count is not None and not callable(token_count):
        raise ChainInputError("Evidence token counting must be callable")
    if (
        isinstance(near_duplicate_threshold, bool)
        or not isinstance(near_duplicate_threshold, (int, float))
        or not math.isfinite(near_duplicate_threshold)
        or not 0 <= near_duplicate_threshold <= 1
    ):
        raise ChainInputError("The duplicate threshold must be between zero and one")
    if not isinstance(hits, (list, tuple)):
        raise ChainInputError("Retrieval hits must be a list")
    unique = {}
    for hit in hits:
        if (
            not isinstance(hit, RetrievalHit)
            or isinstance(hit.relevance, bool) or not isinstance(hit.relevance, (int, float))
            or not math.isfinite(hit.relevance) or not 0 <= hit.relevance <= 1
            or type(hit.rank) is not int or hit.rank < 1
        ):
            raise ChainInputError("Retrieval hits require a bounded affinity and positive rank")
        validate_evidence([hit.document], asset_id=asset_id, index_version=index_version)
        identifier = hit.document.metadata["chunk_id"]
        previous = unique.get(identifier)
        if previous is not None:
            if previous.document.page_content != hit.document.page_content or previous.document.metadata != hit.document.metadata:
                raise ChainInputError("A repeated evidence identifier has inconsistent content")
            if previous.relevance >= hit.relevance:
                continue
        unique[identifier] = hit
    selected = []
    for hit in sorted(unique.values(), key=lambda value: (-value.relevance, value.rank)):
        if len(selected) >= max_documents:
            break
        document = hit.document
        if any(_near_duplicate(document.page_content, previous.page_content, near_duplicate_threshold) for previous in selected):
            continue
        packed = packed_evidence_text([*selected, document], asset_id, index_version)
        if len(packed) > max_characters:
            continue
        if max_tokens is not None and token_count is not None:
            count = token_count(packed)
            if type(count) is not int or count < 0:
                raise ChainInputError("The evidence tokenizer returned an invalid count")
            if count > max_tokens:
                continue
        selected.append(document)
    return selected


def _claim_value(claim, key):
    return claim.get(key) if isinstance(claim, Mapping) else getattr(claim, key, None)


def _citation(document, number, asset_id, index_version):
    metadata = document.metadata
    source = metadata.get("source")
    name = PureWindowsPath(source).name if isinstance(source, str) and source else asset_id
    name = "".join(character for character in name if not unicodedata.category(character).startswith("C"))[:255].strip()
    if name in {"", ".", ".."}:
        name = asset_id
    citation = {
        "number": number, "chunk_id": metadata["chunk_id"],
        "asset_id": asset_id, "index_version": index_version, "source": name,
    }
    citation.update({key: metadata[key] for key in (
        "source_type", "modality", "content_type", "page", "start", "end", "section", "paragraph",
        "element_id", "heading_level", "table_index", "segment_id", "record_id", "timestamp_precision",
    ) if metadata.get(key) is not None})
    return citation


def _timestamp(value):
    seconds = max(0, int(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _source_label(citation):
    # Escape source names independently of trusted formatting; never emit a path,
    # Markdown link, HTML tag, or embedded line break from source metadata.
    name = re.sub(r"([\\`*_{}\[\]()#+<>|])", r"\\\1", citation["source"])
    location = []
    if "page" in citation:
        location.append(f"Page {citation['page']}")
    if citation.get("source_type") == "docx":
        section = citation.get("section")
        if section:
            section = " ".join("".join(
                character for character in section if not unicodedata.category(character).startswith("C")
            ).split())[:256]
            section = re.sub(r"([\\`*_{}\[\]()#+<>|])", r"\\\1", section)
            if section:
                location.append("Section: " + section)
        if "paragraph" in citation:
            location.append(f"Paragraph {citation['paragraph']}")
        if "table_index" in citation:
            location.append(f"Table {citation['table_index']}")
        if not any(key in citation for key in ("paragraph", "table_index")):
            element = re.sub(r"([\\`*_{}\[\]()#+<>|])", r"\\\1", citation["element_id"])
            location.append("Element: " + element)
    if citation.get("source_type") == "video":
        location.append("Transcript")
    if "start" in citation:
        time = _timestamp(citation["start"])
        if "end" in citation:
            time += "–" + _timestamp(citation["end"])
        location.append(("Audio window " if citation.get("timestamp_precision") == "audio_window" else "") + time)
    elif "end" in citation:
        location.append("Until " + _timestamp(citation["end"]))
    return name + (" — " + ", ".join(location) if location else "")


def render_cited_answer(
    claims, documents, *, asset_id: str, index_version: str,
    missing_information: str | None = None,
) -> tuple[str, list[dict]]:
    """Render only validated source membership; factual support is checked upstream."""
    validate_evidence(documents, asset_id=asset_id, index_version=index_version)
    if not isinstance(claims, (list, tuple)) or not claims:
        raise StructuredOutputError("A cited answer requires supported claims")
    by_id = {document.metadata["chunk_id"]: document for document in documents}
    references, citations, paragraphs = {}, [], []
    for claim in claims:
        text, identifiers = _claim_value(claim, "text"), _claim_value(claim, "chunk_ids")
        if not isinstance(text, str) or not text.strip() or _CITATION_MARKER.search(text):
            raise StructuredOutputError("Claim text must be nonblank and contain no supplied citation markers")
        if (
            not isinstance(identifiers, (list, tuple)) or not identifiers
            or any(not isinstance(identifier, str) or identifier not in by_id for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise StructuredOutputError("Each claim must cite unique chunks supplied for this answer")
        numbers = []
        for identifier in identifiers:
            if identifier not in references:
                references[identifier] = len(references) + 1
                citations.append(_citation(by_id[identifier], references[identifier], asset_id, index_version))
            numbers.append(references[identifier])
        paragraphs.append(text.strip() + " " + "".join(f"[{number}]" for number in numbers))
    if missing_information is not None:
        if not isinstance(missing_information, str) or not missing_information.strip() or _CITATION_MARKER.search(missing_information):
            raise StructuredOutputError("Missing-information text must contain no citation markers")
        paragraphs.append("The source does not provide: " + missing_information.strip())
    paragraphs.append("Sources:\n" + "\n".join(f"[{citation['number']}] {_source_label(citation)}" for citation in citations))
    return "\n\n".join(paragraphs), citations
