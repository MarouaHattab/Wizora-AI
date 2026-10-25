from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy

from langchain_core.documents import Document

from .ingestion_errors import ChunkingError, EmbeddingError

SPLITTER_VERSION = 2
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", "")


def _validate_config(chunk_size: int, chunk_overlap: int, strategy: str) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ChunkingError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ChunkingError("chunk_overlap must be an integer smaller than chunk_size.")
    if strategy not in {"recursive", "token"}:
        raise ChunkingError("Unsupported chunking strategy; use recursive or token.")


def _largest_prefix(text: str, limit: int, measure: Callable[[str], int]) -> int:
    if measure(text[:1]) > limit:
        raise ChunkingError("A source character exceeds the embedding model token budget.")
    lower, upper = 1, len(text)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if measure(text[:middle]) <= limit:
            lower = middle
        else:
            upper = middle - 1
    return lower


def _partition(
    text: str, limit: int, measure: Callable[[str], int], separators: Sequence[str]
) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=limit,
        chunk_overlap=0,
        # Zero-length whitespace can survive LangChain's overlap removal loop.
        length_function=lambda piece: max(1, measure(piece)) if piece else 0,
        separators=list(separators),
        keep_separator="end",
        strip_whitespace=False,
    )
    pieces = splitter.split_text(text)
    if "".join(pieces) != text:
        raise ChunkingError("The text splitter could not preserve the source text.")
    bounded = []
    for piece in pieces:
        while piece and measure(piece) > limit:
            end = _largest_prefix(piece, limit, measure)
            bounded.append(piece[:end])
            piece = piece[end:]
        if piece:
            bounded.append(piece)
    return bounded


def _suffix_start(text: str, end: int, overlap: int, measure: Callable[[str], int]) -> int:
    lower, upper = 0, end
    while lower < upper:
        middle = (lower + upper) // 2
        if measure(text[middle:end]) <= overlap:
            upper = middle
        else:
            lower = middle + 1
    return lower


def split_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
    *,
    embedding=None,
    strategy: str = "recursive",
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Document]:
    _validate_config(chunk_size, chunk_overlap, strategy)
    if not separators or any(not isinstance(separator, str) for separator in separators):
        raise ChunkingError("separators must contain text boundaries.")
    separators = tuple(separators)
    if separators[-1] != "":
        separators += ("",)

    token_count = getattr(embedding, "token_count", None)
    max_tokens = getattr(embedding, "max_tokens", None)
    if embedding is not None and (
        not callable(token_count)
        or isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ChunkingError("The embedding resource must expose its tokenizer and token limit.")
    if strategy == "token" and embedding is None:
        raise ChunkingError("Token splitting requires the effective embedding resource.")

    try:
        overhead = token_count("") if embedding is not None else 0
        token_measure = (
            (lambda text: max(0, token_count(text) - overhead)) if embedding is not None else None
        )
        if embedding is not None and overhead >= max_tokens:
            raise ChunkingError("The embedding model has no token budget for source text.")
        if strategy == "token":
            limit = min(chunk_size, max_tokens) - overhead
            if chunk_overlap >= limit:
                raise ChunkingError(
                    "chunk_overlap must be smaller than the effective model token budget."
                )
            measure = token_measure
        else:
            limit, measure = chunk_size, len

        result = []
        for record_index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise ChunkingError("Source records must be LangChain Documents.")
            text = document.page_content
            if not text.strip():
                continue
            pieces = _partition(text, limit - chunk_overlap, measure, separators)
            if embedding is not None and strategy == "recursive":
                pieces = [
                    part
                    for piece in pieces
                    for part in (
                        _partition(piece, max_tokens - overhead, token_measure, separators)
                        if token_count(piece) > max_tokens
                        else [piece]
                    )
                ]

            source_position = 0
            chunk_index = 0
            for piece in pieces:
                end = source_position + len(piece)
                if strategy == "recursive":
                    start = max(0, source_position - chunk_overlap)
                else:
                    start = _suffix_start(text, source_position, chunk_overlap, measure)

                def fits(candidate_start: int) -> bool:
                    candidate = text[candidate_start:end]
                    return measure(candidate) <= limit and (
                        embedding is None or token_count(candidate) <= max_tokens
                    )

                if not fits(source_position):
                    raise ChunkingError("A source chunk exceeds the embedding model token budget.")
                if not fits(start):
                    lower, upper = start, source_position
                    while lower < upper:
                        middle = (lower + upper) // 2
                        if fits(middle):
                            upper = middle
                        else:
                            lower = middle + 1
                    start = lower
                chunk = text[start:end]
                if chunk.strip():
                    metadata = deepcopy(document.metadata)
                    metadata.update(
                        {
                            "source_record_index": record_index,
                            "source_start": start,
                            "source_end": end,
                            "chunk_index": chunk_index,
                        }
                    )
                    if document.id is not None:
                        metadata.setdefault("source_document_id", document.id)
                    result.append(Document(page_content=chunk, metadata=metadata))
                    chunk_index += 1
                source_position = end
        return result
    except ChunkingError:
        raise
    except EmbeddingError as exc:
        raise ChunkingError(
            "The source text could not be measured with the embedding tokenizer."
        ) from exc
    except Exception as exc:
        raise ChunkingError("The source documents could not be split.") from exc


def text_splitter(
    text: str,
    type: int = 1,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    *,
    embedding=None,
) -> list[str]:
    if type not in {1, 2}:
        raise ChunkingError(
            "The optional spaCy splitter is unsupported; use recursive or token splitting."
        )
    if not isinstance(text, str):
        raise ChunkingError("Source content must be text.")
    return [
        document.page_content
        for document in split_documents(
            [Document(page_content=text)],
            chunk_size,
            chunk_overlap,
            embedding=embedding,
            strategy="recursive" if type == 1 else "token",
            separators=separators,
        )
    ]
