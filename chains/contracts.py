from collections.abc import Mapping

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError


class ChainInputError(ValueError):
    pass


class StructuredOutputError(ValueError):
    pass


class TextOutputError(ValueError):
    pass


class ProviderExecutionError(RuntimeError):
    pass


class InsufficientSourceError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def text_fragment(value):
    if isinstance(value, BaseMessage):
        value = value.content
    elif not isinstance(value, str):
        raise TextOutputError("Expected text or a LangChain message")
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TextOutputError("Expected text or a LangChain message")
    parts = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            kind = block.get("type")
            if not isinstance(kind, str):
                raise TextOutputError("The model returned an unsupported text block")
            if kind in {"reasoning", "thinking"}:
                continue
            if kind not in {"text", "output_text"} or not isinstance(block.get("text"), str):
                raise TextOutputError("The model returned an unsupported text block")
            parts.append(block["text"])
        else:
            raise TextOutputError("The model returned an unsupported text block")
    return "".join(parts)


def text_output(value):
    result = text_fragment(value)
    if not result.strip():
        raise TextOutputError("The model returned no text")
    return result


def structured_output(schema, value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise StructuredOutputError("The model did not return the required structured result")
    try:
        return schema.model_validate(dict(value))
    except (ValidationError, TypeError) as exc:
        raise StructuredOutputError(f"Invalid {schema.__name__} output") from exc


def requested_question(question, question_type):
    if question.question_type != question_type:
        raise StructuredOutputError("Generated question type does not match the requested type")
    return question
