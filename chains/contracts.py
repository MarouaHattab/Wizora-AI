from collections.abc import Mapping

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError


class StructuredOutputError(ValueError):
    pass


class InsufficientSourceError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def text_output(value):
    if isinstance(value, BaseMessage):
        value = value.text()
    if not isinstance(value, str):
        raise TypeError("Expected text or a LangChain message")
    if not value.strip():
        raise ValueError("The model returned no text")
    return value


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
