from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from helper.question_review import assessment_matches, question_assessment, validate_question_draft
from schemas import NonemptyText, Question, StructuredResult


class PublicQuestion(StructuredResult):
    question: NonemptyText
    explanation: NonemptyText
    difficulty: Literal["easy", "medium", "hard"]
    source_file_ids: list[NonemptyText] = Field(min_length=1)
    source_citations: list[dict]
    validation: dict
    review_required: bool


class PublicMCQ(PublicQuestion):
    type: Literal["mcq"]
    options: list[NonemptyText] = Field(min_length=4, max_length=4)
    correct_answer: int = Field(ge=0, le=3, strict=True)


class PublicTrueFalse(PublicQuestion):
    type: Literal["true_false"]
    correct_answer: bool = Field(strict=True)


QUESTION_OUTPUT = TypeAdapter(Annotated[PublicMCQ | PublicTrueFalse, Field(discriminator="type")])


def question_json(draft, state):
    if isinstance(draft, dict):
        draft = {key: draft[key] for key in Question.model_fields if key in draft}
    question = validate_question_draft(state, draft)
    citations = [dict(item) for item in state.get("source_citations", []) if isinstance(item, dict)]
    sources = []
    for citation in citations:
        locations = citation.get("origin_locations", [])
        identifiers = (
            [item.get("source_file_id") for item in locations if isinstance(item, dict)]
            if isinstance(locations, list)
            else []
        )
        compact = citation.get("source_file_ids", [])
        identifiers.extend(compact if isinstance(compact, list) else [])
        identifiers.append(
            citation.get("source_file_id")
            or citation.get("original_asset_id")
            or citation.get("asset_id")
        )
        for identifier in identifiers:
            if isinstance(identifier, str) and identifier and identifier not in sources:
                sources.append(identifier)
    if not sources:
        sources = [state.get("asset_id") or "inline-source"]
    context = state.get("source_context") or state.get("context", "")
    validation = state.get("validation")
    if not assessment_matches(validation, question, context):
        validation = question_assessment(
            question,
            context,
            state.get("critic_feedback"),
            understanding=state.get("content_understanding"),
            previous=state.get("previous_questions", ()),
        )
    common = {
        "question": question.question,
        "explanation": question.explanation,
        "difficulty": question.difficulty,
        "source_file_ids": sources,
        "source_citations": citations,
    }
    common.update(validation=validation, review_required=validation["status"] != "validated")
    if question.question_type == "T/F":
        result = {**common, "type": "true_false", "correct_answer": question.answer == "True"}
    else:
        result = {
            **common,
            "type": "mcq",
            "options": [option[3:] for option in question.options],
            "correct_answer": "ABCD".index(question.answer),
        }
    return QUESTION_OUTPUT.validate_python(result).model_dump(mode="json")
