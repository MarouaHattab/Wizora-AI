"""Pure quiz transitions over a detached snapshot of saved questions.

The complete question records belong in server-side session state. Only the
explicit projection returned by ``student_question`` should be sent to a student
before submission. Transitions return copies so a previous attempt, or later
changes to the saved questions, cannot change an existing score.
"""

from copy import deepcopy
import unicodedata

from pydantic import ValidationError

from chains.schemes.QuestionSchem import Question


def _normalized(text):
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


def _choices(question):
    if question["question_type"] == "T/F":
        return [("True", "True"), ("False", "False")]
    return [(option, label) for label, option in zip("ABCD", question["options"])]


def _current_question(quiz):
    if not quiz or not quiz.get("questions"):
        raise ValueError("Start a quiz with saved questions first.")
    return quiz["questions"][quiz["index"]]


def start_quiz(records):
    """Snapshot valid completed saved records, discarding duplicate questions.

    Input records contain ``thread_id``, ``asset_id`` and a schema-valid
    ``result``. A status field is optional; if present it must be ``completed``.
    The first valid occurrence of a thread or normalized stem/type/options wins.
    Raises ValueError if no usable questions remain.
    """
    questions = []
    threads = set()
    contents = set()
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if "status" in record and record["status"] != "completed":
            continue
        try:
            question = Question.model_validate(record.get("result")).model_dump()
        except (ValidationError, TypeError, ValueError):
            continue
        thread = record.get("thread_id")
        content = (
            question["question_type"],
            _normalized(question["question"]),
            tuple(_normalized(option) for option in question["options"] or []),
        )
        if (thread and thread in threads) or content in contents:
            continue
        if thread:
            threads.add(thread)
        contents.add(content)
        questions.append({
            **question,
            "thread_id": deepcopy(thread),
            "asset_id": deepcopy(record.get("asset_id")),
        })
    if not questions:
        raise ValueError("Save at least one completed valid question before starting a quiz.")
    return {"questions": questions, "index": 0, "answers": {}, "submitted": False}


def student_question(quiz):
    """Return only the current prompt, type and (display, value) choices."""
    question = _current_question(quiz)
    return {
        "question": question["question"],
        "question_type": question["question_type"],
        "choices": _choices(question),
    }


def answer_question(quiz, answer):
    """Return a copy with a canonical answer, or clear the answer with None.

    MCQ answers are exactly A/B/C/D; True/False answers are strings True/False.
    Submitted attempts cannot be edited. Invalid choices raise ValueError.
    """
    question = _current_question(quiz)
    if quiz["submitted"]:
        raise ValueError("This quiz has been submitted. Start a new attempt to answer again.")
    if answer is not None and (
        not isinstance(answer, str) or answer not in {value for _, value in _choices(question)}
    ):
        raise ValueError("Choose one of the available answers.")
    updated = deepcopy(quiz)
    if answer is None:
        updated["answers"].pop(updated["index"], None)
    else:
        updated["answers"][updated["index"]] = answer
    return updated


def move_question(quiz, direction):
    """Move by one question within bounds, preserving existing selections.

    Direction is 1/-1 or next/previous. Navigation is also allowed after
    submission so a caller can review the same saved attempt.
    """
    _current_question(quiz)
    if direction == "next":
        step = 1
    elif direction == "previous":
        step = -1
    elif type(direction) is int and direction in (-1, 1):
        step = direction
    else:
        raise ValueError("Direction must be next, previous, 1 or -1.")
    updated = deepcopy(quiz)
    updated["index"] = max(0, min(len(updated["questions"]) - 1, updated["index"] + step))
    return updated


def submit_quiz(quiz):
    """Freeze the attempt and calculate counts and percentage of all questions.

    Unanswered items count toward the total but are separate from incorrect
    answers. Repeated submissions return the same result in a detached copy.
    """
    _current_question(quiz)
    updated = deepcopy(quiz)
    correct = incorrect = unanswered = 0
    for index, question in enumerate(updated["questions"]):
        answer = updated["answers"].get(index)
        if answer is None:
            unanswered += 1
        elif answer == question["answer"]:
            correct += 1
        else:
            incorrect += 1
    total = len(updated["questions"])
    updated["submitted"] = True
    updated["score"] = {
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "total": total,
        "percentage": correct * 100 / total,
    }
    return updated
