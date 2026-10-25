import unicodedata
from copy import deepcopy

from pydantic import ValidationError

from schemas import Question


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
        questions.append(
            {
                **question,
                "thread_id": deepcopy(thread),
                "asset_id": deepcopy(record.get("asset_id")),
            }
        )
    if not questions:
        raise ValueError("Save at least one completed valid question before starting a quiz.")
    return {"questions": questions, "index": 0, "answers": {}, "submitted": False}


def student_question(quiz):
    question = _current_question(quiz)
    return {
        "question": question["question"],
        "question_type": question["question_type"],
        "choices": _choices(question),
    }


def answer_question(quiz, answer):
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
