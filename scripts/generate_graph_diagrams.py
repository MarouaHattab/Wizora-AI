"""Render Wizora AI's compiled LangGraph workflows as PNG diagrams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.QuestionAnswerGraphController import compile_qa_graph
from controllers.QuestionGenGraphController import compile_question_graph
from controllers.SummarizerGenGraphController import compile_summary_graph


WORKFLOWS = (
    ("question_generation", compile_question_graph, "question_generation_graph.png"),
    ("summarization", compile_summary_graph, "summarization_graph.png"),
    ("question_answering", compile_qa_graph, "question_answering_graph.png"),
)


def render(output_dir: Path) -> list[Path]:
    """Compile each production graph and render its native LangGraph topology."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        for workflow_name, compile_graph, filename in WORKFLOWS:
            compiled_graph = compile_graph(checkpointer)
            png = compiled_graph.get_graph().draw_mermaid_png()
            destination = output_dir / filename
            destination.write_bytes(png)
            if png[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"{workflow_name} did not produce a PNG image")
            generated.append(destination)
            print(f"generated {workflow_name}: {destination}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "architecture",
        help="Directory in which to write the generated PNG files.",
    )
    args = parser.parse_args()
    render(args.output_dir.resolve())


if __name__ == "__main__":
    main()
