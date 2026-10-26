"""Render Wizora AI's compiled LangGraph workflows as PNG diagrams using IPython.display."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from IPython.display import Image, display
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


def render(
    output_dir: Path,
    theme: str | None = None,
    show_display: bool = True,
) -> list[Path]:
    """Compile each production graph and render its native LangGraph topology using IPython.display."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    render_kwargs = {}
    if theme and theme != "default":
        render_kwargs["frontmatter_config"] = {"config": {"theme": theme}}

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        for workflow_name, compile_graph, filename in WORKFLOWS:
            compiled_graph = compile_graph(checkpointer)
            png_bytes = compiled_graph.get_graph().draw_mermaid_png(**render_kwargs)

            if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError(f"{workflow_name} did not produce a valid PNG image")

            # Wrap and display via IPython.display.Image
            image = Image(png_bytes)
            if show_display:
                try:
                    display(image)
                except Exception:
                    pass

            destination = output_dir / filename
            destination.write_bytes(image.data)
            generated.append(destination)
            print(f"[OK] Generated {workflow_name} ({len(image.data)} bytes) -> {destination}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "architecture",
        help="Directory in which to write the generated PNG files.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        choices=["default", "neutral", "forest", "dark", "base"],
        help="Mermaid theme to use for rendering diagrams.",
    )
    args = parser.parse_args()
    render(args.output_dir.resolve(), theme=args.theme)


if __name__ == "__main__":
    main()
