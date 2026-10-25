"""Check local import boundaries without importing providers or starting services."""

import argparse
import ast
import json
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("chains", "controllers", "frontend", "helper", "llm", "routes", "schemas", "tools")


def import_statements(nodes):
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from import_statements(ast.iter_child_nodes(node))


def audit(root=ROOT):
    paths = [root / "main.py", root / "gradio_ui.py"]
    paths.extend(path for name in PACKAGES for path in (root / name).rglob("*.py"))
    modules = {
        ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__"): path
        for path in paths
    }
    graph = {name: set() for name in modules}
    violations = []
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in import_statements(tree.body):
            if isinstance(node, ast.Import):
                imports = [item.name for item in node.names]
            else:
                prefix = (
                    package.split(".")[: len(package.split(".")) - node.level + 1]
                    if node.level
                    else []
                )
                base = ".".join([*prefix, *([node.module] if node.module else [])])
                imports = [base]
                imports.extend(
                    base + "." + item.name
                    for item in node.names
                    if base + "." + item.name in modules
                )
            for target in imports:
                if target.split(".")[0] in PACKAGES:
                    if target not in modules:
                        violations.append(f"{name}:{node.lineno}: unresolved local import {target}")
                    elif target != name:
                        graph[name].add(target)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(item.name == "*" for item in node.names):
                    violations.append(f"{name}:{node.lineno}: wildcard import")
                if name.startswith("frontend.") and node.module == "gradio_ui":
                    violations.append(f"{name}:{node.lineno}: frontend imports its executable")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
                    if (
                        owner.value.id == "sys"
                        and owner.attr == "path"
                        and node.func.attr in {"insert", "append", "extend"}
                    ):
                        violations.append(f"{name}:{node.lineno}: application mutates sys.path")
        if path.name == "__init__.py" and not any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            for node in tree.body
        ):
            violations.append(f"{name}: package has no explicit public API")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        violations.append("Import-time cycle: " + " -> ".join(error.args[1]))
    return {
        "passed": not violations,
        "module_count": len(modules),
        "import_time_edges": sum(map(len, graph.values())),
        "scope": "Application modules and import-time dependencies; deferred call graphs are not cycles checked here.",
        "violations": violations,
        "imports": {name: sorted(targets) for name, targets in sorted(graph.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "imports"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
