import ast
from pathlib import Path

UTILS_DIR = Path("src/utils")
INIT_FILE = UTILS_DIR / "__init__.py"

exports = {}

for py_file in UTILS_DIR.glob("*.py"):
    if py_file.name == "__init__.py":
        continue

    with open(py_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(py_file))

    functions = [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    if functions:
        exports[py_file.stem] = functions

lines = []
all_names = []

for module, funcs in exports.items():
    line = f"from .{module} import {', '.join(funcs)}"
    lines.append(line)
    all_names.extend(funcs)

lines.append("\n__all__ = [")
for name in all_names:
    lines.append(f'    "{name}",')
lines.append("]")

INIT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("__init__.py generated successfully.")
