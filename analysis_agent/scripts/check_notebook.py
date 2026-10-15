"""Static check of Kaggle_45min.ipynb: every code cell must parse (IPython magics are stripped)."""
import ast
import json
import sys
from pathlib import Path

nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "kaggle45" / "Kaggle_45min.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    if src.startswith("%%writefile"):
        src = src.split("\n", 1)[1]
    ast.parse(src)
    print(f"cell {i}: ok ({len(src.splitlines())} lines)")
