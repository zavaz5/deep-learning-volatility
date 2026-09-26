"""The test set is locked: only src/final_eval.py may read test rows.

These tests read the source tree as text. They fail if any other file subscripts the saved
"test_idx" array, or calls load_raw(), the one function that returns all 80,000 rows.
src/data.py is allowed to define load_raw and to write the split; it never hands out test rows.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIPPED_FOLDERS = {".venv", "venv", "env", "tests", "__pycache__", ".git"}
SOURCE_FILES = [p for p in ROOT.rglob("*.py")
                if not SKIPPED_FOLDERS.intersection(p.relative_to(ROOT).parts)]

NOTEBOOKS = [p for p in ROOT.rglob("*.ipynb")
             if not (SKIPPED_FOLDERS | {".ipynb_checkpoints"}).intersection(p.relative_to(ROOT).parts)]

READS_TEST_IDX = re.compile(r"""\[\s*["']test_idx["']\s*\]""")
CALLS_LOAD_RAW = re.compile(r"\bload_raw\(")


def offenders(pattern: re.Pattern, allowed: set[str]) -> list[str]:
    found = [str(p.relative_to(ROOT)) for p in SOURCE_FILES if pattern.search(p.read_text(encoding="utf-8"))]
    return sorted(set(found) - allowed)


def test_source_tree_was_found():
    assert ROOT / "src" / "data.py" in SOURCE_FILES


def test_only_final_eval_reads_test_indices():
    assert offenders(READS_TEST_IDX, {"src/final_eval.py"}) == []


def test_only_data_and_final_eval_load_the_whole_file():
    assert offenders(CALLS_LOAD_RAW, {"src/data.py", "src/final_eval.py"}) == []


def notebook_code(path: Path) -> str:
    """All code cells of a notebook as one string; markdown cells are prose and may discuss the rule."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_notebooks_do_not_read_test_rows():
    assert NOTEBOOKS, "the research notebook should exist"
    for notebook in NOTEBOOKS:
        code = notebook_code(notebook)
        assert not READS_TEST_IDX.search(code) and not CALLS_LOAD_RAW.search(code), notebook.name
