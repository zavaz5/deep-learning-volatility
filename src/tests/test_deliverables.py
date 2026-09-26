"""Tests for the generated deliverables: the README results table, the traceability of every number, and the executed notebook."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README, MANIFEST = ROOT / "README.md", ROOT / "results" / "numbers_manifest.json"
needs_deliverables = pytest.mark.skipif(not (README.exists() and MANIFEST.exists()), reason="build first: python -m src.readme")


@needs_deliverables
def test_every_number_traces_to_results():
    from src.report import check
    assert check() == []


@needs_deliverables
def test_results_table_explains_every_large_difference():
    from src.ledger import Ledger
    from src.readme import check_comments, results_rows
    rows = results_rows(Ledger("test"))
    check_comments(rows)
    assert sum(1 for r in rows if r["paper_value"] is not None) >= 20


def test_the_research_notebook_is_executed_without_errors():
    import json
    cells = json.loads((ROOT / "DeepLearningVolatility.ipynb").read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert code_cells and all(c.get("outputs") for c in code_cells), "run the notebook top to bottom before committing it"
    assert not [o for c in code_cells for o in c["outputs"] if o.get("output_type") == "error"]
