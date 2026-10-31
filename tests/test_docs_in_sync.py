"""Fail when the maintained docs stop matching the code they describe.

Each doc names real paths and functions and holds no copied counts, so a
rename, a removal or a Tier 1 change breaks a test instead of staling a doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from drift_lab.config import DOCUMENTED_EVENTS_CSV, ROOT_DIR

DOCS = (ROOT_DIR / "docs" / "DATA_QUALITY.md", ROOT_DIR / "docs" / "EVENT_TIERING_CRITERIA.md")
CODE_FILES = [*(ROOT_DIR / "src").rglob("*.py"), *(ROOT_DIR / "experiments").rglob("*.py")]


def code_spans(doc: Path) -> list[str]:
    return re.findall(r"`([^`\n]+)`", doc.read_text())


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_named_paths_exist(doc):
    paths = [s for s in code_spans(doc) if re.fullmatch(r"[\w.-]+(/[\w.*-]+)+/?|[A-Z_]+\.md", s)]
    assert paths
    missing = [p for p in paths if not (ROOT_DIR / p).exists()]
    assert not missing


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_named_functions_exist(doc):
    names = [s[:-2] for s in code_spans(doc) if re.fullmatch(r"\w+\(\)", s)]
    assert names
    source = "\n".join(path.read_text() for path in CODE_FILES)
    assert not [name for name in names if f"def {name}(" not in source]


def test_tier_1_list_matches_catalogue():
    listed = set(re.findall(r"^\| `(EVT-[\d-]+)` \|", DOCS[1].read_text(), flags=re.MULTILINE))
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV)
    assert listed == set(events.loc[events["tier"] == 1, "event_id"])
