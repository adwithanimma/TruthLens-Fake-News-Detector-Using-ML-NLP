"""Static-asset integrity tests. Run: python -m pytest tests -q

A stray brace in static/js/app.js silently disables the entire frontend: the
page renders, but no event handler binds and the status pill stays on
"Checking model…". Nothing else in the suite would notice, so the JS is
parsed here and the template's asset references are checked.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from flask import render_template

from app import ROOT, app

JS = ROOT / "static" / "js" / "app.js"
CSS = ROOT / "static" / "css" / "style.css"
TEMPLATE = ROOT / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_app_js_parses():
    result = subprocess.run(
        ["node", "--check", str(JS)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_brace_balance_is_a_strict_subset_of_the_real_check():
    """Cheap guard used when node is unavailable."""
    source = JS.read_text(encoding="utf-8")
    assert source.count("{") == source.count("}"), "unbalanced braces in app.js"
    assert source.count("(") == source.count(")"), "unbalanced parens in app.js"


def test_css_braces_balanced():
    source = CSS.read_text(encoding="utf-8")
    assert source.count("{") == source.count("}")


def test_no_double_scaling_in_percentage_rendering():
    """Regression: scores were multiplied by 100 before pct(), showing 7040%."""
    source = JS.read_text(encoding="utf-8")
    assert not re.search(r"scores\.(real|fake)\s*\*\s*100", source), (
        "percentage values must be passed to pct() unscaled"
    )
    assert "const real = result.scores.real;" in source


def test_template_references_existing_assets():
    """Render the real page so Jinja's url_for/asset_version actually run."""
    with app.test_request_context("/"):
        html = render_template("index.html")
    soup = BeautifulSoup(html, "lxml")
    refs = [t.get("href") for t in soup.find_all("link") if t.get("href")]
    refs += [s.get("src") for s in soup.find_all("script") if s.get("src")]
    static_refs = [r for r in refs if r and "/static/" in r]
    assert static_refs, "template references no static assets"
    for ref in static_refs:
        assert "?" in ref, f"{ref} is missing the cache-busting version param"
        rel = ref.split("?")[0].split("/static/")[-1]
        assert (ROOT / "static" / rel).exists(), f"{ref} does not exist on disk"


def test_every_id_referenced_in_js_exists_in_template():
    """Catches renamed DOM ids that would make the JS silently no-op."""
    html = TEMPLATE.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    template_ids = set(re.findall(r'id="([^"]+)"', html))
    used_ids = set(re.findall(r'\$\("([^"]+)"\)', js))
    missing = used_ids - template_ids
    assert not missing, f"app.js references ids absent from the template: {sorted(missing)}"
