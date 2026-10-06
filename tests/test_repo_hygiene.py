"""Fail the build if a secret or a real fact base ever gets committed."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Real Anthropic keys look like sk-ant-api03-<long base64ish string>. The
# placeholder "sk-ant-..." in .env.example deliberately does not match.
ANTHROPIC_KEY = re.compile(r"sk-ant-[a-z0-9]+-[A-Za-z0-9_\-]{20,}")
GENERIC_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{32,}")
NEVER_TRACKED = {".env", "data/master_resume.yaml", "token.json", "credentials.json"}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.splitlines() if line]


def test_no_private_files_are_tracked():
    tracked = {p.relative_to(ROOT).as_posix() for p in _tracked_files()}
    assert not (tracked & NEVER_TRACKED), tracked & NEVER_TRACKED
    assert not [t for t in tracked if t.startswith("out/")]


def test_no_api_keys_in_tracked_files():
    leaks = []
    for path in _tracked_files():
        if path.suffix in {".pdf", ".png", ".jpg", ".zip"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in (ANTHROPIC_KEY, GENERIC_SECRET):
            if m := pattern.search(text):
                leaks.append(f"{path.relative_to(ROOT)}: {m.group(0)[:16]}...")
    assert not leaks, leaks


def test_gitignore_covers_private_files():
    ignore = (ROOT / ".gitignore").read_text()
    for entry in (".env", "data/master_resume.yaml", "out/", "token.json",
                  "credentials.json"):
        assert entry in ignore, entry


def test_example_fact_base_is_fictional():
    example = (ROOT / "data" / "master_resume.example.yaml").read_text()
    assert "example.com" in example
    assert "555-555-" in example


def test_config_never_shows_the_key():
    from resume_tailor.config import Config

    cfg = Config(api_key="sk-ant-api03-" + "x" * 40)
    assert "sk-ant" not in repr(cfg)
