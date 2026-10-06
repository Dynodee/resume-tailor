"""Run every test against the fictional example fact base.

Without this, a developer who has created their own data/master_resume.yaml
would run the suite against their real resume -- and the expectations here are
written for the example. Profile and effort settings from a developer's .env are
neutralised for the same reason (an empty value stops .env from filling it in).
"""

import os
from pathlib import Path

import pytest

os.environ["RT_MASTER"] = str(
    Path(__file__).resolve().parents[1] / "data" / "master_resume.example.yaml"
)
os.environ["RT_PROFILE"] = ""
os.environ["RT_EFFORT"] = ""


@pytest.fixture(autouse=True)
def _fresh_lexicon():
    """Lexicon packs are process-wide; never let one test's packs leak into the next."""
    yield
    from resume_tailor.ats import lexicon

    lexicon.reset()
