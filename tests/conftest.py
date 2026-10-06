"""Run every test against the fictional example fact base.

Without this, a developer who has created their own data/master_resume.yaml
would run the suite against their real resume -- and the expectations here are
written for the example.
"""

import os
from pathlib import Path

os.environ["RT_MASTER"] = str(
    Path(__file__).resolve().parents[1] / "data" / "master_resume.example.yaml"
)
