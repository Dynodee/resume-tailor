"""Configuration, resolved once and passed around explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = REPO_ROOT / "data" / "master_resume.yaml"
EXAMPLE_MASTER = REPO_ROOT / "data" / "master_resume.example.yaml"


def resolve_master() -> Path:
    """Your fact base if you have made one, otherwise the shipped example.

    data/master_resume.yaml is gitignored: it holds your real contact details
    and work history, and never belongs in the repository. RT_MASTER overrides
    both (the test suite points it at the example).
    """
    if env := os.getenv("RT_MASTER"):
        return Path(env)
    return DEFAULT_MASTER if DEFAULT_MASTER.exists() else EXAMPLE_MASTER


DEFAULT_OUT = REPO_ROOT / "out"


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment.

    Deliberately tiny so there is no extra dependency. A variable already set in
    the shell wins over the file, so `ANTHROPIC_API_KEY=... resume-tailor` still
    overrides whatever is saved.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_dotenv()


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.getenv("RT_MODEL", "claude-opus-5-5"))
    # repr=False so the key never appears if a Config is printed or logged.
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
                                repr=False)
    master_path: Path = field(default_factory=resolve_master)
    out_dir: Path = DEFAULT_OUT
    max_revisions: int = 2
    coverage_target: float = 80.0
    # The renderer shrinks type slightly to hit this, and never by dropping
    # content -- what appears on the page is the tailor node's decision.
    max_pages: int = 2
    # Bold the posting's terms in the summary and bullets (render/emphasis.py).
    bold: bool = True
    # Hard cap so a pathological posting cannot blow up the prompt.
    max_posting_chars: int = 24_000
    timeout_s: float = 120.0
    # When no API key is present the graph still runs end to end using
    # deterministic rules. Output is weaker, but the pipeline is testable.
    offline: bool = False

    def __post_init__(self) -> None:
        # The placeholder from .env.example is not a key; treat it as absent so
        # the run falls back to offline instead of failing on the first call.
        if self.api_key and (self.api_key.endswith("...") or len(self.api_key) < 20):
            self.api_key = None
        if not self.api_key:
            self.offline = True
        self.out_dir.mkdir(parents=True, exist_ok=True)
