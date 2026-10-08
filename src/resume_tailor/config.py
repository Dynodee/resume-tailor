"""Configuration, resolved once and passed around explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_MASTER = DATA_DIR / "master_resume.yaml"
EXAMPLE_MASTER = DATA_DIR / "master_resume.example.yaml"
PROFILES_DIR = REPO_ROOT / "profiles"


def profile_dir(profile: str | None) -> Path:
    """Where one person's files live.

    With a profile: profiles/<name>/ holds fact_base.yaml, writing_samples/,
    style.yaml and lexicon.yaml. Without one, the same files live in data/, so
    a single-user setup keeps working exactly as before.
    """
    return PROFILES_DIR / profile if profile else DATA_DIR


def resolve_master(profile: str | None = None) -> Path:
    """The fact base to use, most specific first.

    A profile's fact base, then RT_MASTER (the test suite points it at the
    example), then data/master_resume.yaml, then the shipped example. Real fact
    bases are gitignored: they hold contact details and work history.
    """
    if profile:
        return profile_dir(profile) / "fact_base.yaml"
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
    # One person's files: see profile_dir(). None means the single-user layout.
    profile: str | None = field(default_factory=lambda: os.getenv("RT_PROFILE") or None)
    # Resolved in __post_init__ when left as None (see resolve_master).
    master_path: Path | None = None
    # Where writing samples, style.yaml and lexicon.yaml live; defaults to
    # profile_dir(profile). Tests point it at a temporary folder.
    home_dir: Path | None = None
    out_dir: Path | None = None
    max_revisions: int = 2
    coverage_target: float = 80.0
    # The tailor sizes the resume for this, the renderer shrinks type slightly,
    # and as a last resort the render node cuts the least relevant bullets.
    max_pages: int = 1
    # Bold the posting's terms in the summary and bullets (render/emphasis.py).
    bold: bool = True
    # Hard cap so a pathological posting cannot blow up the prompt.
    max_posting_chars: int = 24_000
    timeout_s: float = 300.0
    # Overrides every node's own effort level when set (low / medium / high /
    # xhigh / max). Lower is cheaper and faster.
    effort: str | None = field(default_factory=lambda: os.getenv("RT_EFFORT") or None)
    # Retry a safety decline on another model server-side (supported models only).
    fallbacks: bool = field(default_factory=lambda: os.getenv("RT_FALLBACKS", "1") != "0")
    # Interview mode: the gap-analysis node calls this with a question and uses
    # the answer. None means never ask (questions are listed in the report).
    ask: Callable[[str], str] | None = field(default=None, repr=False)
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
        if self.master_path is None:
            self.master_path = resolve_master(self.profile)
        if self.out_dir is None:
            self.out_dir = DEFAULT_OUT / self.profile if self.profile else DEFAULT_OUT
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # --- per-person files ---------------------------------------------------------

    @property
    def home(self) -> Path:
        return self.home_dir or profile_dir(self.profile)

    @property
    def samples_dir(self) -> Path:
        """Cover letters the user wrote, used to learn their voice."""
        return self.home / "writing_samples"

    @property
    def style_path(self) -> Path:
        return self.home / "style.yaml"

    @property
    def learned_lexicon_path(self) -> Path:
        return self.home / "lexicon.yaml"

    @property
    def confirmed_path(self) -> Path:
        """Interview answers, kept beside the fact base rather than inside it.

        Writing them into the fact base would mean re-serialising a file the
        user edits by hand, which throws away their comments.
        """
        return self.master_path.with_name(self.master_path.stem + ".confirmed.yaml")

    @property
    def writable_fact_base(self) -> bool:
        """Never write next to the shipped example -- it is tracked in git."""
        return self.master_path.resolve() != EXAMPLE_MASTER.resolve()
