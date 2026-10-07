"""Learn how a person writes from cover letters they wrote themselves.

    resume-tailor style add my_letter_1.pdf my_letter_2.docx

Two halves, for the same reason keyword extraction has two halves:

- **Measured.** Sentence length, contractions, how often a sentence starts with
  "I", exclamation marks, dashes -- counted in plain Python. Cheap, repeatable,
  and checkable: the letter checker compares a draft against these numbers.
- **Described.** Tone, how the person opens and closes, how they talk about
  their own work, phrases they use and ones they never would -- written by the
  model, because no counter can see them.

The samples themselves are kept too (as text). Showing the model real letters
carries a voice far better than any description of one.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ..llm import LLM
from ..sources import resume_source

VOICE_SYSTEM = """You describe how a person writes, from cover letters they wrote.

Describe the voice, not the content: someone should be able to write a new
letter for a different job that sounds like this person. Be concrete -- "opens
with the role and one sentence on why this company" beats "engaging opening".

signature_phrases: short phrases (2 to 6 words) the person actually uses,
copied exactly, that would fit any letter -- never a phrase tied to one job,
employer or project. avoids: habits this person clearly does not have (for
example "no exclamation marks", "never says 'passionate'")."""


class Voice(BaseModel):
    tone: str
    formality: str
    opening: str
    closing: str
    self_presentation: str
    structure: str
    signature_phrases: list[str]
    avoids: list[str]
    notes: str


# --- measuring --------------------------------------------------------------------------

_CONTRACTION = re.compile(r"\b\w+(?:n't|'re|'ve|'ll|'m|'d)\b|\b(?:it|that|there|what|here|"
                          r"let|who|he|she)'s\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_GREETING = re.compile(r"^\s*(dear|hello|hi|to whom|greetings)\b", re.IGNORECASE)
_SIGNOFF = re.compile(r"^\s*(sincerely|best|regards|thank you|thanks|warmly|respectfully|"
                      r"cheers|yours)\b", re.IGNORECASE)


def body(text: str) -> str:
    """The letter without its address block, greeting and sign-off."""
    text = text.replace("’", "'").replace("\r\n", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    start = 0
    for i, ln in enumerate(lines):
        if _GREETING.match(ln):
            start = i + 1
            break
    end = len(lines)
    for i in range(len(lines) - 1, start - 1, -1):
        if _SIGNOFF.match(lines[i]):
            end = i
            break
    kept = "\n".join(lines[start:end]).strip()
    if not kept:
        return text.strip()
    # Without a greeting, skip leading short lines (name, address, date).
    if start == 0:
        out = kept.split("\n")
        while out and len(out[0].split()) < 8:
            out.pop(0)
        kept = "\n".join(out)
    return kept


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s for s in _SENTENCE_END.split(flat) if len(s.split()) >= 2]


def measure(text: str) -> dict[str, float]:
    text = body(text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    n = max(len(words), 1)
    sents = sentences(text)
    lengths = [len(s.split()) for s in sents] or [0]
    i_start = sum(1 for s in sents if re.match(r"^I(?:'[a-z]+)?\b", s))
    first_person = sum(1 for w in words if w.lower() in {"i", "me", "my", "i'm", "i've", "i'd"})
    per_1000 = lambda pattern: round(1000 * len(re.findall(pattern, text)) / n, 2)  # noqa: E731
    return {
        "words": len(words),
        "sentences": len(sents),
        "avg_sentence_words": round(statistics.mean(lengths), 1),
        "sd_sentence_words": round(statistics.pstdev(lengths), 1),
        "contractions_per_100_words": round(100 * len(_CONTRACTION.findall(text)) / n, 2),
        "i_start_share": round(i_start / max(len(sents), 1), 2),
        "first_person_per_100_words": round(100 * first_person / n, 2),
        "exclamations_per_1000_words": per_1000(r"!"),
        "questions_per_1000_words": per_1000(r"\?"),
        "semicolons_per_1000_words": per_1000(r";"),
        "dashes_per_1000_words": per_1000(r"\s-\s|—|–|--"),
        "parentheses_per_1000_words": per_1000(r"\("),
    }


def average(stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {}
    return {k: round(statistics.mean(s[k] for s in stats), 2) for k in stats[0]}


# --- samples on disk ----------------------------------------------------------------------

SUPPORTED = (".pdf", ".docx", ".txt", ".md")


def resolve_letters(paths: list[Path]) -> list[Path]:
    """Turn what the user typed into letter files.

    A folder means every supported file inside it. A name without its
    extension -- easy to type on Windows, which hides extensions -- is matched
    to the one file with that name and a supported extension.
    """
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(p for p in path.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED)
            if not found:
                raise ValueError(f"{path} is a folder with no PDF, DOCX, TXT or MD files in it")
            out.extend(found)
            continue
        if not path.exists():
            matches = sorted(p for p in path.parent.glob(path.name + ".*")
                             if p.suffix.lower() in SUPPORTED)
            if len(matches) == 1:
                out.append(matches[0])
                continue
            hint = (" -- did you mean one of: " + ", ".join(m.name for m in matches)
                    if matches else "")
            raise FileNotFoundError(f"no such file: {path}{hint}")
        out.append(path)
    return out


def add_samples(paths: list[Path], samples_dir: Path) -> list[Path]:
    """Copy letters into the samples folder as plain text."""
    letters = resolve_letters(paths)
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for path in letters:
        text = resume_source.read_document(path).strip()
        if len(text.split()) < 60:
            raise ValueError(f"{path.name}: under 60 words -- is this the whole letter?")
        target = samples_dir / (re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem) + ".txt")
        n = 2
        while target.exists() and target.read_text(encoding="utf-8").strip() != text:
            target = target.with_name(f"{target.stem}_{n}.txt")
            n += 1
        target.write_text(text + "\n", encoding="utf-8")
        saved.append(target)
    return saved


def load_samples(samples_dir: Path) -> list[tuple[str, str]]:
    if not samples_dir.exists():
        return []
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(samples_dir.glob("*.txt"))]


# --- the profile ----------------------------------------------------------------------------

def build_profile(samples: list[tuple[str, str]], llm: LLM) -> dict[str, Any]:
    stats = average([measure(text) for _, text in samples])
    voice: dict[str, Any] = {}
    if not llm.offline and samples:
        joined = "\n\n".join(f'<letter name="{name}">\n{text[:8000]}\n</letter>'
                             for name, text in samples[:5])
        voice = llm.complete_json(VOICE_SYSTEM, f"Describe how this person writes.\n\n{joined}",
                                  schema=Voice, max_tokens=8000, effort="medium") or {}
        # A signature phrase must really be the person's: drop any not in a sample.
        corpus = " ".join(t.lower() for _, t in samples).replace("’", "'")
        voice["signature_phrases"] = [p for p in voice.get("signature_phrases", [])
                                      if p.lower().replace("’", "'") in corpus]
    return {"samples": [name for name, _ in samples], "stats": stats, "voice": voice}


def save_profile(profile: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Your writing style, learned from {len(profile.get('samples', []))} "
                 "letter(s) you wrote.\n# Rebuilt by `resume-tailor style add` and "
                 "`resume-tailor style rebuild`. Edit the voice notes freely.\n")
        yaml.safe_dump(profile, fh, sort_keys=False, allow_unicode=True, width=100)


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def describe(profile: dict[str, Any]) -> str:
    """The profile as prompt text."""
    lines = []
    voice = profile.get("voice") or {}
    for key in ("tone", "formality", "opening", "closing", "self_presentation", "structure",
                "notes"):
        if voice.get(key):
            lines.append(f"- {key.replace('_', ' ')}: {voice[key]}")
    if voice.get("signature_phrases"):
        lines.append("- phrases they use: " + "; ".join(f'"{p}"' for p in voice["signature_phrases"]))
    if voice.get("avoids"):
        lines.append("- they avoid: " + "; ".join(voice["avoids"]))
    stats = profile.get("stats") or {}
    if stats:
        lines.append(
            f"- measured: sentences average {stats.get('avg_sentence_words')} words; "
            f"{stats.get('contractions_per_100_words')} contractions per 100 words; "
            f"{round(100 * stats.get('i_start_share', 0))}% of sentences start with \"I\"; "
            f"{stats.get('exclamations_per_1000_words')} exclamation marks per 1000 words; "
            f"letters run about {round(stats.get('words', 0))} words")
    return "\n".join(lines) or "(no style notes)"


def distance(letter_text: str, stats: dict[str, float]) -> list[tuple[str, str]]:
    """Where a draft drifts from the measured style: (rule, message) pairs."""
    if not stats:
        return []
    got = measure(letter_text)
    out = []
    want = stats.get("avg_sentence_words") or 0
    if want and abs(got["avg_sentence_words"] - want) / want > 0.3:
        out.append(("style-sentence-length",
                    f"sentences average {got['avg_sentence_words']} words; yours average {want}"))
    c_want, c_got = stats.get("contractions_per_100_words", 0), got["contractions_per_100_words"]
    if c_want < 0.3 and c_got > 1.0:
        out.append(("style-contractions", "you rarely use contractions; the draft uses them"))
    elif c_want > 1.5 and c_got < 0.3:
        out.append(("style-contractions", "you use contractions; the draft has none"))
    if stats.get("exclamations_per_1000_words", 0) == 0 and got["exclamations_per_1000_words"] > 0:
        out.append(("style-exclamations", "you never use exclamation marks; the draft does"))
    i_want = stats.get("i_start_share", 0)
    if abs(got["i_start_share"] - i_want) > 0.25:
        out.append(("style-i-start",
                    f"{round(100 * got['i_start_share'])}% of sentences start with \"I\"; "
                    f"yours: {round(100 * i_want)}%"))
    return out
