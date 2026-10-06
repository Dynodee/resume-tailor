"""Command line entry point.

    resume-tailor https://boards.greenhouse.io/acme/jobs/1234
    resume-tailor --text "$(pbpaste)"
    resume-tailor <url> --resume <drive-file-id>   # refresh facts from Drive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ats import linter, scorer
from .config import Config
from .graph import mermaid, run
from .llm import LLM
from .state import PipelineState, Priority

GREEN, YELLOW, RED, DIM, BOLD, OFF = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)


def _colour(enabled: bool):
    if enabled:
        return GREEN, YELLOW, RED, DIM, BOLD, OFF
    return ("",) * 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resume-tailor",
        description="Tailor a resume to a job posting and render an ATS-safe PDF.",
    )
    parser.add_argument("url", nargs="?", help="job posting URL")
    parser.add_argument("--text", help="paste the posting text instead of fetching it")
    parser.add_argument("--text-file", type=Path, help="read the posting from a file")
    parser.add_argument("--resume", help="Drive file id/URL or path to refresh facts from")
    parser.add_argument("--master", type=Path, help="path to the master fact base YAML")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--model", help="Anthropic model id")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--target", type=float, default=80.0,
                        help="required-term coverage target, percent")
    parser.add_argument("--phrase-target", type=float, default=70.0,
                        help="exact-phrase coverage target, percent")
    parser.add_argument("--no-bold", action="store_true",
                        help="don't bold the posting's terms in the summary and bullets")
    parser.add_argument("--max-pages", type=int, default=2,
                        help="shrink type slightly until the PDF fits this many pages")
    parser.add_argument("--offline", action="store_true",
                        help="run without the model (deterministic selection only)")
    parser.add_argument("--keywords-only", action="store_true",
                        help="stop after extraction and print the keyword table")
    parser.add_argument("--graph", action="store_true", help="print the graph and exit")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    if args.graph:
        print(mermaid())
        return 0

    text = args.text
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    if not args.url and not text:
        parser.error("give a job posting URL, or --text / --text-file")

    cfg = Config(max_revisions=args.max_revisions, coverage_target=args.target,
                 max_pages=args.max_pages)
    if args.model:
        cfg.model = args.model
    if args.master:
        cfg.master_path = args.master
    if args.out:
        cfg.out_dir = args.out
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if args.offline:
        cfg.offline = True
    if args.no_bold:
        cfg.bold = False

    g, y, r, dim, bold, off = _colour(not args.no_color and sys.stdout.isatty())

    from .config import EXAMPLE_MASTER

    if cfg.master_path == EXAMPLE_MASTER:
        print(f"{y}using the example fact base{off} (a fictional person). Copy "
              f"data/master_resume.example.yaml to data/master_resume.yaml and fill "
              f"in your own background.\n")

    llm = LLM(cfg)
    if llm.offline:
        print(f"{y}running offline{off} -- no ANTHROPIC_API_KEY, so bullets are "
              f"selected and reordered but not rewritten.\n")

    state = PipelineState(
        job_url=args.url, job_text=text, resume_ref=args.resume,
        max_revisions=cfg.max_revisions, coverage_target=cfg.coverage_target,
        phrase_target=args.phrase_target,
    )

    seen = 0

    def on_step(node: str, s: PipelineState) -> None:
        nonlocal seen
        for line in s.log[seen:]:
            print(f"{dim}{line}{off}")
        seen = len(s.log)

    if args.keywords_only:
        from .graph import build_nodes

        nodes = build_nodes(cfg, llm)
        for name in ("ingest", "extract"):
            state = state.model_copy(update=nodes[name](state))
            on_step(name, state)
        _print_keywords(state, bold, off, dim)
        return 0

    state = run(state, cfg, llm, on_step=on_step)

    if state.pdf_path is None:
        print(f"\n{r}failed{off}")
        for err in state.errors:
            print(f"  - {err}")
        return 1

    cov = state.coverage
    print()
    if cov:
        colour = g if cov.attainable_score >= cfg.coverage_target else y
        print(f"{bold}Coverage{off}  overall {cov.score}%  |  "
              f"required terms your fact base supports {colour}{cov.attainable_score}%{off}")
        if cov.phrase_items:
            from .ats.phrases import summary_line

            pcol = g if cov.phrase_score >= state.phrase_target else y
            print(f"{bold}Exact phrases{off}  {pcol}{summary_line(cov)}{off}")
            other = [p.text for p in cov.phrases("concept") if not p.soft]
            if other:
                print(f"{y}Said in other words only:{off} " + ", ".join(other))
        # One entry per term, however the posting capitalised it.
        unattainable_map: dict[str, str] = {}
        for term in [*cov.unattainable,
                     *(p.text for p in cov.phrases("unsupported") if not p.soft)]:
            unattainable_map.setdefault(term.lower(), term)
        unattainable = set(unattainable_map.values())
        gaps = [k for k in cov.missing(Priority.REQUIRED)
                if k.category != "soft" and k.term.lower() not in unattainable_map]
        if gaps:
            print(f"{y}Supported but not used:{off} " + ", ".join(k.term for k in gaps[:10]))
        if unattainable:
            print(f"{y}Not in your fact base:{off} " + ", ".join(sorted(unattainable,
                                                                   key=str.lower)))
            print(f"{dim}Add any that are true to master_resume.yaml; the rest are cover-letter "
                  f"or interview topics.{off}")

    hard = [i for i in state.grammar_issues if i.severity == "error"]
    if hard:
        print(f"\n{r}Grammar errors remaining:{off}")
        print(linter.summarize(hard))
    elif state.grammar_issues:
        warn = [i for i in state.grammar_issues if i.severity == "warning"]
        if warn:
            print(f"\n{y}Writing warnings:{off}")
            for i in warn[:8]:
                print(f"  - {i.location}: {i.message}")
        else:
            print(f"\n{dim}{len(state.grammar_issues)} minor style note(s) -- "
                  f"listed in the changes file.{off}")

    flags = [e for e in state.errors if e.startswith("fabrication check")]
    if flags:
        print(f"\n{y}Check these before sending{off} -- terms on the page that are "
              f"not in your fact base:")
        for f in flags:
            print(f"  - {f.split(': ', 1)[-1]}")

    for e in state.errors:
        if e.startswith("layout:"):
            print(f"\n{y}{e}{off}")
    print(f"\n{g}PDF{off} {state.pdf_path}")
    print(f"{dim}Parser check: {Path(state.pdf_path).with_suffix('.txt')}{off}")
    pdf = Path(state.pdf_path)
    print(f"{dim}What changed:  {pdf.with_name(pdf.stem + '_changes.md')}{off}")
    return 0


def _print_keywords(state: PipelineState, bold: str, off: str, dim: str) -> None:
    print()
    if state.phrases:
        print(f"{bold}EXACT PHRASES{off} (matched word for word)")
        for ph in state.phrases:
            tag = "soft" if ph.soft else ph.priority.value
            print(f"  {ph.text:<46} {dim}{tag:<9} {ph.source}{off}")
        print()
    for priority in (Priority.REQUIRED, Priority.PREFERRED, Priority.MENTIONED):
        group = [k for k in state.keywords if k.priority is priority]
        if not group:
            continue
        print(f"{bold}{priority.value.upper()}{off} ({len(group)})")
        for k in group:
            print(f"  {k.term:<34} {dim}{k.category:<11} x{k.frequency}{off}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
