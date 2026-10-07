"""Command line entry point.

    resume-tailor https://boards.greenhouse.io/acme/jobs/1234
    resume-tailor --text "$(pbpaste)"
    resume-tailor <url> --interview --letter        # ask about gaps, write a letter
    resume-tailor <url> --resume <drive-file-id>    # refresh facts from Drive

    resume-tailor import my_resume.pdf --profile jane   # build a fact base
    resume-tailor style add letter1.pdf letter2.docx    # learn your writing voice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ats import lexicon, linter
from .config import Config, EXAMPLE_MASTER
from .graph import mermaid, run
from .llm import LLM, LLMError
from .state import PipelineState, Priority

GREEN, YELLOW, RED, DIM, BOLD, OFF = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)


def _colour(enabled: bool):
    if enabled:
        return GREEN, YELLOW, RED, DIM, BOLD, OFF
    return ("",) * 6


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        try:
            return SUBCOMMANDS[argv[0]](argv[1:])
        except (LLMError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return tailor_command(argv)


def tailor_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="resume-tailor",
        description="Tailor a resume to a job posting and render an ATS-safe PDF. "
                    "Other commands: `resume-tailor import <resume>` builds a fact base "
                    "from your resume; `resume-tailor style add <letters>` learns your "
                    "writing voice for cover letters.",
    )
    parser.add_argument("url", nargs="?", help="job posting URL")
    parser.add_argument("--text", help="paste the posting text instead of fetching it")
    parser.add_argument("--text-file", type=Path, help="read the posting from a file")
    parser.add_argument("--profile", help="whose files to use: profiles/<name>/")
    parser.add_argument("--resume", help="Drive file id/URL or path to refresh facts from")
    parser.add_argument("--master", type=Path, help="path to the master fact base YAML")
    parser.add_argument("--interview", action="store_true",
                        help="ask about requirements your fact base does not show, and "
                             "save what you confirm")
    parser.add_argument("--letter", action="store_true",
                        help="also write a cover letter in your voice")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--model", help="Anthropic model id")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help="override every step's effort level (lower is cheaper)")
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
                 max_pages=args.max_pages, profile=args.profile or None,
                 master_path=args.master, out_dir=args.out)
    if args.model:
        cfg.model = args.model
    if args.effort:
        cfg.effort = args.effort
    if args.offline:
        cfg.offline = True
    if args.no_bold:
        cfg.bold = False

    g, y, r, dim, bold, off = _colour(not args.no_color and sys.stdout.isatty())

    if not cfg.master_path.exists():
        print(f"{r}no fact base at {cfg.master_path}{off}\nCreate one from your resume: "
              f"resume-tailor import my_resume.pdf"
              + (f" --profile {cfg.profile}" if cfg.profile else ""))
        return 1
    if cfg.master_path == EXAMPLE_MASTER:
        print(f"{y}using the example fact base{off} (a fictional person). Build your own "
              f"from your resume with `resume-tailor import my_resume.pdf`.\n")

    if args.interview:
        if sys.stdin.isatty():
            cfg.ask = lambda q: input(f"\n{bold}?{off} {q}")
        else:
            print(f"{y}--interview needs an interactive terminal; skipping the questions.{off}\n")

    llm = LLM(cfg)
    if llm.offline:
        print(f"{y}running offline{off} -- no ANTHROPIC_API_KEY, so bullets are "
              f"selected and reordered but not rewritten.\n")

    state = PipelineState(
        job_url=args.url, job_text=text, resume_ref=args.resume,
        max_revisions=cfg.max_revisions, coverage_target=cfg.coverage_target,
        phrase_target=args.phrase_target, want_letter=args.letter,
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
        for name in ("ingest", "load_master", "extract"):
            state = state.model_copy(update=nodes[name](state))
            on_step(name, state)
        _print_keywords(state, bold, off, dim)
        return 0

    try:
        state = run(state, cfg, llm, on_step=on_step)
    except LLMError as exc:
        print(f"\n{r}model error:{off} {exc}")
        return 1

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
        _print_accounting(state, cfg, g, y, dim, bold, off, interviewed=cfg.ask is not None)

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
    letter_flags = [i for i in state.letter_issues if i.rule == "fabrication-check"]
    if flags or letter_flags:
        print(f"\n{y}Check these before sending{off} -- not in your fact base:")
        for f in flags:
            print(f"  - {f.split(': ', 1)[-1]}")
        for i in letter_flags:
            print(f"  - cover letter: {i.message}")

    for e in state.errors:
        if e.startswith("layout:"):
            print(f"\n{y}{e}{off}")
    print(f"\n{g}PDF{off} {state.pdf_path}")
    print(f"{dim}Parser check: {Path(state.pdf_path).with_suffix('.txt')}{off}")
    pdf = Path(state.pdf_path)
    print(f"{dim}What changed:  {pdf.with_name(pdf.stem + '_changes.md')}{off}")
    if state.letter_path:
        print(f"{g}Cover letter{off} {state.letter_path}")
        left = [i for i in state.letter_issues if i.severity in ("error", "warning")
                and i.rule != "fabrication-check"]
        for i in left[:5]:
            print(f"  {y}-{off} {i.message}")

    # Terms the model found that no lexicon knew: matched exactly next time.
    if not llm.offline and cfg.writable_fact_base:
        lexicon.remember(cfg.learned_lexicon_path, state.keywords)
    return 0


def _print_accounting(state, cfg, g, y, dim, bold, off, interviewed: bool) -> None:
    from .report import accounting, accounting_line

    rows = accounting(state)
    if not rows:
        return
    print(f"{bold}Requirements{off}  {accounting_line(rows)}")
    unused = [t for t, _, s, _ in rows if s in ("unused", "reframe", "confirmed")]
    if unused:
        print(f"{y}Supported but not used:{off} " + ", ".join(unused[:10]))
    related = [t for t, _, s, _ in rows if s == "adjacent"]
    if related:
        print(f"{y}Related experience (covered in a cover letter):{off} " + ", ".join(related))
    gaps = [t for t, _, s, _ in rows if s in ("gap", "declined")]
    if gaps:
        print(f"{y}True gaps:{off} " + ", ".join(gaps))
    open_items = [t for t, _, s, _ in rows if s == "ask"]
    if open_items and not interviewed:
        print(f"{y}Not yet placed:{off} " + ", ".join(open_items)
              + f"\n{dim}Re-run with --interview to put any you have done on your resume.{off}")
    if (gaps or related) and not state.want_letter:
        print(f"{dim}Add --letter to address these in a cover letter.{off}")


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


# --- resume-tailor import ---------------------------------------------------------------

def import_command(argv: list[str]) -> int:
    from .sources import importer

    parser = argparse.ArgumentParser(
        prog="resume-tailor import",
        description="Build a fact base from your resume (PDF, DOCX, TXT or MD).")
    parser.add_argument("file", type=Path, help="your resume")
    parser.add_argument("--profile", help="save to profiles/<name>/fact_base.yaml")
    parser.add_argument("--out", type=Path, help="save to this path instead")
    parser.add_argument("--force", action="store_true", help="replace an existing fact base")
    parser.add_argument("--model", help="Anthropic model id")
    parser.add_argument("--offline", action="store_true",
                        help="use the rule-based parser instead of the model")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    g, y, r, dim, bold, off = _colour(not args.no_color and sys.stdout.isatty())

    cfg = Config(profile=args.profile or None, master_path=args.out)
    if args.model:
        cfg.model = args.model
    if args.offline:
        cfg.offline = True
    dest = args.out or cfg.master_path
    if dest.resolve() == EXAMPLE_MASTER.resolve():
        from .config import DEFAULT_MASTER

        dest = DEFAULT_MASTER
    if dest.exists() and not args.force:
        print(f"{r}{dest} already exists{off} -- pass --force to replace it, or --profile "
              "<name> to keep a separate one.")
        return 1
    if not args.file.exists():
        raise FileNotFoundError(args.file)

    llm = LLM(cfg)
    print(f"{dim}reading {args.file.name} "
          f"({'rule-based parser, no API key' if llm.offline else cfg.model})...{off}")
    result = importer.import_resume(args.file, llm)
    importer.save(result, dest, args.file)

    m = result.master
    bullets = sum(len(e["bullets"]) for k in ("experience", "projects") for e in m.get(k, []))
    print(f"\n{g}Saved{off} {dest}")
    print(f"  {len(m.get('experience', []))} role(s), {len(m.get('projects', []))} project(s), "
          f"{bullets} bullets, {sum(len(x['items']) for x in m.get('skills', {}).values())} "
          f"skills, {len(m.get('extra_sections', []))} other section(s)")
    print(f"  {result.verified} item(s) checked against the original"
          + (f", {result.dropped_hints} keyword hint(s) dropped as unsupported"
             if result.dropped_hints else ""))
    if result.packs:
        print(f"  lexicon packs: {', '.join(result.packs)}")
    for w in result.warnings:
        print(f"  {y}!{off} {w}")
    if result.unverified:
        print(f"\n{y}{len(result.unverified)} item(s) could not be found in the original{off} "
              f"and are parked under `unverified:` (never used until you move them):")
        for u in result.unverified[:8]:
            print(f"  - {u[:110]}")
    run_hint = f" --profile {args.profile}" if args.profile else ""
    print(f"\nRead the file once, then: resume-tailor <job url>{run_hint}")
    return 0


# --- resume-tailor style --------------------------------------------------------------------

def style_command(argv: list[str]) -> int:
    from .letter import style

    parser = argparse.ArgumentParser(
        prog="resume-tailor style",
        description="Learn your writing voice from cover letters you wrote.")
    parser.add_argument("action", choices=["add", "rebuild", "show"])
    parser.add_argument("files", nargs="*", type=Path,
                        help="letters to add (PDF, DOCX, TXT, MD), or a folder of them")
    parser.add_argument("--profile", help="whose samples: profiles/<name>/")
    parser.add_argument("--model", help="Anthropic model id")
    args = parser.parse_args(argv)

    cfg = Config(profile=args.profile or None)
    if args.model:
        cfg.model = args.model
    if args.action == "show":
        profile = style.load_profile(cfg.style_path)
        if not profile:
            print(f"No style profile yet. Add letters: resume-tailor style add letter.pdf"
                  + (f" --profile {args.profile}" if args.profile else ""))
            return 1
        print(f"Learned from: {', '.join(profile.get('samples', []))}\n")
        print(style.describe(profile))
        return 0

    if args.action == "add":
        if not args.files:
            parser.error("give one or more letters to add")
        saved = style.add_samples(args.files, cfg.samples_dir)
        print(f"Saved {len(saved)} sample(s) to {cfg.samples_dir}")
    samples = style.load_samples(cfg.samples_dir)
    if not samples:
        print("No writing samples yet -- add some with: resume-tailor style add letter.pdf")
        return 1
    llm = LLM(cfg)
    profile = style.build_profile(samples, llm)
    style.save_profile(profile, cfg.style_path)
    print(f"Style profile saved to {cfg.style_path}"
          + ("" if profile.get("voice") else " (measured only -- set ANTHROPIC_API_KEY for "
                                             "the voice notes)"))
    print()
    print(style.describe(profile))
    return 0


SUBCOMMANDS = {"import": import_command, "style": style_command}


if __name__ == "__main__":
    raise SystemExit(main())
