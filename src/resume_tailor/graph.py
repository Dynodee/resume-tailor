"""The workflow as an explicit state machine.

    ingest -> load_master -> extract -> gap_analysis -> tailor -> grammar -+-> render -+-> END
                                                         ^                 |           |
                                                         +---- revise <----+           |
                                                                                       v
                         END <- render_letter <-+- letter_check <- letter <------- (--letter)
                                                |                    ^
                                                +-> revise_letter ---+

The revision edges are the part that makes this agentic rather than a script:
the grammar and coverage results decide whether the tailoring runs again, and
the letter checker decides whether the letter does. Both loops are bounded so
they always terminate. Everything else is deliberately linear, because a step
that cannot fail in an interesting way does not need a router.

The fact base loads before extraction so the person's lexicon packs are in
place when the posting is scanned.

LangGraph is used when available; the same graph runs on a small built-in
executor otherwise, so the package has no hard orchestration dependency.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import Config
from .llm import LLM
from .nodes import extract_keywords, gap_analysis, grammar, ingest, letter, render, tailor
from .state import PipelineState

Node = Callable[[PipelineState], dict[str, Any]]


def build_nodes(cfg: Config, llm: LLM) -> dict[str, Node]:
    return {
        "ingest": lambda s: ingest.ingest(s, cfg),
        "load_master": lambda s: tailor.load_master(s, cfg),
        "extract": lambda s: extract_keywords.extract(s, cfg, llm),
        "gap_analysis": lambda s: gap_analysis.analyze(s, cfg, llm),
        "tailor": lambda s: tailor.tailor(s, cfg, llm),
        "grammar": lambda s: grammar.review_and_revise(s, cfg, llm),
        "revise": lambda s: grammar.bump_revision(s),
        "render": lambda s: render.render(s, cfg),
        "letter": lambda s: letter.write(s, cfg, llm),
        "letter_check": lambda s: letter.check(s, cfg),
        "revise_letter": lambda s: letter.bump_revision(s),
        "render_letter": lambda s: letter.render(s, cfg),
    }


EDGES = [
    ("ingest", "load_master"),
    ("load_master", "extract"),
    ("extract", "gap_analysis"),
    ("gap_analysis", "tailor"),
    ("tailor", "grammar"),
    ("revise", "tailor"),
    ("letter", "letter_check"),
    ("revise_letter", "letter"),
]


def wants_letter(state: PipelineState) -> str:
    return "letter" if state.want_letter and state.pdf_path else "end"


def run(state: PipelineState, cfg: Config, llm: LLM,
        on_step: Callable[[str, PipelineState], None] | None = None) -> PipelineState:
    """Execute the graph. Uses LangGraph if installed, the fallback if not."""
    nodes = build_nodes(cfg, llm)
    try:
        return _run_langgraph(state, nodes, on_step)
    except ImportError:
        return _run_simple(state, nodes, on_step)


def _apply(state: PipelineState, update: dict[str, Any]) -> PipelineState:
    return state.model_copy(update=update or {})


def _run_simple(state: PipelineState, nodes: dict[str, Node],
                on_step: Callable[[str, PipelineState], None] | None) -> PipelineState:
    def step(name: str) -> None:
        nonlocal state
        state = _apply(state, nodes[name](state))
        if on_step:
            on_step(name, state)

    for name in ("ingest", "load_master", "extract", "gap_analysis"):
        step(name)
        if state.errors and name == "ingest" and state.posting is None:
            return state

    while True:
        step("tailor")
        step("grammar")
        if grammar.should_revise(state) == "render":
            break
        step("revise")
    step("render")

    if wants_letter(state) == "letter":
        while True:
            step("letter")
            step("letter_check")
            if letter.should_revise(state) == "render_letter":
                break
            step("revise_letter")
        step("render_letter")
    return state


def _run_langgraph(state: PipelineState, nodes: dict[str, Node],
                   on_step: Callable[[str, PipelineState], None] | None) -> PipelineState:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PipelineState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "ingest")
    for src, dst in EDGES:
        builder.add_edge(src, dst)
    builder.add_conditional_edges(
        "grammar", grammar.should_revise, {"revise": "revise", "render": "render"}
    )
    builder.add_conditional_edges("render", wants_letter, {"letter": "letter", "end": END})
    builder.add_conditional_edges(
        "letter_check", letter.should_revise,
        {"revise_letter": "revise_letter", "render_letter": "render_letter"},
    )
    builder.add_edge("render_letter", END)

    graph = builder.compile()
    # recursion_limit bounds the loops at the framework level as well as in the
    # routers, so a bug in a router cannot spin forever.
    config = {"recursion_limit": 8 + 4 * (state.max_revisions + 1)
              + 4 * (state.letter_max_revisions + 1)}

    final: PipelineState | None = None
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            final = _apply(final or state, update)
            if on_step:
                on_step(node_name, final)
    return final or state


def mermaid() -> str:
    """The graph as a diagram, for the README and for sanity-checking edges."""
    lines = ["flowchart TD", "    START([start]) --> ingest"]
    lines += [f"    {src} --> {dst}" for src, dst in EDGES]
    lines += [
        "    grammar -->|coverage below target or grammar errors| revise",
        "    grammar -->|clean| render",
        "    render -->|cover letter requested| letter",
        "    render -->|no letter| END([done])",
        "    letter_check -->|problems found| revise_letter",
        "    letter_check -->|clean| render_letter",
        "    render_letter --> END",
    ]
    return "\n".join(lines)
