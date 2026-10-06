"""The workflow as an explicit state machine.

    ingest -> extract -> load_master -> tailor -> grammar -+-> render -> END
                                          ^                |
                                          +---- revise <---+

The revision edge is the part that makes this agentic rather than a script: the
grammar and coverage results decide whether the tailoring runs again, and the
loop is bounded so it always terminates. Everything else is deliberately linear,
because a step that cannot fail in an interesting way does not need a router.

LangGraph is used when available; the same graph runs on a small built-in
executor otherwise, so the package has no hard orchestration dependency.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import Config
from .llm import LLM
from .nodes import extract_keywords, grammar, ingest, render, tailor
from .state import PipelineState

Node = Callable[[PipelineState], dict[str, Any]]


def build_nodes(cfg: Config, llm: LLM) -> dict[str, Node]:
    return {
        "ingest": lambda s: ingest.ingest(s, cfg),
        "extract": lambda s: extract_keywords.extract(s, cfg, llm),
        "load_master": lambda s: tailor.load_master(s, cfg),
        "tailor": lambda s: tailor.tailor(s, cfg, llm),
        "grammar": lambda s: grammar.review_and_revise(s, cfg, llm),
        "revise": lambda s: grammar.bump_revision(s),
        "render": lambda s: render.render(s, cfg),
    }


EDGES = [
    ("ingest", "extract"),
    ("extract", "load_master"),
    ("load_master", "tailor"),
    ("tailor", "grammar"),
    ("revise", "tailor"),
]


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
    order = ["ingest", "extract", "load_master"]
    for name in order:
        state = _apply(state, nodes[name](state))
        if on_step:
            on_step(name, state)
        if state.errors and name == "ingest" and state.posting is None:
            return state

    while True:
        state = _apply(state, nodes["tailor"](state))
        if on_step:
            on_step("tailor", state)
        state = _apply(state, nodes["grammar"](state))
        if on_step:
            on_step("grammar", state)
        if grammar.should_revise(state) == "render":
            break
        state = _apply(state, nodes["revise"](state))
        if on_step:
            on_step("revise", state)

    state = _apply(state, nodes["render"](state))
    if on_step:
        on_step("render", state)
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
    builder.add_edge("render", END)

    graph = builder.compile()
    # recursion_limit bounds the loop at the framework level as well as in the
    # router, so a bug in should_revise cannot spin forever.
    config = {"recursion_limit": 4 + 4 * (state.max_revisions + 1)}

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
        "    render --> END([done])",
    ]
    return "\n".join(lines)
