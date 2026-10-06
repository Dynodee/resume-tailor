"""The deterministic half of keyword extraction.

An LLM alone is a bad ATS simulator: it paraphrases, and paraphrase is exactly
what ATS keyword matching punishes. So the canonical terms live here, in a
dictionary with explicit aliases, and the LLM is only used to catch terms this
file has never heard of.

Aliases matter more than the canonical form. A posting that says "LLMs" and a
resume that says "large language models" is a match to a human and a miss to a
naive matcher; the alias table is what closes that gap in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

# canonical term -> aliases (all matched case-insensitively, word-boundary aware)
LEXICON: dict[str, dict] = {}


def _add(term: str, category: str, *aliases: str) -> None:
    LEXICON[term] = {"category": category, "aliases": list(aliases)}


# --- languages, frameworks, runtimes -----------------------------------------
_add("Python", "skill", "python3", "py")
_add("SQL", "skill", "structured query language", "t-sql", "ansi sql")
_add("MDX", "skill", "multidimensional expressions")
_add("R", "skill")
_add("Java", "skill")
_add("JavaScript", "skill", "js", "ecmascript")
_add("TypeScript", "skill", "ts")
_add("Scala", "skill")
_add("Go", "skill", "golang")
_add("Bash", "skill", "shell scripting", "shell")
_add("FastAPI", "tool")
_add("Flask", "tool")
_add("Django", "tool")
_add("Pydantic", "tool")
_add("pytest", "tool", "py.test")
_add("Pandas", "tool", "pandas dataframe")
_add("NumPy", "tool")
_add("scikit-learn", "tool", "sklearn", "scikit learn")
_add("PyTorch", "tool", "torch")
_add("TensorFlow", "tool", "tf")

# --- AI / LLM / agentic -------------------------------------------------------
_add("LLM", "skill", "large language model", "large language models", "llms",
     "foundation model", "foundation models")
_add("Agentic AI", "skill", "agentic", "agentic systems", "agentic workflows",
     "ai agents", "autonomous agents", "agent systems")
_add("Multi-Agent Systems", "skill", "multi agent", "multiagent",
     "multi-agent orchestration", "agent orchestration")
_add("LangGraph", "tool", "lang graph")
_add("LangChain", "tool", "lang chain")
_add("Model Context Protocol (MCP)", "tool", "mcp", "model context protocol")
_add("Prompt Engineering", "skill", "prompt design", "prompting", "prompt optimization")
_add("RAG", "skill", "retrieval augmented generation", "retrieval-augmented generation")
_add("Vector Database", "tool", "vector store", "embeddings database", "pinecone",
     "weaviate", "chroma", "pgvector", "faiss")
_add("Embeddings", "skill", "embedding models", "semantic search")
_add("Fine-Tuning", "skill", "finetuning", "fine tuning", "lora", "peft")
_add("LLM Evaluation", "skill", "evals", "eval harness", "model evaluation",
     "llm evals", "offline evaluation")
_add("Function Calling", "skill", "tool use", "tool calling", "tool-use")
_add("AI Safety", "skill", "guardrails", "responsible ai", "ai governance",
     "model safety", "content filtering")
_add("Machine Learning", "skill", "ml", "statistical modeling", "predictive modeling")
_add("NLP", "skill", "natural language processing", "text analytics")
_add("OpenAI API", "tool", "openai", "gpt-4", "gpt")
_add("Anthropic API", "tool", "anthropic", "claude")

# --- data platforms, warehouses, pipelines -----------------------------------
_add("Snowflake", "tool")
_add("dbt", "tool", "data build tool")
_add("Databricks", "tool")
_add("Airflow", "tool", "apache airflow")
_add("Spark", "tool", "apache spark", "pyspark")
_add("Kafka", "tool", "apache kafka")
_add("BigQuery", "tool", "google bigquery")
_add("Redshift", "tool", "amazon redshift")
_add("Postgres", "tool", "postgresql")
_add("MySQL", "tool")
_add("SQL Server", "tool", "mssql", "microsoft sql server")
_add("Oracle", "tool")
_add("ETL", "skill", "elt", "extract transform load", "data pipelines",
     "data pipeline", "ingestion pipelines")
_add("Data Warehousing", "skill", "data warehouse", "dimensional modeling",
     "star schema", "kimball")
_add("Data Modeling", "skill", "schema design", "logical data model")
_add("Data Quality", "skill", "data validation", "data integrity",
     "data reconciliation", "data cleansing")
_add("Data Governance", "skill", "master data management", "mdm", "data stewardship")

# --- BI / visualization -------------------------------------------------------
_add("Tableau", "tool")
_add("Power BI", "tool", "powerbi", "microsoft power bi")
_add("MicroStrategy", "tool", "micro strategy")
_add("Looker", "tool", "looker studio")
_add("Excel", "tool", "microsoft excel", "advanced excel", "pivot tables")
_add("SSAS", "tool", "analysis services", "excel/ssas", "olap cubes", "cubes")
_add("Qlik", "tool", "qlikview", "qlik sense")
_add("Dashboards", "skill", "dashboard development", "dashboarding",
     "reporting dashboards", "visualizations", "data visualization")

# --- cloud / infra / engineering practice ------------------------------------
_add("AWS", "tool", "amazon web services", "ec2", "s3", "lambda")
_add("Azure", "tool", "microsoft azure")
_add("GCP", "tool", "google cloud", "google cloud platform")
_add("Docker", "tool", "containers", "containerization")
_add("Kubernetes", "tool", "k8s")
_add("CI/CD", "skill", "continuous integration", "continuous deployment",
     "github actions", "jenkins", "build pipeline")
_add("Git", "tool", "github", "gitlab", "version control", "bitbucket")
_add("REST API", "skill", "rest apis", "restful", "api development",
     "api integration", "web services")
_add("Microservices", "skill", "service oriented architecture")
_add("Unit Testing", "skill", "unit tests", "test coverage", "automated testing",
     "test automation")
_add("Observability", "skill", "monitoring", "logging", "tracing", "telemetry")

# --- process / delivery / soft ------------------------------------------------
_add("Agile", "soft", "scrum", "kanban", "sprint", "agile methodology")
_add("Requirements Gathering", "soft", "requirements elicitation",
     "business requirements", "brd", "requirements analysis", "user stories")
_add("Technical Specification", "soft", "technical specs", "functional specification",
     "solution design", "design documentation")
_add("UAT", "soft", "user acceptance testing", "acceptance testing")
_add("Test Planning", "soft", "test strategy", "test cases", "test scripts", "qa")
_add("Stakeholder Management", "soft", "stakeholder engagement", "stakeholder partnering",
     "stakeholders", "stakeholder", "stakeholder meetings", "business stakeholders",
     "cross-functional collaboration", "cross functional", "business partnering")
_add("Project Management", "soft", "program management", "project delivery",
     "project lifecycle")
_add("Documentation", "soft", "technical writing", "runbooks", "knowledge transfer")
_add("Communication", "soft", "presentation skills", "written communication",
     "verbal communication", "storytelling")
_add("Problem Solving", "soft", "problem-solving", "analytical thinking", "critical thinking",
     "troubleshooting", "root cause analysis")
_add("Mentoring", "soft", "coaching", "knowledge sharing", "peer review")

# --- domain -------------------------------------------------------------------
_add("Retail Analytics", "skill", "retail reporting", "channel reporting",
     "store analytics", "merchandising analytics")
_add("Category Management", "skill", "category analytics", "assortment planning",
     "assortment", "sku rationalization", "planogram", "shelf analysis")
_add("Market Share", "skill", "share analysis", "share of market")
_add("Pricing Analysis", "skill", "price leadership", "price elasticity", "pricing strategy")
_add("Nielsen", "skill", "nielseniq", "syndicated data", "iri", "circana", "panel data")
_add("CPG", "skill", "consumer packaged goods", "fmcg")
_add("Forecasting", "skill", "demand planning", "demand forecasting", "time series")
_add("A/B Testing", "skill", "ab testing", "experimentation", "split testing")
_add("KPI Reporting", "skill", "kpis", "key performance indicators", "metrics definition",
     "scorecards")

_add("Microsoft Fabric", "tool", "ms fabric", "fabric lakehouse", "onelake")
_add("Big Data", "skill", "big-data", "large-scale data")
_add("Data Analysis", "skill", "data analytics", "analyzing data")
_add("Ad Hoc Analysis", "skill", "ad-hoc analysis", "ad hoc reporting", "ad-hoc reporting")
_add("Advanced Analytics", "skill")
_add("Insurance", "skill", "p&c", "property and casualty", "insurance industry",
     "claims data", "underwriting")
_add("Data Storytelling", "skill", "communicating insights", "insight communication",
     "non-technical audiences", "non-technical stakeholders", "without technical skills")
_add("Business Influence", "soft", "influence strategy", "influencing stakeholders",
     "influences strategy")

# --- credentials --------------------------------------------------------------
_add("Bachelor's Degree", "credential", "bachelors", "ba", "bs", "b.s.", "b.a.",
     "undergraduate degree", "four-year degree")
_add("Master's Degree", "credential", "masters", "ms", "m.s.", "msc",
     "graduate degree", "advanced degree")
_add("Computer Science", "credential", "cs degree", "computer engineering")
_add("Statistics", "credential", "applied statistics", "quantitative field")

# Terms a posting uses that mean "we will read your title", handled separately
TITLE_HINTS = (
    "engineer", "analyst", "scientist", "developer", "architect", "manager",
    "consultant", "specialist", "lead", "director",
)

# Section headers that tell us how badly a term is wanted.
REQUIRED_MARKERS = (
    "requirement", "required", "must have", "must-have", "minimum qualification",
    "basic qualification", "what you need", "what you'll need", "you have",
    "qualifications", "who you are", "essential", "skills",
)
# Sections whose vocabulary is not about the job itself.
SKIP_MARKERS = (
    "benefit", "compensation", "salary", "pay range", "perks",
    "equal opportunity", "eeo", "accommodation",
)
PREFERRED_MARKERS = (
    "preferred", "nice to have", "nice-to-have", "bonus", "plus", "desired",
    "a plus", "ideally", "additional qualification",
)

_WORD = re.compile(r"[a-z0-9+#./&'-]+")


def _pattern(term: str) -> re.Pattern[str]:
    """Word-boundary-ish matcher tolerant of the punctuation in tech names.

    A plain \\b fails on 'C++' and 'CI/CD', so we bound on "not a term
    character" instead of on "not a word character".
    """
    escaped = re.escape(term)
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def _compile(term: str, meta: dict) -> list[tuple[str, re.Pattern[str]]]:
    return [(term, _pattern(term))] + [(a, _pattern(a)) for a in meta["aliases"]]


_COMPILED: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    term: _compile(term, meta) for term, meta in LEXICON.items()
}


# --- domain packs and learned terms --------------------------------------------
#
# The built-in list above is tech, data and BI. Everyone else's vocabulary --
# nursing, accounting, marketing -- lives in packs under ats/lexicons/, loaded
# per person (the fact base's `lexicon_packs`). Terms the model discovers on a
# real posting are saved to a per-person lexicon.yaml and loaded the same way,
# so the exact-match half of extraction gets better with use.
#
# Loading never overrides a term already present: the curated core wins.

PACKS_DIR = Path(__file__).with_name("lexicons")
_CORE = {term: {"category": m["category"], "aliases": list(m["aliases"])}
         for term, m in LEXICON.items()}
_CATEGORIES = {"skill", "tool", "credential", "title", "soft", "other"}


def _register(term: str, category: str, aliases: list[str]) -> bool:
    term = str(term).strip()
    if not term or term in LEXICON:
        return False
    if any(normalize(term) == normalize(t) for t in LEXICON):
        return False
    meta = {"category": category if category in _CATEGORIES else "skill",
            "aliases": [str(a).strip() for a in aliases if str(a).strip()]}
    LEXICON[term] = meta
    _COMPILED[term] = _compile(term, meta)
    return True


def load_terms(entries: list[dict]) -> int:
    """Register {term, category, aliases} entries; returns how many were new."""
    return sum(_register(e.get("term", ""), str(e.get("category", "skill")),
                         list(e.get("aliases") or []))
               for e in entries if isinstance(e, dict))


def _read_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def available_packs() -> dict[str, str]:
    """Pack name -> one-line description."""
    return {p.stem: str(_read_yaml(p).get("description", ""))
            for p in sorted(PACKS_DIR.glob("*.yaml"))}


def load_pack(name: str) -> int:
    path = PACKS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown lexicon pack '{name}' "
                         f"(available: {', '.join(available_packs())})")
    return load_terms(_read_yaml(path).get("terms") or [])


def load_learned(path: Path) -> int:
    if not path.exists():
        return 0
    return load_terms(_read_yaml(path).get("terms") or [])


def remember(path: Path, keywords) -> int:
    """Save terms the model found that no list knew about, for next time.

    Soft skills are skipped: they are scored at a quarter weight and never
    steer anything, so learning them only adds noise.
    """
    import yaml

    new = [{"term": k.term, "category": k.category, "aliases": list(k.aliases)}
           for k in keywords if k.term not in LEXICON and k.category != "soft"]
    if not new:
        return 0
    existing = _read_yaml(path).get("terms", []) if path.exists() else []
    known = {normalize(e.get("term", "")) for e in existing}
    added = [e for e in new if normalize(e["term"]) not in known]
    if not added:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Terms the model found in postings that no lexicon knew about.\n"
                 "# Loaded on every run so they are matched exactly. Edit freely.\n")
        yaml.safe_dump({"terms": existing + added}, fh, sort_keys=False,
                       allow_unicode=True)
    return len(added)


def suggest_packs(text: str, min_hits: int = 3) -> list[str]:
    """Packs whose vocabulary shows up in `text` -- used when importing a resume."""
    out = []
    for path in sorted(PACKS_DIR.glob("*.yaml")):
        hits = 0
        for entry in _read_yaml(path).get("terms") or []:
            forms = [entry.get("term", ""), *(entry.get("aliases") or [])]
            if any(f and _pattern(f).search(text) for f in forms):
                hits += 1
        if hits >= min_hits:
            out.append(path.stem)
    return out


def reset() -> None:
    """Back to the built-in list only (tests, and a second run in one process)."""
    LEXICON.clear()
    _COMPILED.clear()
    for term, meta in _CORE.items():
        LEXICON[term] = {"category": meta["category"], "aliases": list(meta["aliases"])}
        _COMPILED[term] = _compile(term, LEXICON[term])


def find_terms(text: str) -> dict[str, dict]:
    """Return every lexicon term present in `text`, with counts and evidence.

    The evidence string is the first line the term appeared on. It is what makes
    the extraction auditable -- you can always ask why a keyword is in the list.
    """
    hits: dict[str, dict] = {}
    lines = text.splitlines()
    for term, patterns in _COMPILED.items():
        count = 0
        evidence = ""
        matched_aliases: list[str] = []
        for surface, pat in patterns:
            found = pat.findall(text)
            if found:
                count += len(found)
                if surface.lower() != term.lower():
                    matched_aliases.append(surface)
                if not evidence:
                    for line in lines:
                        if pat.search(line):
                            evidence = line.strip()[:220]
                            break
        if count:
            hits[term] = {
                "category": LEXICON[term]["category"],
                "frequency": count,
                "evidence": evidence,
                "aliases": LEXICON[term]["aliases"],
                "matched_aliases": matched_aliases,
            }
    return hits


def normalize(term: str) -> str:
    return " ".join(_WORD.findall(term.lower()))


def surface_forms(term: str) -> list[str]:
    """All the ways a term can legitimately appear, canonical first."""
    meta = LEXICON.get(term)
    if not meta:
        return [term]
    return [term, *meta["aliases"]]
