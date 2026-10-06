"""Fetch a job posting from a URL and reduce it to clean text plus sections.

Most ATS vendors render the posting client-side, so scraping the HTML you get
back from a plain GET often yields an empty shell. Where a vendor exposes a
JSON endpoint we use it; the HTML path is the fallback, and JSON-LD JobPosting
markup is tried before falling back to tag-stripping.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import urlparse

import httpx

from ..state import JobPosting

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

# Words that mark a line as a section heading. Matching on these rather than on
# a fixed list of exact headings matters: Workday says "Essential
# Qualifications", Greenhouse boards say "What you'll bring", and a fixed list
# silently drops every term under a heading it has not seen into "mentioned".
HEAD_WORDS = (
    "about", "responsibilit", "duties", "what you", "what we", "who you",
    "you have", "you'll", "you will", "the role", "the opportunity",
    "your impact", "day to day", "requirement", "required", "qualification",
    "must have", "must-have", "essential", "skills", "experience",
    "preferred", "nice to have", "nice-to-have", "bonus", "desired",
    "benefit", "compensation", "salary", "pay range", "perks",
    "equal opportunity", "eeo", "job description", "overview", "education",
)
_HEAD_LINE = re.compile(r"^\s*[#*\-\u2022\s]*(?P<head>[^\n]{2,60}?)\s*:?\s*$", re.MULTILINE)


def _is_heading(line: str) -> bool:
    """Short, label-like line that names a section.

    Heuristic: at most six words, no sentence punctuation, and containing one of
    HEAD_WORDS. Bullets never qualify -- "- 5 years of SQL" is content even
    though it contains "years".
    """
    raw = line.strip()
    if not raw or raw.startswith(("-", "\u2022", "*")) and not raw.startswith("**"):
        return False
    text = raw.strip("#* ").rstrip(":").strip()
    if not text or len(text.split()) > 6 or text.endswith((".", ",", ";")):
        return False
    low = text.lower()
    return any(w in low for w in HEAD_WORDS)


class FetchError(RuntimeError):
    pass


def fetch(url: str, timeout: float = 30.0) -> JobPosting:
    host = (urlparse(url).netloc or "").lower()
    for matcher, handler in (
        ("greenhouse.io", _greenhouse),
        ("lever.co", _lever),
        ("ashbyhq.com", _ashby),
        ("myworkdayjobs.com", _workday),
    ):
        if matcher in host:
            try:
                posting = handler(url, timeout)
                if posting and len(posting.raw_text) > 300:
                    return _finish(posting)
            except Exception:
                # Vendor APIs move. Fall through to the generic path rather than
                # failing the run.
                pass
    return _finish(_generic(url, timeout))


def from_text(text: str, title: str = "", company: str = "") -> JobPosting:
    return _finish(JobPosting(source="paste", raw_text=text, title=title, company=company))


def _finish(posting: JobPosting) -> JobPosting:
    posting.raw_text = _clean(posting.raw_text)
    posting.sections = split_sections(posting.raw_text)
    if not posting.title:
        posting.title = _guess_title(posting.raw_text)
    return posting


# --- vendor-specific ---------------------------------------------------------

def _get(url: str, timeout: float) -> httpx.Response:
    r = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r


def _greenhouse(url: str, timeout: float) -> JobPosting | None:
    # .../{board}/jobs/{id} on either boards.greenhouse.io or job-boards.greenhouse.io
    m = re.search(r"greenhouse\.io/(?:embed/job_app\?for=)?([^/?#]+)", url)
    jid = re.search(r"(?:/jobs/|gh_jid=)(\d+)", url)
    if not (m and jid):
        return None
    board = m.group(1)
    api = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{jid.group(1)}"
    data = _get(api, timeout).json()
    return JobPosting(
        url=url, source="greenhouse",
        title=data.get("title", ""),
        company=(data.get("company_name") or board).replace("-", " ").title(),
        location=(data.get("location") or {}).get("name", ""),
        raw_text=_html_to_text(html.unescape(data.get("content", ""))),
    )


def _lever(url: str, timeout: float) -> JobPosting | None:
    m = re.search(r"lever\.co/([^/?#]+)/([0-9a-f-]{8,})", url)
    if not m:
        return None
    company, pid = m.groups()
    data = _get(f"https://api.lever.co/v0/postings/{company}/{pid}", timeout).json()
    cats = data.get("categories") or {}
    body = data.get("descriptionPlain") or _html_to_text(data.get("description", ""))
    extra = "\n".join(
        f"{s.get('text','')}\n" + _html_to_text(s.get("content", ""))
        for s in data.get("lists", [])
    )
    return JobPosting(
        url=url, source="lever", title=data.get("text", ""),
        company=company.replace("-", " ").title(),
        location=cats.get("location", ""),
        raw_text=f"{body}\n\n{extra}",
    )


def _ashby(url: str, timeout: float) -> JobPosting | None:
    r = _get(url, timeout)
    m = re.search(r'window\.__appData\s*=\s*({.*?});?\s*</script>', r.text, re.DOTALL)
    if not m:
        return None
    data = json.loads(m.group(1))
    posting = (data.get("posting") or {})
    return JobPosting(
        url=url, source="ashby", title=posting.get("title", ""),
        company=(data.get("organization") or {}).get("name", ""),
        location=posting.get("locationName", ""),
        raw_text=_html_to_text(posting.get("descriptionHtml", "")),
    )


def _workday(url: str, timeout: float) -> JobPosting | None:
    # Workday renders client-side but serves the same path as JSON.
    api = url.split("?")[0]
    r = httpx.get(api, headers={**HEADERS, "Accept": "application/json"},
                  timeout=timeout, follow_redirects=True)
    if "json" not in r.headers.get("content-type", ""):
        return None
    info = (r.json().get("jobPostingInfo") or {})
    return JobPosting(
        url=url, source="workday", title=info.get("title", ""),
        location=info.get("location", ""),
        raw_text=_html_to_text(info.get("jobDescription", "")),
    )


def _generic(url: str, timeout: float) -> JobPosting:
    r = _get(url, timeout)
    text = r.text

    # JSON-LD is the single most reliable generic source when it is present.
    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        text, re.DOTALL | re.IGNORECASE,
    ):
        try:
            payload = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for node in payload if isinstance(payload, list) else [payload]:
            if isinstance(node, dict) and "JobPosting" in str(node.get("@type", "")):
                org = node.get("hiringOrganization") or {}
                loc = node.get("jobLocation") or {}
                if isinstance(loc, list) and loc:
                    loc = loc[0]
                addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
                return JobPosting(
                    url=url, source="json-ld",
                    title=node.get("title", ""),
                    company=org.get("name", "") if isinstance(org, dict) else "",
                    location=", ".join(
                        str(addr.get(k)) for k in ("addressLocality", "addressRegion")
                        if addr.get(k)
                    ),
                    raw_text=_html_to_text(node.get("description", "")),
                )

    return JobPosting(url=url, source="generic", raw_text=_html_to_text(text))


# --- text handling -----------------------------------------------------------

def _html_to_text(markup: str) -> str:
    if not markup:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover
        return re.sub(r"<[^>]+>", " ", markup)

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    # Preserve list structure -- "requirements" lists are where the ATS terms live.
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for br in soup.find_all(["br", "p", "div", "h1", "h2", "h3", "h4"]):
        br.insert_before("\n")
    main = soup.find("main") or soup.find("article") or soup
    return main.get_text(" ")


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()).strip()


def split_sections(text: str) -> dict[str, str]:
    """Bucket the posting under its own headings.

    The bucket a term falls into is what decides required vs preferred later, so
    this is load-bearing rather than cosmetic.
    """
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if _is_heading(ln)]
    if not heads:
        return {"body": text}
    sections: dict[str, str] = {}
    if heads[0] > 0:
        sections["intro"] = "\n".join(lines[: heads[0]]).strip()
    for n, i in enumerate(heads):
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        key = lines[i].strip().strip("#*-\u2022 ").rstrip(":").strip().lower()
        body = "\n".join(lines[i + 1:end]).strip()
        if body:
            sections[key] = (sections.get(key, "") + "\n" + body).strip()
    return sections or {"body": text}


def _guess_title(text: str) -> str:
    for line in text.splitlines()[:8]:
        line = line.strip(" -•")
        if 3 < len(line) < 90 and any(
            h in line.lower() for h in
            ("engineer", "analyst", "scientist", "developer", "manager", "architect",
             "consultant", "specialist", "lead", "director", "associate")
        ):
            return line
    return ""
