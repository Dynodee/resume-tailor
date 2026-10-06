"""Where the master resume comes from.

Two backends. The YAML fact base is canonical -- it is structured, it carries
the per-bullet keyword hints and the do-not-claim list, and it is what the
tailor node actually reasons over. The Google Drive backend exists so the fact
base can be refreshed from the document you keep editing by hand: it pulls the
doc, parses it back into the same structure, and merges it in.

Merging rather than replacing is deliberate. A parsed document cannot recover
the do_not_claim list or the keyword hints, so a naive overwrite would silently
throw away the parts that keep the tailoring honest.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import yaml

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DRIVE_ID = re.compile(r"(?:/d/|/file/d/|[?&]id=)([A-Za-z0-9_-]{20,})")


def load(ref: str | None, master_path: Path) -> dict[str, Any]:
    """Load the master fact base.

    `ref` may be None (use the YAML as-is), a path to another YAML file, or a
    Google Drive file id / URL to refresh from.
    """
    base = _load_yaml(master_path)
    if not ref:
        return base
    if _drive_id(ref):
        parsed = from_drive(ref)
        return merge(base, parsed)
    path = Path(ref)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    if path.exists():
        return merge(base, parse_resume_text(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"cannot resolve resume reference: {ref}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save(master: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(master, fh, sort_keys=False, allow_unicode=True, width=100)


def _drive_id(ref: str) -> str | None:
    if m := DRIVE_ID.search(ref):
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{25,}", ref):
        return ref
    return None


# --- Google Drive -------------------------------------------------------------

def from_drive(ref: str, token_path: str = "token.json",
               creds_path: str = "credentials.json") -> dict[str, Any]:
    """Pull a resume out of Drive and parse it into the fact-base shape.

    Requires a Google OAuth desktop client. See the README; the first run opens
    a browser once and caches the token.
    """
    file_id = _drive_id(ref)
    if not file_id:
        raise ValueError(f"not a Drive file id or URL: {ref}")
    text = drive_text(file_id, token_path, creds_path)
    return parse_resume_text(text)


def drive_text(file_id: str, token_path: str = "token.json",
               creds_path: str = "credentials.json") -> str:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Google Drive support needs the optional extras: "
            "pip install 'resume-tailor[drive]'"
        ) from exc

    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_path).write_text(creds.to_json(), encoding="utf-8")

    service = build("drive", "v3", credentials=creds)
    meta = service.files().get(fileId=file_id, fields="name,mimeType").execute()
    mime = meta.get("mimeType", "")

    buf = io.BytesIO()
    if mime == "application/vnd.google-apps.document":
        req = service.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        req = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    raw = buf.getvalue()

    if mime.endswith("wordprocessingml.document"):
        return _docx_text(raw)
    if mime == "application/pdf":
        return _pdf_text(raw)
    return raw.decode("utf-8", errors="replace")


def _docx_text(raw: bytes) -> str:
    import zipfile
    from xml.etree import ElementTree

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        xml = z.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    lines = []
    for para in root.iter(f"{ns}p"):
        text = "".join(t.text or "" for t in para.iter(f"{ns}t"))
        lines.append(text)
    return "\n".join(lines)


def _pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("reading a PDF resume needs: pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# --- parsing a rendered resume back into facts --------------------------------

HEAD = re.compile(
    r"^\s*\**\s*(professional summary|summary|profile|work experience|experience|"
    r"employment|ai & analytics projects|projects|education|skills|"
    r"technical skills|certifications)\s*\**\s*:?\s*$",
    re.IGNORECASE,
)
BULLET = re.compile(r"^\s*(?:[-•*●▪]|\d+\.)\s+(.*)$")
DATES = re.compile(
    r"(?P<start>(?:[A-Z][a-z]+\s+)?\d{4})\s*[-–—to]+\s*"
    r"(?P<end>present|current|(?:[A-Z][a-z]+\s+)?\d{4})",
    re.IGNORECASE,
)
ROLE_SEP = re.compile(r"\s+[-–—]{1,2}\s+")


def parse_resume_text(text: str) -> dict[str, Any]:
    """Best-effort structural parse of a rendered resume.

    This is intentionally forgiving: it is a refresh mechanism for the YAML, not
    an authority. Anything it cannot classify lands in `unparsed` so nothing is
    silently lost.
    """
    text = text.replace("\r\n", "\n")
    blocks: dict[str, list[str]] = {}
    current = "header"
    for line in text.split("\n"):
        stripped = line.strip().strip("*").strip()
        if not stripped:
            continue
        if m := HEAD.match(line.strip()):
            current = m.group(1).lower()
            blocks.setdefault(current, [])
            continue
        blocks.setdefault(current, []).append(stripped)

    out: dict[str, Any] = {"unparsed": {}}

    header = blocks.get("header", [])
    if header:
        contact: dict[str, str] = {"name": header[0]}
        joined = " ".join(header)
        if m := re.search(r"[\w.+_-]+@[\w-]+\.[\w.]+", joined):
            contact["email"] = m.group(0).replace("\\", "")
        if m := re.search(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", joined):
            contact["phone"] = m.group(0)
        if m := re.search(r"linkedin\.com/in/[\w-]+", joined, re.IGNORECASE):
            contact["linkedin"] = m.group(0)
        out["contact"] = contact
        tail = [h for h in header[1:] if "|" in h and "@" not in h]
        if tail:
            out["headlines"] = [{"id": "parsed", "text": tail[0], "fits": []}]

    for key in ("professional summary", "summary", "profile"):
        if key in blocks:
            out["summary_facts"] = [s for s in blocks[key] if len(s) > 20]
            break

    for key in ("work experience", "experience", "employment"):
        if key in blocks:
            out["experience"] = _parse_entries(blocks[key], prefix="role")
            break

    for key in ("ai & analytics projects", "projects"):
        if key in blocks:
            out["projects"] = _parse_entries(blocks[key], prefix="proj")
            break

    if "education" in blocks:
        edu = []
        lines = blocks["education"]
        i = 0
        while i < len(lines):
            line = lines[i]
            dates = ""
            if i + 1 < len(lines) and DATES.search(lines[i + 1]):
                dates = lines[i + 1]
                i += 1
            parts = ROLE_SEP.split(line, maxsplit=1)
            edu.append({
                "degree": parts[0].strip(),
                "school": parts[1].strip() if len(parts) > 1 else "",
                "dates": dates.strip(),
            })
            i += 1
        out["education"] = edu

    for key in ("skills", "technical skills"):
        if key in blocks:
            groups: dict[str, Any] = {}
            for idx, line in enumerate(blocks[key]):
                label, _, rest = line.partition(":")
                if rest:
                    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
                    groups[slug or f"group_{idx}"] = {
                        "label": label.strip().strip("*"),
                        "items": [s.strip() for s in rest.split(",") if s.strip()],
                    }
            if groups:
                out["skills"] = groups
            break

    for key, lines in blocks.items():
        if key not in {"header", "professional summary", "summary", "profile",
                       "work experience", "experience", "employment", "projects",
                       "ai & analytics projects", "education", "skills",
                       "technical skills"}:
            out["unparsed"][key] = lines

    return out


def _parse_entries(lines: list[str], prefix: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    counter = 0
    for line in lines:
        if m := BULLET.match(line):
            if current is not None:
                bid = f"{current['id']}_b{len(current['bullets'])}"
                current["bullets"].append({
                    "id": bid,
                    "text": m.group(1).strip().strip("*").strip(),
                    "keywords": [],
                })
            continue
        if d := DATES.search(line):
            if current is not None:
                current["start"] = d.group("start")
                current["end"] = d.group("end")
            continue
        counter += 1
        parts = ROLE_SEP.split(line, maxsplit=1)
        current = {
            "id": f"{prefix}{counter}",
            "title" if prefix == "role" else "name": parts[0].strip().strip("*"),
            "company" if prefix == "role" else "subtitle":
                parts[1].strip().strip("*") if len(parts) > 1 else "",
            "start": "", "end": "", "location": "", "tags": [], "bullets": [],
        }
        entries.append(current)
    return entries


def merge(base: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    """Overlay parsed document content onto the curated YAML.

    Curated metadata (do_not_claim, per-bullet keyword hints, headline `fits`)
    always wins, because a rendered document cannot express it.
    """
    out = dict(base)
    for key in ("contact", "summary_facts", "education"):
        if parsed.get(key):
            out[key] = parsed[key]

    for key, id_field in (("experience", "id"), ("projects", "id")):
        parsed_items = parsed.get(key) or []
        if not parsed_items:
            continue
        by_label = {_label(i): i for i in (base.get(key) or [])}
        merged = []
        for item in parsed_items:
            existing = by_label.get(_label(item))
            if existing:
                item = {**existing, **{k: v for k, v in item.items() if v}}
                item["bullets"] = _merge_bullets(existing.get("bullets", []),
                                                 item.get("bullets", []))
                item[id_field] = existing[id_field]
            merged.append(item)
        out[key] = merged

    if parsed.get("skills"):
        out["skills"] = {**(base.get("skills") or {}), **parsed["skills"]}
    out["do_not_claim"] = base.get("do_not_claim", [])
    return out


def _label(item: dict[str, Any]) -> str:
    raw = item.get("title") or item.get("name") or ""
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def _merge_bullets(existing: list[dict], parsed: list[dict]) -> list[dict]:
    """Keep the parsed text, but carry over keyword hints by fuzzy text match."""
    hints = {_label({"name": b.get("text", "")[:40]}): b.get("keywords", [])
             for b in existing}
    out = []
    for b in parsed:
        key = _label({"name": b.get("text", "")[:40]})
        out.append({**b, "keywords": b.get("keywords") or hints.get(key, [])})
    return out
