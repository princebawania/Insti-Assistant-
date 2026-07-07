"""
Ingestion: load raw source material into {"text", "source"} documents.

Supported out of the box:
  - .txt / .md  : read as UTF-8 text
  - .pdf        : extracted page-by-page with pdfplumber
  - .html/.htm  : visible text extracted with BeautifulSoup

Also includes a small `scrape_url` helper so you can pull an official IITB
web page straight into the corpus.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import config


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def load_html(path_or_html: str, is_file: bool = True) -> str:
    from bs4 import BeautifulSoup

    raw = Path(path_or_html).read_text(encoding="utf-8", errors="ignore") \
        if is_file else path_or_html
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".html", ".htm"}:
        return load_html(str(path), is_file=True)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_directory(raw_dir: Path = config.RAW_DIR) -> List[dict]:
    """Load every supported file in raw_dir into documents."""
    docs: List[dict] = []
    for path in sorted(Path(raw_dir).glob("**/*")):
        if not path.is_file():
            continue
        try:
            text = load_file(path)
        except ValueError:
            continue  # skip unsupported files silently
        if text.strip():
            docs.append({"text": text, "source": path.name})
    return docs


def scrape_url(url: str) -> dict:
    """Fetch a web page and return a document dict. Use for official IITB pages."""
    import requests

    resp = requests.get(url, timeout=30, headers={"User-Agent": "InstiAssist/1.0"})
    resp.raise_for_status()
    text = load_html(resp.text, is_file=False)
    return {"text": text, "source": url}
