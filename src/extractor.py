"""URL-based article extraction.

Fetches a news URL, strips boilerplate, and returns the readable article text
plus metadata (title, source domain, image, published date when available).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 TruthLens/1.0"
)

DEFAULT_TIMEOUT = 10

# Tags whose text is never article content.
_DROP_TAGS = (
    "script", "style", "noscript", "iframe", "form", "nav", "footer", "header",
    "aside", "svg", "canvas", "button", "figure",
)

# Ordered by how strongly they signal the real article body.
_CONTENT_SELECTORS = (
    "article",
    "main",
    '[itemprop="articleBody"]',
    ".article-body",
    ".article-content",
    ".story-body",
    ".post-content",
    ".entry-content",
    "#article-body",
    ".content-body",
)

_MIN_WORDS = 25


class ExtractionError(Exception):
    """Raised when an article body cannot be retrieved from a URL."""


@dataclass
class Article:
    url: str
    title: str = ""
    text: str = ""
    source: str = ""
    image: str = ""
    published: str = ""
    word_count: int = 0
    paragraphs: list[str] = field(default_factory=list)


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return re.sub(r"^www\.", "", netloc)


def _collect_blocks(soup: BeautifulSoup) -> list[str]:
    """Return paragraph-level text blocks, longest first."""
    blocks: list[str] = []
    for tag in soup.find_all(["p", "h2", "h3", "li"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text.split()) >= 6:
            blocks.append(text)
    return blocks


def _pick_title(soup: BeautifulSoup) -> str:
    for getter in (
        lambda: soup.find("meta", property="og:title"),
        lambda: soup.find("meta", attrs={"name": "twitter:title"}),
    ):
        tag = getter()
        if tag and tag.get("content"):
            return tag["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _pick_meta(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = (
            soup.find("meta", property=name)
            or soup.find("meta", attrs={"name": name})
            or soup.find("meta", attrs={"itemprop": name})
        )
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def extract_article(url: str, timeout: int = DEFAULT_TIMEOUT) -> Article:
    """Download `url` and return its cleaned article text.

    Raises ExtractionError for invalid URLs, network failures, and pages that
    yield too little readable text.
    """
    url = (url or "").strip()
    if not url:
        raise ExtractionError("No URL provided.")
    if not _is_http_url(url):
        raise ExtractionError("Invalid URL. Include http:// or https://.")

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise ExtractionError(f"Could not establish a secure connection: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise ExtractionError("The page took too long to respond.") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in (401, 403):
            reason = "the site blocked the automated request"
        elif status == 404:
            reason = "the page was not found"
        else:
            reason = f"the site returned HTTP {status}"
        raise ExtractionError(f"Could not fetch the page: {reason}.") from exc
    except requests.exceptions.RequestException as exc:
        raise ExtractionError(f"Could not fetch the page: {exc}") from exc

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    title = _pick_title(soup)
    blocks = _collect_blocks(soup)

    # Prefer a semantic container when it holds the bulk of the prose.
    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if not node:
            continue
        scoped = [
            re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
            for p in node.find_all(["p", "li"])
        ]
        scoped = [p for p in scoped if len(p.split()) >= 6]
        if len(scoped) > max(5, len(blocks) * 0.4):
            blocks = scoped
            break

    if title and blocks:
        blocks = [b for b in blocks if title.lower() not in b.lower()[:60]]

    text = "\n\n".join(dict.fromkeys(blocks)).strip()
    word_count = len(text.split())
    if word_count < _MIN_WORDS:
        raise ExtractionError(
            "Could not extract article text from this page "
            f"(only {word_count} words found). It may be paywalled or JavaScript-rendered."
        )

    return Article(
        url=response.url,
        title=title,
        text=text,
        source=_domain(response.url),
        image=_pick_meta(soup, "og:image", "twitter:image"),
        published=_pick_meta(
            soup, "article:published_time", "datePublished", "pubdate", "date"
        ),
        word_count=word_count,
        paragraphs=blocks,
    )