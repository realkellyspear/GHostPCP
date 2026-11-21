# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""GHostPCP — Opt-In Web Search & SSRF-Safe URL Reader.

Disabled by default so GHostPCP stays strictly offline unless the user
explicitly enables web access.
"""
from __future__ import annotations

import html as html_lib
from html.parser import HTMLParser
import ipaddress
import re
import socket
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

USER_AGENT = "GHostPCP/1.3 (+https://github.com/realkellyspear/GHostPCP)"
MAX_FETCH_BYTES = 512 * 1024
MAX_REDIRECTS = 4

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)

_CASUAL_NO_SEARCH_RE = re.compile(
    r"^(?:hi|hey|hello|yo|sup|good morning|good night|good evening|thanks|thank you|ok|okay|lol|haha|yes|no|yeah|nah|bye|love you|miss you)[!.\s?]*$",
    re.IGNORECASE,
)

_WEB_COMMAND_RE = re.compile(r"^(?:/web|/search|search:|web:)\s+(.+)$", re.IGNORECASE | re.DOTALL)


class UnsafeURLError(ValueError):
    """Raised when a URL targets a disallowed scheme, loopback, or private network address."""


class _ReadableHTMLParser(HTMLParser):
    """Extract clean article/page prose and page title from raw HTML."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "aside", "iframe"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return
        if t == "title":
            self.in_title = True
        elif t in self._BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth > 0:
            return
        if t == "title":
            self.in_title = False
        elif t in self._BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        clean = re.sub(r"\s+", " ", data)
        if not clean.strip():
            return
        if self.in_title:
            self.title_parts.append(clean.strip())
        else:
            self.text_parts.append(clean)

    def result(self) -> Tuple[str, str]:
        title = " ".join(self.title_parts).strip()
        raw = "".join(self.text_parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        non_empty = [ln for ln in lines if ln]
        return title, "\n".join(non_empty)


def html_to_clean_text(raw_html: str) -> Tuple[str, str]:
    parser = _ReadableHTMLParser()
    parser.feed(raw_html)
    parser.close()
    return parser.result()


def _is_disallowed_ip(ip_str: str) -> bool:
    ip_obj = ipaddress.ip_address(ip_str)
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def validate_public_url(url: str) -> str:
    """Validate that a URL is public HTTP(S) and does not resolve to local/private networks."""
    parsed = urllib.parse.urlsplit(str(url or "").strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError("Only http:// and https:// URLs are allowed.")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise UnsafeURLError("URL is missing a hostname.")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise UnsafeURLError(f"Local hostname '{host}' is blocked.")

    try:
        ipaddress.ip_address(host)
        if _is_disallowed_ip(host):
            raise UnsafeURLError(f"Private or loopback IP '{host}' is blocked.")
        return parsed.geturl()
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"Could not resolve hostname '{host}': {exc}") from exc

    if not addr_infos:
        raise ValueError(f"Hostname '{host}' returned no addresses.")
    for info in addr_infos:
        resolved_ip = info[4][0]
        if _is_disallowed_ip(resolved_ip):
            raise UnsafeURLError(f"Hostname '{host}' resolves to a private or loopback IP ({resolved_ip}).")
    return parsed.geturl()


def _safe_http_get(
    url: str,
    timeout: float = 8.0,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    current_url = validate_public_url(url)
    req_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for _ in range(MAX_REDIRECTS + 1):
        resp = requests.get(
            current_url,
            headers=req_headers,
            params=params,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
        params = None  # params only belong on the initial URL
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            resp.close()
            if not location:
                raise ValueError("Redirect missing Location header.")
            current_url = validate_public_url(urllib.parse.urljoin(current_url, location))
            continue
        return resp
    raise ValueError("Too many redirects while fetching URL.")


def fetch_url_content(url: str, max_chars: int = 2400, timeout: float = 8.0) -> Dict[str, str]:
    """Fetch and extract readable prose from a public web page or plain-text URL."""
    resp = _safe_http_get(url, timeout=timeout)
    try:
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        chunks: List[bytes] = []
        total = 0
        for part in resp.iter_content(chunk_size=16384):
            if not part:
                continue
            chunks.append(part)
            total += len(part)
            if total >= MAX_FETCH_BYTES:
                break
        raw_bytes = b"".join(chunks)
        encoding = resp.encoding or "utf-8"
        text_raw = raw_bytes.decode(encoding, errors="replace")
        final_url = resp.url or url
    finally:
        resp.close()

    if "html" in content_type or "<html" in text_raw[:500].lower():
        title, text = html_to_clean_text(text_raw)
    else:
        title, text = "", text_raw.strip()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[... page truncated ...]"
    return {
        "url": final_url,
        "title": title or urllib.parse.urlsplit(final_url).netloc or final_url,
        "text": text,
    }


def extract_urls_from_text(text: str, limit: int = 2) -> List[str]:
    found: List[str] = []
    for raw in _URL_RE.findall(str(text or "")):
        cleaned = raw.rstrip(".,;:!?'\"")
        if cleaned not in found:
            found.append(cleaned)
        if len(found) >= limit:
            break
    return found


def derive_heuristic_web_query(user_text: str) -> str:
    """Turn a user prompt into a concise web query, or '' when no web lookup is needed."""
    cleaned = re.sub(
        r"\[Attached File:\s*([^\]]+)\][\s\S]*?(?:\[/Attached File\]|\Z)",
        "",
        str(user_text or ""),
    )
    cleaned = _URL_RE.sub("", cleaned).strip()
    if not cleaned or _CASUAL_NO_SEARCH_RE.match(cleaned):
        return ""

    explicit = _WEB_COMMAND_RE.match(cleaned)
    if explicit:
        return explicit.group(1).strip()[:160]

    # Strip conversational filler prefixes so private names/greetings aren't sent to search engines
    trimmed = re.sub(
        r"^(?:hey|hi|hello|yo|please|can you|could you|would you|tell me|do you know|look up|search for|google|find out)\b[\s,:-]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    query = trimmed or cleaned
    if len(query) > 160:
        query = query[:160].rsplit(" ", 1)[0].strip() or query[:160]
    return query


def _parse_ddg_lite_results(raw_html: str, limit: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo Lite/HTML search results cleanly without external dependencies."""
    if "anomaly.js" in raw_html or "cc=botnet" in raw_html:
        raise RuntimeError("Search engine requested anti-bot verification.")
    results: List[Dict[str, str]] = []

    # DuckDuckGo HTML endpoint (<a class="result__a" href="...">title</a> ... <a class="result__snippet">...</a>)
    blocks = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>[\s\S]*?(?:class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>)?',
        raw_html,
        flags=re.IGNORECASE,
    )
    for href, title_html, snip_html in blocks:
        url = html_lib.unescape(href)
        if "duckduckgo.com/l/?" in url:
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            if qs.get("uddg"):
                url = qs["uddg"][0]
        if not url.startswith(("http://", "https://")):
            continue
        title = html_lib.unescape(re.sub(r"<[^>]+>", "", title_html)).strip()
        snippet = html_lib.unescape(re.sub(r"<[^>]+>", "", snip_html or "")).strip()
        if title or snippet:
            results.append({"title": title or url, "url": url, "snippet": snippet})
        if len(results) >= limit:
            return results

    # Fallback DuckDuckGo Lite endpoint (<a rel="nofollow" href="..." class='result-link'>...</a>)
    lite_links = re.findall(
        r'<a[^>]*class=[\'"]result-link[\'"][^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',
        raw_html,
        flags=re.IGNORECASE,
    )
    lite_snips = re.findall(
        r'<td[^>]*class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for idx, (href, title_html) in enumerate(lite_links):
        url = html_lib.unescape(href)
        if not url.startswith(("http://", "https://")):
            continue
        title = html_lib.unescape(re.sub(r"<[^>]+>", "", title_html)).strip()
        snippet = html_lib.unescape(re.sub(r"<[^>]+>", "", lite_snips[idx])).strip() if idx < len(lite_snips) else ""
        results.append({"title": title or url, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _search_wikipedia(query: str, limit: int = 3, timeout: float = 6.0) -> List[Dict[str, str]]:
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max(1, min(limit, 5)),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    hits = (data.get("query") or {}).get("search") or []
    out: List[Dict[str, str]] = []
    for item in hits[:limit]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        snippet = html_lib.unescape(re.sub(r"<[^>]+>", "", str(item.get("snippet") or ""))).strip()
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        out.append({"title": f"{title} — Wikipedia", "url": page_url, "snippet": snippet})
    return out


def _search_ddg(query: str, limit: int = 4, timeout: float = 7.0) -> List[Dict[str, str]]:
    headers = {"User-Agent": USER_AGENT}
    for endpoint in ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"):
        resp = requests.post(endpoint, data={"q": query}, headers=headers, timeout=timeout)
        if resp.status_code in (202, 403, 429):
            continue
        resp.raise_for_status()
        parsed = _parse_ddg_lite_results(resp.text, limit=limit)
        if parsed:
            return parsed
    return []


def _search_searxng(query: str, searxng_url: str, limit: int = 4, timeout: float = 8.0) -> List[Dict[str, str]]:
    base = str(searxng_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("SearXNG URL is empty.")
    resp = requests.get(
        f"{base}/search",
        params={"q": query, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out: List[Dict[str, str]] = []
    for item in (data.get("results") or [])[:limit]:
        url = str(item.get("url") or "").strip()
        if url:
            out.append({
                "title": str(item.get("title") or url).strip(),
                "url": url,
                "snippet": str(item.get("content") or "").strip(),
            })
    return out


def _search_brave(query: str, api_key: str, limit: int = 4, timeout: float = 8.0) -> List[Dict[str, str]]:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("Brave Search API key is not configured.")
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max(1, min(limit, 10))},
        headers={"Accept": "application/json", "X-Subscription-Token": key, "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    results = ((data.get("web") or {}).get("results") or [])[:limit]
    return [
        {
            "title": str(r.get("title") or r.get("url") or "").strip(),
            "url": str(r.get("url") or "").strip(),
            "snippet": str(r.get("description") or "").strip(),
        }
        for r in results
        if r.get("url")
    ]


def _search_tavily(query: str, api_key: str, limit: int = 4, timeout: float = 10.0) -> List[Dict[str, str]]:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("Tavily API key is not configured.")
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": max(1, min(limit, 8))},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": str(r.get("title") or r.get("url") or "").strip(),
            "url": str(r.get("url") or "").strip(),
            "snippet": str(r.get("content") or "").strip(),
        }
        for r in (data.get("results") or [])[:limit]
        if r.get("url")
    ]


def search_web(
    query: str,
    provider: str = "auto",
    max_results: int = 4,
    searxng_url: str = "",
    api_key: str = "",
) -> List[Dict[str, str]]:
    """Run an opt-in web search using the configured provider."""
    clean_query = str(query or "").strip()
    if not clean_query:
        return []
    prov = str(provider or "auto").strip().lower()
    if prov == "searxng":
        return _search_searxng(clean_query, searxng_url=searxng_url, limit=max_results)
    if prov == "brave":
        return _search_brave(clean_query, api_key=api_key, limit=max_results)
    if prov == "tavily":
        return _search_tavily(clean_query, api_key=api_key, limit=max_results)

    # Free zero-key auto chain: DuckDuckGo -> Wikipedia fallback
    results = _search_ddg(clean_query, limit=max_results)
    if results:
        return results
    return _search_wikipedia(clean_query, limit=min(3, max_results))


def gather_web_context(
    user_text: str,
    provider: str = "auto",
    searxng_url: str = "",
    api_key: str = "",
    search_mode: str = "smart",
    query_rewriter: Optional[Callable[[str], str]] = None,
    max_results: int = 4,
) -> Dict[str, Any]:
    """Fetch any pasted URLs and optionally run a web search, returning formatted WEB RESULTS."""
    urls = extract_urls_from_text(user_text, limit=2)
    pages_read: List[Dict[str, str]] = []
    search_results: List[Dict[str, str]] = []
    sources: List[Dict[str, str]] = []
    warnings: List[str] = []

    for url in urls:
        try:
            page = fetch_url_content(url)
            if page.get("text"):
                pages_read.append(page)
                sources.append({"title": page["title"], "url": page["url"]})
        except Exception as exc:
            warnings.append(f"Could not read {url}: {exc}")

    # Only search if no URL satisfied the prompt, or if user explicitly asked/triggered search
    query = ""
    if not pages_read:
        explicit_cmd = _WEB_COMMAND_RE.match(str(user_text or "").strip())
        if explicit_cmd:
            query = explicit_cmd.group(1).strip()[:160]
        else:
            rewriter_vetoed = False
            if query_rewriter is not None:
                try:
                    rewritten = str(query_rewriter(user_text) or "").strip()
                    if rewritten.upper() == "NONE":
                        rewriter_vetoed = True
                    elif rewritten:
                        query = rewritten[:160]
                except Exception:
                    query = ""
            if not query and not rewriter_vetoed:
                query = derive_heuristic_web_query(user_text)
            if not query and search_mode == "always":
                query = re.sub(r"\s+", " ", str(user_text or "")).strip()[:160]

        if query:
            try:
                search_results = search_web(
                    query,
                    provider=provider,
                    max_results=max_results,
                    searxng_url=searxng_url,
                    api_key=api_key,
                )
                for item in search_results:
                    if item.get("url"):
                        sources.append({"title": item["title"], "url": item["url"]})
            except Exception as exc:
                warnings.append(f"Web search warning: {exc}")

    sections: List[str] = []
    for page in pages_read:
        sections.append(f"[Page: {page['title']} ({page['url']})]\n{page['text']}")
    if search_results:
        lines = []
        for idx, item in enumerate(search_results, start=1):
            snip = f" — {item['snippet']}" if item.get("snippet") else ""
            lines.append(f"{idx}. {item['title']} ({item['url']}){snip}")
        sections.append("[Web Search Results]\n" + "\n".join(lines))

    return {
        "query": query,
        "pages_read": pages_read,
        "search_results": search_results,
        "sources": sources,
        "warnings": warnings,
        "context_block": "\n\n---\n\n".join(sections).strip(),
    }
