# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""GHostPCP — Archive Extractor, Corporate Refusal Stripper, and Volume Writer.

Auto-detects and transforms OpenAI ChatGPT exports (.zip, conversations.json,
conversations-*.json), Google Gemini Takeout exports (MyActivity.json,
MyActivity.html, or Takeout .zip/folders), JSONL pairs, and Markdown/text/PDF
documents into clean, high-density Markdown volumes:
    ghost_pcp_01.md, ghost_pcp_02.md, ...
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html as html_lib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import zipfile

from persona import DEFAULT_BLOCKED_PHRASES, DEFAULT_PROTECTED_WORDS

INTERACTIONS_PER_VOLUME = 1000
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log"}

_IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "backups",
}

_NON_CONVERSATION_JSON_NAMES = {
    "user.json",
    "message_feedback.json",
    "model_comparisons.json",
    "shared_conversations.json",
    "settings.json",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "personas.json",
    "persona.json",
    "ghost_pcp_settings.json",
}

_NON_GEMINI_TAKEOUT_HEADERS = (
    "search",
    "youtube",
    "maps",
    "chrome",
    "google play",
    "play store",
    "ads",
    "shopping",
    "gmail",
    "drive",
    "photos",
    "news",
    "discover",
    "android",
)


class _HTMLToMarkdownParser(HTMLParser):
    """Zero-dependency HTML-to-Markdown converter for Gemini safeHtmlItem payloads."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.list_stack: List[Tuple[str, int]] = []
        self.in_pre = False
        self.pre_lang = ""
        self.in_code = False
        self.in_blockquote = False
        self.skip_depth = 0
        self.link_stack: List[Tuple[str, int]] = []
        self.in_table_row = False
        self.row_cells: List[str] = []
        self.in_cell = False
        self.cell_parts: List[str] = []

    def _emit(self, text: str) -> None:
        if self.skip_depth > 0:
            return
        if self.in_cell:
            self.cell_parts.append(text)
        else:
            self.parts.append(text)

    def _ensure_blank_line(self) -> None:
        if self.in_cell:
            return
        joined = "".join(self.parts)
        if not joined:
            return
        if joined.endswith("\n\n"):
            return
        if joined.endswith("\n"):
            self.parts.append("\n")
        else:
            self.parts.append("\n\n")

    def _ensure_newline(self) -> None:
        if self.in_cell:
            return
        joined = "".join(self.parts)
        if joined and not joined.endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        if t in ("script", "style"):
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return

        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._ensure_blank_line()
            level = int(t[1])
            self._emit("#" * level + " ")
        elif t == "p":
            if not self.list_stack:
                self._ensure_blank_line()
        elif t == "br":
            self._emit(" " if self.in_cell else "\n")
        elif t == "hr":
            self._ensure_blank_line()
            self._emit("---\n\n")
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "blockquote":
            self._ensure_blank_line()
            self.in_blockquote = True
            self._emit("> ")
        elif t == "pre":
            self._ensure_blank_line()
            self.in_pre = True
            cls = attr_map.get("class", "")
            m = re.search(r"language-([A-Za-z0-9_+-]+)", cls)
            self.pre_lang = m.group(1) if m else ""
            self._emit(f"```{self.pre_lang}\n")
        elif t == "code":
            if self.in_pre:
                cls = attr_map.get("class", "")
                if not self.pre_lang:
                    m = re.search(r"language-([A-Za-z0-9_+-]+)", cls)
                    if m and self.parts and self.parts[-1] == "```\n":
                        self.pre_lang = m.group(1)
                        self.parts[-1] = f"```{self.pre_lang}\n"
            else:
                self.in_code = True
                self._emit("`")
        elif t == "ul":
            if not self.list_stack:
                self._ensure_blank_line()
            else:
                self._ensure_newline()
            self.list_stack.append(("ul", 0))
        elif t == "ol":
            if not self.list_stack:
                self._ensure_blank_line()
            else:
                self._ensure_newline()
            self.list_stack.append(("ol", 0))
        elif t == "li":
            self._ensure_newline()
            indent = "  " * max(0, len(self.list_stack) - 1)
            if self.list_stack:
                kind, count = self.list_stack[-1]
                count += 1
                self.list_stack[-1] = (kind, count)
                bullet = f"{count}. " if kind == "ol" else "- "
            else:
                bullet = "- "
            self._emit(f"{indent}{bullet}")
        elif t == "a":
            href = attr_map.get("href", "").strip()
            self.link_stack.append((href, len(self.cell_parts if self.in_cell else self.parts)))
            self._emit("[")
        elif t == "tr":
            self.in_table_row = True
            self.row_cells = []
        elif t in ("th", "td"):
            self.in_cell = True
            self.cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style"):
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth > 0:
            return

        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._ensure_blank_line()
        elif t == "p":
            if self.list_stack:
                self._ensure_newline()
            else:
                self._ensure_blank_line()
        elif t in ("div", "section", "article"):
            if not self.in_pre and not self.list_stack:
                self._ensure_newline()
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "blockquote":
            self.in_blockquote = False
            self._ensure_blank_line()
        elif t == "pre":
            if self.in_pre:
                joined = "".join(self.parts)
                if not joined.endswith("\n"):
                    self._emit("\n")
                self._emit("```\n\n")
                self.in_pre = False
                self.pre_lang = ""
        elif t == "code":
            if self.in_code and not self.in_pre:
                self._emit("`")
                self.in_code = False
        elif t in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            if not self.list_stack:
                self._ensure_blank_line()
        elif t == "li":
            self._ensure_newline()
        elif t == "a":
            href, start_idx = self.link_stack.pop() if self.link_stack else ("", -1)
            target_buf = self.cell_parts if self.in_cell else self.parts
            if start_idx >= 0 and start_idx < len(target_buf):
                label = "".join(target_buf[start_idx + 1 :]).strip()
            else:
                label = ""
            if href and href.startswith(("http://", "https://")) and label and label != href:
                self._emit(f"]({href})")
            else:
                # Strip the opening '[' so plain URLs or empty links stay clean
                if start_idx >= 0 and start_idx < len(target_buf) and target_buf[start_idx] == "[":
                    target_buf[start_idx] = ""
        elif t in ("th", "td"):
            if self.in_cell:
                cell_text = re.sub(r"\s+", " ", "".join(self.cell_parts)).strip()
                self.row_cells.append(cell_text)
                self.in_cell = False
                self.cell_parts = []
        elif t == "tr":
            if self.in_table_row and self.row_cells:
                self._ensure_newline()
                self._emit("| " + " | ".join(self.row_cells) + " |\n")
            self.in_table_row = False
            self.row_cells = []
        elif t == "table":
            self._ensure_blank_line()

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        if self.in_pre or self.in_code:
            self._emit(data)
            return
        text = re.sub(r"[ \t\r\n]+", " ", data)
        if not text:
            return
        if text == " ":
            target = self.cell_parts if self.in_cell else self.parts
            if not target or target[-1].endswith((" ", "\n", "[", "**", "*")):
                return
        if self.in_blockquote and "\n" in text:
            text = text.replace("\n", "\n> ")
        self._emit(text)

    def get_markdown(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.rstrip() for ln in raw.splitlines()]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


def html_to_markdown(raw_html: str) -> str:
    """Convert safeHtmlItem HTML to Markdown (uses html2text if installed, else stdlib parser)."""
    raw = str(raw_html or "").strip()
    if not raw:
        return ""
    try:
        import html2text  # type: ignore

        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_images = True
        h.ignore_links = False
        h.skip_internal_links = True
        h.single_newline_on_list_items = True
        return h.handle(raw).strip()
    except ImportError:
        pass

    parser = _HTMLToMarkdownParser()
    parser.feed(raw)
    parser.close()
    md = parser.get_markdown()
    if md:
        return md
    # Fallback plain unescaped text if markup had only text nodes
    return html_lib.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def extract_text_from_document(filename: str, raw_bytes: bytes) -> str:
    """Extract clean readable text from .pdf, .txt, .md, .csv, or .log uploads."""
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf to be installed.") from exc
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages: List[str] = []
        for page in reader.pages:
            extracted = (page.extract_text() or "").strip()
            if extracted:
                pages.append(extracted)
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError(f"No readable text was found in PDF '{safe_name}' (scanned image PDFs need OCR first).")
        return text
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
        if not text:
            raise ValueError(f"File '{safe_name}' is empty.")
        return text
    raise ValueError(f"Unsupported document format '{suffix}'. Supported: .pdf, .txt, .md, .csv, .log")


def _extract_text_from_message(msg_obj: dict) -> Tuple[str, str]:
    """Return (role, text) from a ChatGPT/AI Studio mapping message or flat message dict."""
    if not isinstance(msg_obj, dict):
        return "", ""
    if isinstance(msg_obj.get("message"), dict):
        inner = msg_obj["message"]
    elif isinstance(msg_obj.get("response"), dict):
        inner = msg_obj["response"]
    else:
        inner = msg_obj
    if not isinstance(inner, dict):
        return "", ""

    if inner.get("isThought") is True:
        return "", ""

    author = inner.get("author")
    if isinstance(author, dict):
        role = str(author.get("role") or "")
    elif isinstance(inner.get("sender"), str):
        role = inner["sender"]
    elif isinstance(inner.get("src"), dict) and "is_human" in inner["src"]:
        role = "user" if inner["src"]["is_human"] else "assistant"
    elif "is_human" in inner:
        role = "user" if inner["is_human"] else "assistant"
    else:
        role = str(inner.get("role") or "")
    role = role.lower().strip()
    if role == "human":
        role = "user"
    elif role in ("grok", "claude", "character", "char"):
        role = "assistant"

    # Ignore hidden system or tool scaffolding messages
    metadata = inner.get("metadata")
    if isinstance(metadata, dict) and metadata.get("is_visually_hidden_from_conversation"):
        return "", ""

    content = inner.get("content")
    text_parts: List[str] = []
    if isinstance(content, dict):
        content_type = str(content.get("content_type") or "text")
        if content_type not in ("text", "multimodal_text"):
            return role, ""
        parts = content.get("parts") or []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    elif isinstance(content, list):
        # Anthropic Claude content block list: [{"type": "text", "text": "..."}]
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = str(block.get("type") or "text").lower()
                if btype in ("thinking", "tool_use", "tool_result"):
                    continue
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
    elif isinstance(content, str):
        text_parts.append(content)
    elif isinstance(inner.get("text"), str):
        text_parts.append(inner["text"])
    elif isinstance(inner.get("message"), str):
        text_parts.append(inner["message"])
    elif isinstance(inner.get("candidates"), list):
        # Character.AI candidates fallback
        for cand in inner["candidates"]:
            if isinstance(cand, dict) and isinstance(cand.get("raw_content"), str) and cand["raw_content"].strip():
                text_parts.append(cand["raw_content"])
                break

    text = "\n".join(p for p in text_parts if p).strip()
    return role, text


def _ordered_nodes_from_mapping(mapping: dict) -> List[dict]:
    """Traverse a ChatGPT conversation mapping in chronological order."""
    nodes = [v for v in mapping.values() if isinstance(v, dict)]
    roots = [
        n for n in nodes
        if not n.get("parent") or n.get("parent") not in mapping
    ]
    if not roots:
        return nodes

    ordered: List[dict] = []
    visited = set()
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        node_id = node.get("id")
        if node_id and node_id in visited:
            continue
        if node_id:
            visited.add(node_id)
        ordered.append(node)
        children = node.get("children") or []
        for child_id in reversed(children):
            child = mapping.get(child_id)
            if isinstance(child, dict):
                stack.append(child)
    return ordered


def _is_gemini_activity_entry(entry: dict) -> bool:
    """Return True when a JSON dict resembles a Google Takeout Gemini MyActivity entry."""
    if not isinstance(entry, dict):
        return False
    if isinstance(entry.get("safeHtmlItem"), list):
        return True
    title = entry.get("title")
    if isinstance(title, str) and title.startswith(("Prompted ", "Uploaded ")):
        return True
    header = entry.get("header")
    if isinstance(header, str) and any(tok in header.lower() for tok in ("gemini", "bard")):
        return True
    products = entry.get("products")
    if isinstance(products, list) and any(
        "gemini" in str(p).lower() or "bard" in str(p).lower() for p in products
    ):
        return True
    return False


def _is_non_gemini_activity_entry(entry: dict) -> bool:
    """Skip non-Gemini entries if a multi-service Google Takeout MyActivity.json is supplied."""
    title = str(entry.get("title") or "")
    header = str(entry.get("header") or "").strip().lower()
    products = entry.get("products")
    product_text = " ".join(str(p).lower() for p in products) if isinstance(products, list) else ""

    is_gemini_tagged = ("gemini" in header or "bard" in header or "gemini" in product_text or "bard" in product_text)
    if is_gemini_tagged:
        return False
    if header and any(bad in header for bad in _NON_GEMINI_TAKEOUT_HEADERS):
        return True
    if header and not title.startswith(("Prompted ", "Uploaded ")):
        return True
    if isinstance(products, list) and products and not title.startswith(("Prompted ", "Uploaded ")):
        return True
    return False


def _extract_gemini_entries_from_data(data: object) -> List[Tuple[str, str, str]]:
    """Extract (user_text, ai_markdown, timestamp) tuples from Gemini MyActivity.json data."""
    items = data if isinstance(data, list) else [data]
    results: List[Tuple[str, str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if _is_non_gemini_activity_entry(entry):
            continue

        title = str(entry.get("title") or "")
        if title.startswith("Uploaded "):
            continue

        user_text = ""
        if title.startswith("Prompted "):
            user_text = title[len("Prompted "):].strip()
        elif title.strip().rstrip(":") == "Prompted":
            user_text = ""

        ts = str(entry.get("time") or entry.get("date") or "").strip()

        ai_parts: List[str] = []
        safe_items = entry.get("safeHtmlItem")
        if isinstance(safe_items, list):
            for item in safe_items:
                if not isinstance(item, dict):
                    continue
                raw = item.get("html") or ""
                if raw:
                    converted = html_to_markdown(raw)
                    if converted:
                        ai_parts.append(converted)
        ai_text = "\n\n".join(ai_parts).strip()

        if not user_text and ai_text:
            attached = entry.get("attachedFiles")
            if isinstance(attached, list) and attached:
                names = [str(a.get("name") if isinstance(a, dict) else a) for a in attached if a]
                if names:
                    user_text = f"[Attached: {', '.join(names)}]"
            if not user_text and (title.strip().startswith("Prompted") or _is_gemini_activity_entry(entry)):
                user_text = "[Attachment / non-text prompt]"

        if user_text and ai_text:
            results.append((user_text, ai_text, ts))
    return results


def is_gemini_takeout_html(raw_html: str) -> bool:
    """Return True if raw_html looks like a Google Takeout Gemini MyActivity.html export."""
    head = raw_html[:50000]
    has_takeout_cells = ("content-cell" in head) or ("outer-cell" in head)
    has_gemini_cues = ("Gemini Apps" in raw_html or "Bard" in raw_html) and ("Prompted" in raw_html)
    return bool(has_takeout_cells and has_gemini_cues)


class _TakeoutHTMLCardsParser(HTMLParser):
    """Extract outer-cell/content-cell cards from Google Takeout MyActivity.html."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.cards: List[Tuple[str, str]] = []
        self._in_card = False
        self._card_depth = 0
        self._in_header = False
        self._header_depth = 0
        self._in_body = False
        self._body_depth = 0
        self._header_buf: List[str] = []
        self._body_buf: List[str] = []

    def _attr_class(self, attrs: List[Tuple[str, Optional[str]]]) -> str:
        for k, v in attrs:
            if k.lower() == "class" and v:
                return v
        return ""

    @staticmethod
    def _format_start(tag: str, attrs: List[Tuple[str, Optional[str]]]) -> str:
        if not attrs:
            return f"<{tag}>"
        parts = []
        for k, v in attrs:
            if v is None:
                parts.append(k)
            else:
                parts.append(f'{k}="{html_lib.escape(v, quote=True)}"')
        return f"<{tag} {' '.join(parts)}>"

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        cls = self._attr_class(attrs)
        classes = set(cls.split())

        if not self._in_card and "outer-cell" in classes:
            self._in_card = True
            self._card_depth = 1
            self._header_buf = []
            self._body_buf = []
            return

        if not self._in_card:
            return

        if t not in ("br", "hr", "img", "meta", "link", "input"):
            self._card_depth += 1

        if not self._in_header and "header-cell" in classes:
            self._in_header = True
            self._header_depth = 1
            return
        if self._in_header:
            if t not in ("br", "hr", "img", "meta", "link", "input"):
                self._header_depth += 1
            self._header_buf.append(self._format_start(t, attrs))
            return

        if not self._in_body and "content-cell" in classes and "mdl-typography--text-right" not in classes:
            self._in_body = True
            self._body_depth = 1
            return
        if self._in_body:
            if t not in ("br", "hr", "img", "meta", "link", "input"):
                self._body_depth += 1
            self._body_buf.append(self._format_start(t, attrs))

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if not self._in_card or t in ("br", "hr", "img", "meta", "link", "input"):
            return

        if self._in_header:
            self._header_depth -= 1
            if self._header_depth <= 0:
                self._in_header = False
            else:
                self._header_buf.append(f"</{t}>")
        elif self._in_body:
            self._body_depth -= 1
            if self._body_depth <= 0:
                self._in_body = False
            else:
                self._body_buf.append(f"</{t}>")

        self._card_depth -= 1
        if self._card_depth <= 0:
            self._in_card = False
            if self._body_buf:
                self.cards.append(("".join(self._header_buf), "".join(self._body_buf)))

    def handle_data(self, data: str) -> None:
        if self._in_header:
            self._header_buf.append(data)
        elif self._in_body:
            self._body_buf.append(data)

    def handle_entityref(self, name: str) -> None:
        text = f"&{name};"
        if self._in_header:
            self._header_buf.append(text)
        elif self._in_body:
            self._body_buf.append(text)

    def handle_charref(self, name: str) -> None:
        text = f"&#{name};"
        if self._in_header:
            self._header_buf.append(text)
        elif self._in_body:
            self._body_buf.append(text)


def extract_gemini_html_pairs(raw_html: str) -> List[Tuple[str, str]]:
    """Extract (user_text, ai_markdown) pairs from a Google Takeout MyActivity.html file."""
    if not is_gemini_takeout_html(raw_html):
        return []
    parser = _TakeoutHTMLCardsParser()
    parser.feed(raw_html)
    parser.close()

    pairs: List[Tuple[str, str]] = []
    ts_re = re.compile(
        r"^[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?(?:\s+[A-Z]{2,5})?$"
    )
    for header_html, body_html in parser.cards:
        header_plain = html_to_markdown(header_html).lower()
        if header_plain and not any(tok in header_plain for tok in ("gemini", "bard")):
            continue
        md = html_to_markdown(body_html)
        if not md:
            continue
        lines = [ln.strip() for ln in md.splitlines()]
        # Strip trailing Takeout metadata lines if present
        while lines and (not lines[-1] or lines[-1].startswith(("Products:", "Why is this here?"))):
            lines.pop()
        if not lines:
            continue
        first = lines[0]
        if first.startswith("Uploaded "):
            continue
        if not first.startswith("Prompted"):
            continue
        user_text = first[len("Prompted"):].lstrip(":").strip()
        rest = lines[1:]
        while rest and (not rest[0] or ts_re.match(rest[0])):
            rest.pop(0)
        ai_text = "\n".join(rest).strip()
        if not user_text and ai_text:
            user_text = "[Attachment / non-text prompt]"
        if user_text and ai_text:
            pairs.append((user_text, ai_text))
    return pairs


def detect_json_source(data: object) -> str:
    """Detect archive source: 'google', 'openai', 'claude', 'grok', 'character_ai', or 'unknown'."""
    if isinstance(data, dict):
        if isinstance(data.get("chunkedPrompt"), dict):
            return "google"
        if isinstance(data.get("histories"), list):
            return "character_ai"
        if isinstance(data.get("conversations"), list):
            inner_conv = data["conversations"]
            if inner_conv and isinstance(inner_conv[0], dict) and ("responses" in inner_conv[0] or "conversation" in inner_conv[0]):
                return "grok"
            return detect_json_source(inner_conv)

    items = data if isinstance(data, list) else [data]
    sample = [x for x in items[:50] if isinstance(x, dict)]
    if not sample:
        return "unknown"

    for entry in sample:
        if "mapping" in entry and isinstance(entry["mapping"], dict):
            return "openai"
        if "chat_messages" in entry and isinstance(entry["chat_messages"], list):
            return "claude"
        if "responses" in entry and isinstance(entry["responses"], list):
            return "grok"
        if "histories" in entry or "msgs" in entry or (isinstance(entry.get("src"), dict) and "is_human" in entry["src"]):
            return "character_ai"

    if any(_is_gemini_activity_entry(entry) for entry in sample):
        return "google"

    for entry in sample:
        if any(k in entry for k in ("messages", "history", "threads", "items", "role", "instruction")):
            return "openai"
        if isinstance(entry.get("chunkedPrompt"), dict):
            return "google"

    return "unknown"


def _extract_chatgpt_or_generic_pairs(data: object) -> List[Tuple[str, str]]:
    """Extract (user_text, assistant_text) pairs from ChatGPT, Claude, Grok, Character.AI, or generic JSON."""
    pairs: List[Tuple[str, str]] = []
    if isinstance(data, dict) and isinstance(data.get("conversations"), list) and "mapping" not in data:
        items = data["conversations"]
    elif isinstance(data, dict) and isinstance(data.get("histories"), list):
        items = data["histories"]
    else:
        items = data if isinstance(data, list) else [data]

    # Case 1: Flat list of {role/sender, text/content/message} or {instruction, response} messages
    if items and all(
        isinstance(x, dict)
        and ("role" in x or "sender" in x or "instruction" in x or "is_human" in x or isinstance(x.get("src"), dict))
        and not any(k in x for k in ("mapping", "chat_messages", "responses", "msgs", "histories"))
        and not _is_gemini_activity_entry(x)
        for x in items[:5]
    ):
        user_bucket: List[str] = []
        ai_bucket: List[str] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if "instruction" in entry and "response" in entry:
                u = str(entry.get("instruction") or "").strip()
                a = str(entry.get("response") or "").strip()
                if u and a:
                    pairs.append((u, a))
                continue
            role, text = _extract_text_from_message(entry)
            if not text:
                continue
            if role == "user":
                if user_bucket and ai_bucket:
                    pairs.append(("\n".join(user_bucket).strip(), "\n".join(ai_bucket).strip()))
                    user_bucket.clear()
                    ai_bucket.clear()
                user_bucket.append(text)
            elif role in ("assistant", "model", "bot"):
                if user_bucket:
                    ai_bucket.append(text)
        if user_bucket and ai_bucket:
            pairs.append(("\n".join(user_bucket).strip(), "\n".join(ai_bucket).strip()))
        if pairs:
            return pairs

    # Case 2: Nested conversation objects (ChatGPT mapping, Claude chat_messages, Grok responses, Character.AI msgs)
    for entry in items:
        if not isinstance(entry, dict):
            continue

        mapping = entry.get("mapping")
        if isinstance(mapping, dict):
            msg_nodes = _ordered_nodes_from_mapping(mapping)
        else:
            def find_messages(obj: object) -> Optional[Iterable]:
                if isinstance(obj, dict):
                    for key in (
                        "mapping",
                        "chat_messages",
                        "responses",
                        "msgs",
                        "messages",
                        "history",
                        "threads",
                        "items",
                        "chunks",
                    ):
                        if key in obj and isinstance(obj[key], (list, dict)):
                            return obj[key]
                    for val in obj.values():
                        res = find_messages(val)
                        if res is not None:
                            return res
                return None

            source = find_messages(entry)
            if source is None:
                continue
            msg_nodes = list(source.values()) if isinstance(source, dict) else list(source)

        user_bucket = []
        ai_bucket = []
        for node in msg_nodes:
            role, text = _extract_text_from_message(node)
            if not text:
                continue
            if role == "user":
                if user_bucket and ai_bucket:
                    pairs.append(("\n".join(user_bucket).strip(), "\n".join(ai_bucket).strip()))
                    user_bucket.clear()
                    ai_bucket.clear()
                user_bucket.append(text)
            elif role in ("assistant", "model", "bot"):
                if user_bucket:
                    ai_bucket.append(text)
        if user_bucket and ai_bucket:
            pairs.append(("\n".join(user_bucket).strip(), "\n".join(ai_bucket).strip()))

    return pairs


def extract_pairs_with_source(data: object) -> Tuple[List[Tuple[str, str]], str]:
    """Detect source provider and return (pairs, source)."""
    source = detect_json_source(data)
    if source == "google":
        gemini_entries = _extract_gemini_entries_from_data(data)
        if gemini_entries:
            return [(u, a) for u, a, _ in gemini_entries], "google"
        aistudio_pairs = _extract_chatgpt_or_generic_pairs(data)
        if aistudio_pairs:
            return aistudio_pairs, "google"
        return [], "google"

    pairs = _extract_chatgpt_or_generic_pairs(data)
    if pairs:
        if source in ("openai", "claude", "grok", "character_ai"):
            return pairs, source
        return pairs, "unknown"

    gemini_entries = _extract_gemini_entries_from_data(data)
    if gemini_entries:
        return [(u, a) for u, a, _ in gemini_entries], "google"
    return [], source


def extract_pairs_from_data(data: object) -> List[Tuple[str, str]]:
    """Extract (user_text, assistant_text) pairs from parsed Google Gemini or OpenAI JSON data."""
    pairs, _ = extract_pairs_with_source(data)
    return pairs


def filter_pairs(
    pairs: Sequence[Tuple[str, str]],
    blocked_phrases: Optional[Sequence[str]] = None,
    protected_words: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[str, str]], int]:
    """Filter out corporate boilerplate responses unless protected words appear."""
    blocked = [p.strip().lower() for p in (blocked_phrases if blocked_phrases is not None else DEFAULT_BLOCKED_PHRASES) if p and p.strip()]
    protected = [w.strip().lower() for w in (protected_words if protected_words is not None else DEFAULT_PROTECTED_WORDS) if w and w.strip()]

    kept: List[Tuple[str, str]] = []
    stripped_count = 0
    for user_text, ai_text in pairs:
        u = user_text.strip()
        a = ai_text.strip()
        if not u or not a:
            continue
        ai_lower = a.lower()
        is_protected = any(word in ai_lower for word in protected) if protected else False
        is_blocked = any(phrase in ai_lower for phrase in blocked) if blocked else False
        if is_blocked and not is_protected:
            stripped_count += 1
            continue
        kept.append((u, a))
    return kept, stripped_count


def _clean_text(text: str) -> str:
    """Strip stray HTML tags (outside code blocks) and ChatGPT private-use citation markers."""
    text = re.sub(r"\ue200.*?\ue201", "", text)

    # Preserve fenced and inline code blocks so HTML code samples remain intact
    placeholders: List[str] = []

    def _stash(match: re.Match) -> str:
        idx = len(placeholders)
        placeholders.append(match.group(0))
        return f"\x00CODEPH_{idx}\x00"

    text = re.sub(r"```[\s\S]*?```", _stash, text)
    text = re.sub(r"`[^`\n]+`", _stash, text)
    text = re.sub(r"<[^>]+>", "", text)
    for idx, original in enumerate(placeholders):
        text = text.replace(f"\x00CODEPH_{idx}\x00", original)
    return text.strip()


def _next_volume_index(output_dir: Path) -> int:
    max_idx = 0
    for f in output_dir.glob("ghost_pcp_*.md"):
        m = re.search(r"ghost_pcp_(\d+)", f.stem, re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _backup_existing_volumes(output_dir: Path) -> Optional[Path]:
    existing = sorted(output_dir.glob("ghost_pcp_*.md"))
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=f"backup_{stamp}_", dir=backup_root))
    for f in existing:
        shutil.copy2(f, backup_dir / f.name)
    for f in existing:
        f.unlink()
    return backup_dir


def write_pcp_volumes(
    pairs: Sequence[Tuple[str, str]],
    output_dir: Path,
    interactions_per_volume: int = INTERACTIONS_PER_VOLUME,
    replace_existing: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> List[Path]:
    """Write interaction pairs into ghost_pcp_01.md, ghost_pcp_02.md, ... safely."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_pairs: List[Tuple[str, str]] = []
    for u_raw, a_raw in pairs:
        u_clean = _clean_text(u_raw)
        a_clean = _clean_text(a_raw)
        if u_clean and a_clean:
            cleaned_pairs.append((u_clean, a_clean))

    total = len(cleaned_pairs)
    if total == 0:
        raise ValueError("No valid conversation interactions were found in the uploaded file(s).")
    if interactions_per_volume <= 0:
        raise ValueError("interactions_per_volume must be greater than zero.")

    if replace_existing:
        _backup_existing_volumes(output_dir)
        start_vol_idx = 1
    else:
        start_vol_idx = _next_volume_index(output_dir)

    written: List[Path] = []
    num_volumes = (total + interactions_per_volume - 1) // interactions_per_volume

    for offset, start in enumerate(range(0, total, interactions_per_volume)):
        vol_idx = start_vol_idx + offset
        filename = f"ghost_pcp_{vol_idx:02d}.md"
        out_path = output_dir / filename
        batch = cleaned_pairs[start : start + interactions_per_volume]
        lines = [f"# **ARCHIVE VOLUME {vol_idx}**", ""]
        for u_clean, a_clean in batch:
            lines.append(f"**USER:** {u_clean}")
            lines.append("")
            lines.append(f"**AI:** {a_clean}")
            lines.append("")
        with out_path.open("x", encoding="utf-8") as output:
            output.write("\n".join(lines))
        written.append(out_path)
        if progress_callback:
            progress_callback(f"Writing {filename} ({offset + 1}/{num_volumes})...", 0.25 + 0.25 * ((offset + 1) / num_volumes))

    return written


def _write_markdown_payloads(
    md_payloads: Sequence[Tuple[str, bytes]],
    output_dir: Path,
    replace_existing: bool = False,
) -> List[Path]:
    non_empty_mds = [
        (n, b.decode("utf-8-sig", errors="replace").strip())
        for n, b in md_payloads
        if b.decode("utf-8-sig", errors="replace").strip()
    ]
    if not non_empty_mds:
        raise ValueError("Uploaded document or Markdown files were empty.")

    if replace_existing:
        _backup_existing_volumes(output_dir)
        start_vol_idx = 1
    else:
        start_vol_idx = _next_volume_index(output_dir)

    def _natural_key(item: Tuple[str, str]):
        nums = re.findall(r"\d+", item[0])
        return (int(nums[-1]) if nums else 999999, item[0].lower())

    sorted_mds = sorted(non_empty_mds, key=_natural_key)
    written_paths: List[Path] = []
    for offset, (_, text) in enumerate(sorted_mds):
        vol_idx = start_vol_idx + offset
        target = output_dir / f"ghost_pcp_{vol_idx:02d}.md"
        with target.open("x", encoding="utf-8") as output:
            output.write(text + "\n")
        written_paths.append(target)
    return written_paths


def _is_gemini_activity_filename(name: str) -> bool:
    low = Path(name).name.lower()
    return low in ("myactivity.json", "my activity.json")


def _is_gemini_activity_html_filename(name: str) -> bool:
    low = Path(name).name.lower()
    return low in ("myactivity.html", "my activity.html")


def _is_chatgpt_conversations_filename(name: str) -> bool:
    low = Path(name).name.lower()
    return low == "conversations.json" or (low.startswith("conversations-") and low.endswith(".json"))


def _is_gemini_takeout_path(path_str: str) -> bool:
    low = path_str.replace("\\", "/").lower()
    return "gemini" in low or "bard" in low


def _is_obviously_unrelated_takeout_path(path_str: str) -> bool:
    parts = [p.lower() for p in Path(path_str).parts[:-1]]
    if any("gemini" in p or "bard" in p for p in parts):
        return False
    for part in parts:
        if any(bad in part for bad in _NON_GEMINI_TAKEOUT_HEADERS):
            return True
    return False


def _is_generated_volume_md(path_str: str, raw_bytes: Optional[bytes] = None) -> bool:
    name = Path(path_str).name.lower()
    if re.match(r"^ghost_pcp_\d+\.md$", name):
        return True
    if raw_bytes is not None:
        head = raw_bytes[:120].decode("utf-8-sig", errors="ignore").lstrip()
        if head.startswith(("# **ARCHIVE VOLUME ", "# ARCHIVE VOLUME ")):
            return True
    return False


_SOURCE_META = {
    "google": ("gemini_export", "Google Gemini"),
    "openai": ("chatgpt_export", "OpenAI ChatGPT"),
    "claude": ("claude_export", "Anthropic Claude"),
    "grok": ("grok_export", "xAI Grok"),
    "character_ai": ("character_ai_export", "Character.AI"),
}


def _source_summary_fields(detected_sources: Set[str], mode_override: Optional[str] = None) -> Dict[str, str]:
    clean = [s for s in sorted(detected_sources) if s in _SOURCE_META]
    if len(clean) == 1:
        src = clean[0]
        mode, label = _SOURCE_META[src]
        return {
            "mode": mode_override or mode,
            "source_type": src,
            "source_label": label,
        }
    if len(clean) > 1:
        labels = " + ".join(_SOURCE_META[s][1] for s in clean)
        return {
            "mode": mode_override or "mixed_export",
            "source_type": "mixed",
            "source_label": labels,
        }
    if mode_override == "markdown_import":
        return {
            "mode": "markdown_import",
            "source_type": "markdown",
            "source_label": "Documents / Markdown",
        }
    return {
        "mode": mode_override or "chatgpt_export",
        "source_type": "generic",
        "source_label": "Conversation Archive",
    }


def _select_zip_json_members(members: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Classify structured export files inside a .zip archive."""
    files = [m for m in members if not m.endswith("/")]
    openai_jsons = [m for m in files if _is_chatgpt_conversations_filename(m)]

    all_activity_jsons = [m for m in files if _is_gemini_activity_filename(m)]
    gemini_tagged_jsons = [m for m in all_activity_jsons if _is_gemini_takeout_path(m)]
    gemini_jsons = gemini_tagged_jsons if gemini_tagged_jsons else [
        m for m in all_activity_jsons if not _is_obviously_unrelated_takeout_path(m)
    ]

    json_dirs_with_activity = {str(Path(m).parent).replace("\\", "/").lower() for m in gemini_jsons}
    all_activity_htmls = [m for m in files if _is_gemini_activity_html_filename(m)]
    gemini_tagged_htmls = [m for m in all_activity_htmls if _is_gemini_takeout_path(m)]
    candidate_htmls = gemini_tagged_htmls if gemini_tagged_htmls else [
        m for m in all_activity_htmls if not _is_obviously_unrelated_takeout_path(m)
    ]
    gemini_htmls = [
        m for m in candidate_htmls
        if str(Path(m).parent).replace("\\", "/").lower() not in json_dirs_with_activity
    ]

    structured = openai_jsons + gemini_jsons
    if structured:
        return structured, gemini_htmls, []

    if gemini_htmls:
        return [], gemini_htmls, []

    fallback_jsons = [
        m for m in files
        if m.lower().endswith(".json")
        and Path(m).name.lower() not in _NON_CONVERSATION_JSON_NAMES
        and not _is_obviously_unrelated_takeout_path(m)
    ]
    return [], [], fallback_jsons


def discover_source_paths(source_path: Path) -> List[Path]:
    """Discover export files in a source file or directory (auto-detecting Google vs OpenAI)."""
    root = Path(source_path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if root.is_file():
        return [root]

    all_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in _IGNORED_DIR_NAMES and not d.startswith(".")
        )
        for fname in sorted(filenames):
            all_files.append(Path(dirpath) / fname)

    openai_jsons: List[Path] = []
    activity_jsons: List[Path] = []
    activity_htmls: List[Path] = []
    other_jsons: List[Path] = []
    jsonl_files: List[Path] = []
    zip_files: List[Path] = []
    top_docs: List[Path] = []

    for f in all_files:
        rel_str = str(f.relative_to(root))
        low_name = f.name.lower()
        suffix = f.suffix.lower()

        if re.match(r"^ghost_pcp_\d+\.md$", low_name):
            continue
        if low_name in _NON_CONVERSATION_JSON_NAMES:
            continue

        if _is_chatgpt_conversations_filename(f.name):
            openai_jsons.append(f)
        elif _is_gemini_activity_filename(f.name):
            activity_jsons.append(f)
        elif _is_gemini_activity_html_filename(f.name):
            activity_htmls.append(f)
        elif suffix == ".json":
            if not _is_obviously_unrelated_takeout_path(rel_str):
                other_jsons.append(f)
        elif suffix == ".jsonl":
            jsonl_files.append(f)
        elif suffix == ".zip":
            zip_files.append(f)
        elif suffix in (".md", ".txt", ".pdf"):
            if f.parent == root and not _is_generated_volume_md(rel_str):
                if suffix == ".md":
                    try:
                        with f.open("rb") as fh:
                            if _is_generated_volume_md(rel_str, fh.read(160)):
                                continue
                    except OSError:
                        pass
                top_docs.append(f)

    # Filter multiple Google Takeout MyActivity.json files to Gemini/Bard when present
    if activity_jsons:
        tagged = [f for f in activity_jsons if _is_gemini_takeout_path(str(f.relative_to(root)))]
        if tagged:
            activity_jsons = tagged
        else:
            activity_jsons = [
                f for f in activity_jsons
                if not _is_obviously_unrelated_takeout_path(str(f.relative_to(root)))
            ]

    json_parent_dirs = {f.parent.resolve() for f in activity_jsons}
    if activity_htmls:
        tagged_html = [f for f in activity_htmls if _is_gemini_takeout_path(str(f.relative_to(root)))]
        cand_html = tagged_html if tagged_html else [
            f for f in activity_htmls
            if not _is_obviously_unrelated_takeout_path(str(f.relative_to(root)))
        ]
        activity_htmls = [f for f in cand_html if f.parent.resolve() not in json_parent_dirs]

    # Validate generic .json candidates quickly so config/metadata JSONs aren't picked as exports
    valid_other_jsons: List[Path] = []
    if not openai_jsons and not activity_jsons:
        for f in other_jsons:
            try:
                parsed = json.loads(f.read_text(encoding="utf-8-sig", errors="replace"))
                if detect_json_source(parsed) in _SOURCE_META:
                    valid_other_jsons.append(f)
            except Exception:
                continue

    structured = openai_jsons + activity_jsons + activity_htmls + valid_other_jsons + jsonl_files
    if structured:
        # Avoid re-ingesting unextracted .zip copies or subdirectory volume exports alongside raw exports
        return structured + top_docs

    if zip_files:
        return zip_files + top_docs

    # Fallback: folder of .md/.txt/.pdf files (e.g. pre-split archive volumes)
    return [
        f for f in all_files
        if f.suffix.lower() in (".md", ".txt", ".pdf")
        and not _is_generated_volume_md(f.name)
    ]


def ingest_files(
    file_items: Sequence[Tuple[str, bytes]],
    output_dir: Path,
    blocked_phrases: Optional[Sequence[str]] = None,
    protected_words: Optional[Sequence[str]] = None,
    interactions_per_volume: int = INTERACTIONS_PER_VOLUME,
    replace_existing: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Ingest one or more dropped files (.zip, .json, .jsonl, .html, .md, .txt, .pdf) into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Inspecting archive files...", 0.05)

    all_pairs: List[Tuple[str, str]] = []
    md_payloads: List[Tuple[str, bytes]] = []
    detected_sources: Set[str] = set()

    def _consume_json_data(label: str, data_obj: object) -> None:
        pairs, src = extract_pairs_with_source(data_obj)
        if pairs:
            if src in _SOURCE_META:
                detected_sources.add(src)
                if progress_callback:
                    src_name = _SOURCE_META[src][1]
                    progress_callback(f"Detected {src_name} ({label}): {len(pairs):,} turns...", 0.18)
            all_pairs.extend(pairs)

    for name, raw_bytes in file_items:
        safe_name = Path(name).name
        lower_name = safe_name.lower()
        if lower_name.endswith(".zip"):
            if progress_callback:
                progress_callback(f"Unpacking {safe_name}...", 0.10)
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                members = sorted(zf.namelist())
                structured_jsons, gemini_htmls, fallback_jsons = _select_zip_json_members(members)
                jsonl_members = [m for m in members if not m.endswith("/") and m.lower().endswith(".jsonl")]
                has_exports = bool(structured_jsons or gemini_htmls or jsonl_members)

                md_members = [
                    m for m in members
                    if not m.endswith("/")
                    and m.lower().endswith(".md")
                    and not (has_exports and _is_generated_volume_md(m))
                ]
                doc_members = [
                    m for m in members
                    if not m.endswith("/")
                    and Path(m).suffix.lower() in (".txt", ".pdf")
                    and not _is_obviously_unrelated_takeout_path(m)
                ]

                for jm in structured_jsons:
                    if progress_callback:
                        progress_callback(f"Parsing {Path(jm).name}...", 0.15)
                    data = json.loads(zf.read(jm).decode("utf-8-sig", errors="replace"))
                    _consume_json_data(Path(jm).name, data)

                for hm in gemini_htmls:
                    if progress_callback:
                        progress_callback(f"Parsing {Path(hm).name}...", 0.15)
                    html_text = zf.read(hm).decode("utf-8-sig", errors="replace")
                    html_pairs = extract_gemini_html_pairs(html_text)
                    if html_pairs:
                        detected_sources.add("google")
                        all_pairs.extend(html_pairs)

                for jlm in jsonl_members:
                    if progress_callback:
                        progress_callback(f"Parsing {Path(jlm).name}...", 0.15)
                    rows = [
                        json.loads(line)
                        for line in zf.read(jlm).decode("utf-8-sig", errors="replace").splitlines()
                        if line.strip()
                    ]
                    _consume_json_data(Path(jlm).name, rows)

                if not all_pairs and fallback_jsons:
                    for jm in fallback_jsons:
                        try:
                            data = json.loads(zf.read(jm).decode("utf-8-sig", errors="replace"))
                            _consume_json_data(Path(jm).name, data)
                        except Exception:
                            continue

                for mm in md_members:
                    md_bytes = zf.read(mm)
                    if has_exports and _is_generated_volume_md(mm, md_bytes):
                        continue
                    md_payloads.append((Path(mm).name, md_bytes))
                for dm in doc_members:
                    doc_title = Path(dm).name
                    if progress_callback:
                        progress_callback(f"Extracting {doc_title}...", 0.16)
                    extracted = extract_text_from_document(doc_title, zf.read(dm))
                    formatted = f"# **DOCUMENT: {doc_title}**\n\n{extracted}\n".encode("utf-8")
                    md_payloads.append((doc_title, formatted))

        elif lower_name.endswith(".json"):
            if progress_callback:
                progress_callback(f"Parsing {safe_name}...", 0.15)
            data = json.loads(raw_bytes.decode("utf-8-sig", errors="replace"))
            _consume_json_data(safe_name, data)
        elif lower_name.endswith(".jsonl"):
            if progress_callback:
                progress_callback(f"Parsing {safe_name}...", 0.15)
            rows = [
                json.loads(line)
                for line in raw_bytes.decode("utf-8-sig", errors="replace").splitlines()
                if line.strip()
            ]
            _consume_json_data(safe_name, rows)
        elif lower_name.endswith((".html", ".htm")):
            html_text = raw_bytes.decode("utf-8-sig", errors="replace")
            if is_gemini_takeout_html(html_text):
                if progress_callback:
                    progress_callback(f"Parsing Google Takeout HTML {safe_name}...", 0.15)
                html_pairs = extract_gemini_html_pairs(html_text)
                if html_pairs:
                    detected_sources.add("google")
                    all_pairs.extend(html_pairs)
        elif lower_name.endswith(".md"):
            md_payloads.append((safe_name, raw_bytes))
        elif Path(lower_name).suffix in (".txt", ".pdf"):
            if progress_callback:
                progress_callback(f"Extracting {safe_name}...", 0.15)
            extracted = extract_text_from_document(safe_name, raw_bytes)
            formatted = f"# **DOCUMENT: {safe_name}**\n\n{extracted}\n".encode("utf-8")
            md_payloads.append((safe_name, formatted))

    # If only Markdown/document files were provided
    if md_payloads and not all_pairs:
        written_paths = _write_markdown_payloads(md_payloads, output_dir, replace_existing=replace_existing)
        summary = {
            **_source_summary_fields(set(), mode_override="markdown_import"),
            "interactions_total": 0,
            "interactions_kept": 0,
            "interactions_stripped": 0,
            "volumes_written": len(written_paths),
            "volume_files": [p.name for p in written_paths],
        }
        return summary

    # Deduplicate identical interaction pairs within the imported batch while preserving order
    deduped_pairs: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for u, a in all_pairs:
        key = (u.strip(), a.strip())
        if key[0] and key[1] and key not in seen_pairs:
            seen_pairs.add(key)
            deduped_pairs.append((u, a))
    duplicates_skipped = len(all_pairs) - len(deduped_pairs)
    all_pairs = deduped_pairs

    if progress_callback:
        progress_callback("Filtering corporate refusals and boilerplate...", 0.22)

    kept_pairs, stripped_count = filter_pairs(
        all_pairs,
        blocked_phrases=blocked_phrases,
        protected_words=protected_words,
    )
    written_paths = write_pcp_volumes(
        kept_pairs,
        output_dir=output_dir,
        interactions_per_volume=interactions_per_volume,
        replace_existing=replace_existing,
        progress_callback=progress_callback,
    )
    if md_payloads:
        written_paths.extend(
            _write_markdown_payloads(md_payloads, output_dir, replace_existing=False)
        )
    summary = {
        **_source_summary_fields(detected_sources),
        "interactions_total": len(all_pairs),
        "interactions_kept": len(kept_pairs),
        "interactions_stripped": stripped_count,
        "duplicates_skipped": duplicates_skipped,
        "volumes_written": len(written_paths),
        "volume_files": [p.name for p in written_paths],
    }
    return summary


def ingest_path(
    source_path: Path | str,
    output_dir: Path | str,
    blocked_phrases: Optional[Sequence[str]] = None,
    protected_words: Optional[Sequence[str]] = None,
    interactions_per_volume: int = INTERACTIONS_PER_VOLUME,
    replace_existing: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Discover export files in source_path (file or folder), auto-detect Google/OpenAI, and write volumes."""
    paths = discover_source_paths(Path(source_path))
    if not paths:
        raise ValueError(f"No supported export or document files found in {source_path}")
    file_items: List[Tuple[str, bytes]] = []
    for p in paths:
        if progress_callback:
            progress_callback(f"Reading {p.name}...", 0.04)
        file_items.append((p.name, p.read_bytes()))
    return ingest_files(
        file_items=file_items,
        output_dir=Path(output_dir),
        blocked_phrases=blocked_phrases,
        protected_words=protected_words,
        interactions_per_volume=interactions_per_volume,
        replace_existing=replace_existing,
        progress_callback=progress_callback,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GHostPCP Archive Ingester — auto-detects Google Gemini and OpenAI ChatGPT exports."
    )
    parser.add_argument("source", help="Source export file (.zip, .json, .html, .md, .txt, .pdf) or folder.")
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).resolve().parent / "pcp_data"),
        help="Target output directory for ghost_pcp_*.md volumes (default: ./pcp_data).",
    )
    parser.add_argument("--per-volume", type=int, default=INTERACTIONS_PER_VOLUME, help="Interactions per volume.")
    parser.add_argument("--replace", action="store_true", help="Backup and replace existing ghost_pcp_*.md volumes.")
    parser.add_argument("--protect", action="append", default=[], help="Protected word/name to never strip (repeatable).")
    args = parser.parse_args(argv)

    def _cli_progress(msg: str, frac: float) -> None:
        print(f"[{int(frac * 100):3d}%] {msg}")

    summary = ingest_path(
        source_path=args.source,
        output_dir=args.output,
        protected_words=args.protect or None,
        interactions_per_volume=args.per_volume,
        replace_existing=args.replace,
        progress_callback=_cli_progress,
    )
    print(
        f"Complete ({summary['source_label']}): {summary['interactions_kept']:,} kept / "
        f"{summary['interactions_total']:,} total ({summary['interactions_stripped']:,} stripped) -> "
        f"{summary['volumes_written']} volume(s) in {Path(args.output).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
