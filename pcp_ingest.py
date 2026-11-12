"""GHost PCP — Archive Extractor, Corporate Refusal Stripper, and Volume Writer.

Transforms ChatGPT export archives (.zip or conversations.json), JSONL pairs,
or Markdown files into clean, high-density Markdown volumes:
    ghost_pcp_01.md, ghost_pcp_02.md, ...
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from persona import DEFAULT_BLOCKED_PHRASES, DEFAULT_PROTECTED_WORDS

INTERACTIONS_PER_VOLUME = 1000


def _extract_text_from_message(msg_obj: dict) -> Tuple[str, str]:
    """Return (role, text) from a ChatGPT mapping message or flat message dict."""
    if not isinstance(msg_obj, dict):
        return "", ""
    inner = msg_obj.get("message") if isinstance(msg_obj.get("message"), dict) else msg_obj
    if not isinstance(inner, dict):
        return "", ""

    author = inner.get("author")
    if isinstance(author, dict):
        role = str(author.get("role") or "")
    else:
        role = str(inner.get("role") or "")
    role = role.lower().strip()

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
    elif isinstance(content, str):
        text_parts.append(content)
    elif isinstance(inner.get("text"), str):
        text_parts.append(inner["text"])

    text = "\n".join(p for p in text_parts if p).strip()
    return role, text


def _ordered_nodes_from_mapping(mapping: dict) -> List[dict]:
    """Traverse a ChatGPT conversation mapping in chronological order."""
    nodes = [v for v in mapping.values() if isinstance(v, dict)]
    # Find root node(s) (no parent or parent not in mapping)
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


def extract_pairs_from_data(data: object) -> List[Tuple[str, str]]:
    """Extract (user_text, assistant_text) pairs from parsed JSON export data."""
    pairs: List[Tuple[str, str]] = []
    items = data if isinstance(data, list) else [data]

    # Case 1: Flat list of {role, text/content} or {instruction, response} messages
    if items and all(isinstance(x, dict) and ("role" in x or "instruction" in x) and "mapping" not in x for x in items[:5]):
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

    # Case 2: Standard ChatGPT conversations.json (with mapping or message containers)
    for entry in items:
        if not isinstance(entry, dict):
            continue

        mapping = entry.get("mapping")
        if isinstance(mapping, dict):
            msg_nodes = _ordered_nodes_from_mapping(mapping)
        else:
            # Recursive search for message container
            def find_messages(obj: object) -> Optional[Iterable]:
                if isinstance(obj, dict):
                    for key in ("mapping", "messages", "history", "threads", "items"):
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
    """Strip stray HTML tags and ChatGPT private-use citation markers."""
    # Remove ChatGPT private-use citation blocks (\ue200...\ue201)
    text = re.sub(r"\ue200.*?\ue201", "", text)
    # Strip raw HTML tags while preserving normal prose
    text = re.sub(r"<[^>]+>", "", text)
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

    # Filter out any empty pairs BEFORE touching existing files
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


def ingest_files(
    file_items: Sequence[Tuple[str, bytes]],
    output_dir: Path,
    blocked_phrases: Optional[Sequence[str]] = None,
    protected_words: Optional[Sequence[str]] = None,
    interactions_per_volume: int = INTERACTIONS_PER_VOLUME,
    replace_existing: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Ingest one or more dropped files (.zip, .json, .jsonl, or .md) into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Inspecting archive files...", 0.05)

    all_pairs: List[Tuple[str, str]] = []
    md_payloads: List[Tuple[str, bytes]] = []

    for name, raw_bytes in file_items:
        lower_name = Path(name).name.lower()
        if lower_name.endswith(".zip"):
            if progress_callback:
                progress_callback(f"Unpacking {Path(name).name}...", 0.10)
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                members = sorted(zf.namelist())
                json_members = [
                    m for m in members
                    if not m.endswith("/")
                    and (
                        Path(m).name.lower() == "conversations.json"
                        or (Path(m).name.lower().startswith("conversations-") and m.lower().endswith(".json"))
                    )
                ]
                md_members = [m for m in members if not m.endswith("/") and m.lower().endswith(".md")]
                if json_members:
                    for jm in json_members:
                        if progress_callback:
                            progress_callback(f"Parsing {Path(jm).name}...", 0.15)
                        data = json.loads(zf.read(jm).decode("utf-8-sig", errors="replace"))
                        all_pairs.extend(extract_pairs_from_data(data))
                elif md_members:
                    for mm in md_members:
                        md_payloads.append((Path(mm).name, zf.read(mm)))
                else:
                    # Fallback: try any .json file inside the zip
                    any_json = [m for m in members if not m.endswith("/") and m.lower().endswith(".json")]
                    for jm in any_json:
                        try:
                            data = json.loads(zf.read(jm).decode("utf-8-sig", errors="replace"))
                            all_pairs.extend(extract_pairs_from_data(data))
                        except Exception:
                            continue
        elif lower_name.endswith(".json"):
            if progress_callback:
                progress_callback(f"Parsing {Path(name).name}...", 0.15)
            data = json.loads(raw_bytes.decode("utf-8-sig", errors="replace"))
            all_pairs.extend(extract_pairs_from_data(data))
        elif lower_name.endswith(".jsonl"):
            if progress_callback:
                progress_callback(f"Parsing {Path(name).name}...", 0.15)
            rows = [
                json.loads(line)
                for line in raw_bytes.decode("utf-8-sig", errors="replace").splitlines()
                if line.strip()
            ]
            all_pairs.extend(extract_pairs_from_data(rows))
        elif lower_name.endswith(".md"):
            md_payloads.append((Path(name).name, raw_bytes))

    # If the user dropped pre-built Markdown files directly
    if md_payloads and not all_pairs:
        non_empty_mds = [
            (n, b.decode("utf-8-sig", errors="replace").strip())
            for n, b in md_payloads
            if b.decode("utf-8-sig", errors="replace").strip()
        ]
        if not non_empty_mds:
            raise ValueError("Uploaded Markdown files were empty.")

        if replace_existing:
            _backup_existing_volumes(output_dir)
            start_vol_idx = 1
        else:
            start_vol_idx = _next_volume_index(output_dir)

        # Sort naturally by embedded number if present
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
        return {
            "mode": "markdown_import",
            "interactions_total": 0,
            "interactions_kept": 0,
            "interactions_stripped": 0,
            "volumes_written": len(written_paths),
            "volume_files": [p.name for p in written_paths],
        }

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
    return {
        "mode": "chatgpt_export",
        "interactions_total": len(all_pairs),
        "interactions_kept": len(kept_pairs),
        "interactions_stripped": stripped_count,
        "volumes_written": len(written_paths),
        "volume_files": [p.name for p in written_paths],
    }
