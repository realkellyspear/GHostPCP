# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""GHostPCP — Local Semantic Vector Index for ghost_pcp_*.md Volumes.

Uses BAAI/bge-small-en-v1.5 (384-D) with dual execution support:
- Lightweight native Safetensors + Tokenizers + NumPy engine (~24ms query latency,
  0.999999+ cosine equivalence with SentenceTransformers, zero PyTorch bloat in EXE)
- Optional PyTorch / SentenceTransformers acceleration for bulk indexing when present
- Instant incremental session saving (append_session) with timestamp & recency grounding
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500
OVERLAP = 200
INDEX_VERSION = 1
SESSION_BOUNDARY = "<!-- PCP_SESSION_BOUNDARY -->"
LIVE_VOLUME_MAX_BYTES = 1_200_000


class IngestCancelledError(Exception):
    """Raised when archive ingestion or embedding is cancelled by the user."""


def _resolve_bge_model_dir(custom_dir: Optional[Path] = None) -> Path:
    """Locate BAAI/bge-small-en-v1.5 locally or download via huggingface_hub."""
    if custom_dir and Path(custom_dir).is_dir():
        if (Path(custom_dir) / "model.safetensors").is_file():
            return Path(custom_dir)

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5" / "snapshots"
    if hf_cache.is_dir():
        for snap in sorted(hf_cache.iterdir()):
            if (snap / "model.safetensors").is_file() and (snap / "tokenizer.json").is_file():
                return snap

    from huggingface_hub import snapshot_download
    downloaded = snapshot_download(
        repo_id=MODEL_NAME,
        allow_patterns=["model.safetensors", "tokenizer.json", "config.json", "special_tokens_map.json", "vocab.txt"],
    )
    return Path(downloaded)


class NativeBGEModel:
    """Pure NumPy + Safetensors + Tokenizers implementation of BAAI/bge-small-en-v1.5."""

    def __init__(self, model_dir: Optional[Path] = None):
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        self.model_dir = _resolve_bge_model_dir(model_dir)
        self._st_model = None
        weights = load_file(str(self.model_dir / "model.safetensors"))
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self.word_emb = weights["embeddings.word_embeddings.weight"]
        self.pos_emb = weights["embeddings.position_embeddings.weight"]
        self.tok_type_emb = weights["embeddings.token_type_embeddings.weight"][0]
        self.emb_ln_w = weights["embeddings.LayerNorm.weight"]
        self.emb_ln_b = weights["embeddings.LayerNorm.bias"]

        self.layers = []
        for i in range(12):
            p = f"encoder.layer.{i}."
            wqkv = np.concatenate(
                [
                    weights[p + "attention.self.query.weight"],
                    weights[p + "attention.self.key.weight"],
                    weights[p + "attention.self.value.weight"],
                ],
                axis=0,
            ).T.copy()
            bqkv = np.concatenate(
                [
                    weights[p + "attention.self.query.bias"],
                    weights[p + "attention.self.key.bias"],
                    weights[p + "attention.self.value.bias"],
                ],
                axis=0,
            )
            self.layers.append(
                (
                    wqkv,
                    bqkv,
                    weights[p + "attention.output.dense.weight"].T.copy(),
                    weights[p + "attention.output.dense.bias"],
                    weights[p + "attention.output.LayerNorm.weight"],
                    weights[p + "attention.output.LayerNorm.bias"],
                    weights[p + "intermediate.dense.weight"].T.copy(),
                    weights[p + "intermediate.dense.bias"],
                    weights[p + "output.dense.weight"].T.copy(),
                    weights[p + "output.dense.bias"],
                    weights[p + "output.LayerNorm.weight"],
                    weights[p + "output.LayerNorm.bias"],
                )
            )

    @staticmethod
    def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + np.float32(1e-12)) * w + b

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        c0 = np.float32(0.5)
        c1 = np.float32(math.sqrt(2.0 / math.pi))
        c2 = np.float32(0.044715)
        return c0 * x * (np.float32(1.0) + np.tanh(c1 * (x + c2 * x * x * x)))

    def _forward_batch(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.float32)
        bsz, seq_len = ids.shape
        pos = np.arange(seq_len, dtype=np.int64)[None, :]
        x = self.word_emb[ids] + self.pos_emb[pos] + self.tok_type_emb
        x = self._layer_norm(x, self.emb_ln_w, self.emb_ln_b)
        attn_bias = (np.float32(1.0) - mask[:, None, None, :]) * np.float32(-10000.0)
        scale = np.float32(1.0 / math.sqrt(32.0))
        sqrt_2_pi = np.float32(math.sqrt(2.0 / math.pi))
        c_gelu = np.float32(0.044715)
        half = np.float32(0.5)
        one = np.float32(1.0)

        for wqkv, bqkv, wo, bo, ln1w, ln1b, wi, bi, wod, bod, ln2w, ln2b in self.layers:
            # Flatten batch and sequence axes so NumPy uses 2D multi-threaded SGEMM.
            qkv = (x.reshape(-1, 384) @ wqkv + bqkv).reshape(bsz, seq_len, 3, 12, 32).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = np.matmul(q, k.swapaxes(-1, -2))
            scores *= scale
            scores += attn_bias
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            ctx = np.matmul(scores, v).transpose(0, 2, 1, 3).reshape(-1, 384)
            x = self._layer_norm((ctx @ wo + bo).reshape(bsz, seq_len, 384) + x, ln1w, ln1b)
            inter = x.reshape(-1, 384) @ wi + bi
            gelu_arg = inter * inter * inter
            gelu_arg *= c_gelu
            gelu_arg += inter
            gelu_arg *= sqrt_2_pi
            np.tanh(gelu_arg, out=gelu_arg)
            gelu_arg += one
            inter *= half
            inter *= gelu_arg
            x = self._layer_norm((inter @ wod + bod).reshape(bsz, seq_len, 384) + x, ln2w, ln2b)

        cls = x[:, 0, :]
        norms = np.linalg.norm(cls, axis=-1, keepdims=True)
        return (cls / np.clip(norms, np.float32(1e-12), None)).astype(np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        start_offset: int = 0,
        total_count: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        total = len(texts)
        overall_total = total_count if total_count is not None else (start_offset + total)
        if total == 0:
            return np.zeros((0, 384), dtype=np.float32)
        if total <= 4:
            vecs = self._forward_batch(texts)
            if progress_callback:
                try:
                    progress_callback(min(overall_total, start_offset + total), overall_total)
                except IngestCancelledError as exc:
                    exc.partial_vectors = vecs
                    raise
            return vecs

        if total >= 64 or self._st_model is not None:
            results: List[np.ndarray] = []
            try:
                if self._st_model is None:
                    from sentence_transformers import SentenceTransformer

                    self._st_model = SentenceTransformer(str(self.model_dir))
                st = self._st_model
                step = 64
                for start in range(0, total, step):
                    sub = list(texts[start : start + step])
                    vecs = st.encode(
                        sub,
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    results.append(np.asarray(vecs, dtype=np.float32))
                    if progress_callback:
                        progress_callback(min(overall_total, start_offset + start + len(sub)), overall_total)
                return np.vstack(results)
            except IngestCancelledError as exc:
                if results:
                    exc.partial_vectors = np.vstack(results)
                raise
            except Exception:
                pass

        results = []
        step = max(1, min(batch_size, 8))
        try:
            for start in range(0, total, step):
                vecs = self._forward_batch(texts[start : start + step])
                results.append(vecs)
                if progress_callback:
                    progress_callback(min(overall_total, start_offset + start + len(vecs)), overall_total)
        except IngestCancelledError as exc:
            if results:
                exc.partial_vectors = np.vstack(results)
            raise
        return np.vstack(results)


def _chunk_segment(text: str) -> List[str]:
    """Chunk a single text segment into overlapping passages."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - OVERLAP)

    header_match = re.match(r"^\s*#\s+\*\*DOCUMENT:\s*(.+?)\*\*", text)
    if header_match and chunks:
        doc_name = re.sub(r"\s+\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]$", "", header_match.group(1).strip()).strip()
        lines_after = [ln.strip() for ln in text.splitlines()[1:] if ln.strip()][:2]
        title_parts = [ln for ln in lines_after if len(ln) <= 80 and not ln.startswith("#")]
        doc_label = f'{doc_name} ("{" — ".join(title_parts)}")' if title_parts else doc_name
        total = len(chunks)
        labeled: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            if total == 1:
                pos_label = "Full Document"
            elif idx == 1:
                pos_label = "Opening"
            elif idx == total:
                pos_label = "Ending"
            elif idx <= max(2, int(total * 0.25)):
                pos_label = "Early"
            elif idx >= min(total - 1, int(total * 0.75) + 1):
                pos_label = "Late"
            else:
                pos_label = "Middle"
            labeled.append(f"[Document: {doc_label} | Passage {idx} of {total} ({pos_label})]\n{chunk}")
        return labeled

    return chunks


def _compute_files_fingerprint(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{MODEL_NAME}:{CHUNK_SIZE}:{OVERLAP}:{INDEX_VERSION}".encode("utf-8"))
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def load_pcp_chunks(data_dir: Path) -> Tuple[List[str], str, int]:
    """Load and chunk all ghost_pcp_*.md files in data_dir."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("ghost_pcp_*.md"))
    if not files:
        raise ValueError(f"No ghost_pcp_*.md files found in {data_dir}")

    digest = hashlib.sha256()
    digest.update(f"{MODEL_NAME}:{CHUNK_SIZE}:{OVERLAP}:{INDEX_VERSION}".encode("utf-8"))
    chunks: List[str] = []

    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(raw).digest())
        text = raw.decode("utf-8-sig", errors="replace")
        if SESSION_BOUNDARY in text:
            for segment in text.split(SESSION_BOUNDARY):
                seg = segment.strip()
                if seg:
                    chunks.extend(_chunk_segment(seg))
        else:
            chunks.extend(_chunk_segment(text))

    if not chunks:
        raise ValueError("The PCP Markdown volume files are empty.")
    return chunks, digest.hexdigest(), len(files)


def _chunks_prefix_hash(chunks: Sequence[str]) -> str:
    h = hashlib.sha256()
    for ch in chunks:
        h.update(ch.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


class PCPIndex:
    """In-memory semantic vector index backed by cached embeddings.npz."""

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Optional[Path] = None,
        model: Optional[object] = None,
        model_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cache_only: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_dir
        self.model_dir = model_dir
        self._model = model
        self.chunks: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.file_count: int = 0
        self.fingerprint: str = ""
        self.reload(progress_callback=progress_callback, cache_only=cache_only)

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.vectors is not None and len(self.chunks) == self.vectors.shape[0]

    @property
    def passage_count(self) -> int:
        return len(self.chunks)

    def _get_model(self):
        if self._model is None:
            self._model = NativeBGEModel(model_dir=self.model_dir)
        return self._model

    def _save_cache(self) -> None:
        if self.vectors is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / "embeddings.npz"
        temp_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp_path,
            fingerprint=self.fingerprint,
            vectors=self.vectors,
            prefix_sha256=_chunks_prefix_hash(self.chunks),
        )
        os.replace(temp_path, cache_path)

    def reload(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cache_only: bool = False,
    ) -> bool:
        """Load or rebuild the semantic index from ghost_pcp_*.md files."""
        prev_chunks = list(self.chunks)
        prev_vectors = self.vectors
        self.data_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self.data_dir.glob("ghost_pcp_*.md"))
        if not files:
            self.chunks = []
            self.vectors = None
            self.file_count = 0
            self.fingerprint = ""
            return False

        self.chunks, self.fingerprint, self.file_count = load_pcp_chunks(self.data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / "embeddings.npz"
        part_path = self.cache_dir / "embeddings.part.npz"
        self.vectors = None
        cached_tail_prefix: Optional[np.ndarray] = None

        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cached_fp = str(data["fingerprint"].item())
                    vecs = data["vectors"]
                    if cached_fp == self.fingerprint and vecs.shape[0] == len(self.chunks):
                        self.vectors = vecs.astype(np.float32, copy=False)
                        if part_path.exists():
                            try:
                                part_path.unlink()
                            except OSError:
                                pass
                        if progress_callback:
                            progress_callback("Loaded cached PCP embeddings.", 1.0)
                        return True
                    if vecs.ndim == 2 and 0 < vecs.shape[0] < len(self.chunks):
                        cached_sha = str(data["prefix_sha256"].item()) if "prefix_sha256" in data else None
                        if cached_sha is None or cached_sha == _chunks_prefix_hash(self.chunks[: vecs.shape[0]]):
                            cached_tail_prefix = vecs.astype(np.float32, copy=False)
            except (OSError, ValueError, KeyError):
                pass

        if cache_only:
            self.vectors = None
            return False

        completed_parts: List[np.ndarray] = []
        done_offset = 0
        if (
            prev_vectors is not None
            and len(prev_chunks) > 0
            and prev_vectors.shape[0] == len(prev_chunks)
            and len(self.chunks) > len(prev_chunks)
            and self.chunks[: len(prev_chunks)] == prev_chunks
        ):
            completed_parts = [prev_vectors.astype(np.float32, copy=False)]
            done_offset = len(prev_chunks)
        elif cached_tail_prefix is not None:
            completed_parts = [cached_tail_prefix]
            done_offset = int(cached_tail_prefix.shape[0])

        if part_path.exists():
            try:
                with np.load(part_path, allow_pickle=False) as part_data:
                    part_fp = str(part_data["fingerprint"].item())
                    part_vecs = part_data["vectors"]
                    if (
                        part_fp == self.fingerprint
                        and part_vecs.ndim == 2
                        and done_offset < part_vecs.shape[0] < len(self.chunks)
                    ):
                        completed_parts = [part_vecs.astype(np.float32, copy=False)]
                        done_offset = int(part_vecs.shape[0])
            except (OSError, ValueError, KeyError):
                pass

        total_chunks = len(self.chunks)
        if progress_callback:
            if done_offset > 0:
                progress_callback(
                    f"Resuming indexing at {done_offset:,} / {total_chunks:,} passages...",
                    0.52 + 0.46 * (done_offset / total_chunks),
                )
            else:
                progress_callback(
                    f"Indexing {self.file_count} PCP volumes ({total_chunks:,} passages)...",
                    0.52,
                )

        model = self._get_model()
        slice_size = 256
        cursor = done_offset

        def _save_partial() -> None:
            if not completed_parts:
                return
            merged = np.vstack(completed_parts).astype(np.float32, copy=False)
            temp_part = part_path.with_suffix(".tmp.npz")
            np.savez(temp_part, fingerprint=self.fingerprint, vectors=merged)
            os.replace(temp_part, part_path)

        try:
            while cursor < total_chunks:
                batch_end = min(total_chunks, cursor + slice_size)
                sub_chunks = self.chunks[cursor:batch_end]
                base_done = cursor
                slice_len = len(sub_chunks)
                pending_end_progress: Optional[Tuple[str, float]] = None

                def _on_batch(done: int, total: int, _base: int = base_done, _slen: int = slice_len):
                    nonlocal pending_end_progress
                    if progress_callback and total_chunks > 0:
                        overall = min(total_chunks, _base + done)
                        frac = 0.52 + 0.46 * (overall / total_chunks)
                        msg = f"Embedding passages: {overall:,} / {total_chunks:,} ({int(overall * 100 / total_chunks)}%)..."
                        if done >= _slen:
                            # Defer slice-boundary progress until the slice vectors are appended & checkpointed.
                            pending_end_progress = (msg, frac)
                        else:
                            progress_callback(msg, frac)

                if hasattr(model, "encode"):
                    try:
                        vecs = model.encode(
                            sub_chunks,
                            batch_size=16,
                            normalize_embeddings=True,
                            progress_callback=_on_batch,
                        )
                    except TypeError:
                        vecs = model.encode(
                            sub_chunks,
                            batch_size=32,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        _on_batch(slice_len, slice_len)
                else:
                    raise RuntimeError("Invalid embedding model object.")

                completed_parts.append(np.asarray(vecs, dtype=np.float32))
                cursor = batch_end
                if cursor < total_chunks:
                    _save_partial()
                    if pending_end_progress and progress_callback:
                        progress_callback(*pending_end_progress)
        except IngestCancelledError as exc:
            partial = getattr(exc, "partial_vectors", None)
            if partial is not None and len(partial) > 0:
                completed_parts.append(np.asarray(partial, dtype=np.float32))
            _save_partial()
            raise

        self.vectors = np.vstack(completed_parts).astype(np.float32, copy=False)
        self._save_cache()
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass
        if progress_callback:
            progress_callback(
                f"Indexed {self.file_count} volumes ({total_chunks:,} passages).",
                1.0,
            )
        return True

    def _append_text_block(self, block_text: str, ts: str) -> dict:
        """Append a pre-formatted Markdown block into ghost_pcp_*.md and index it incrementally."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not _chunk_segment(block_text):
            raise ValueError("Text produced no indexable passages.")

        files = sorted(self.data_dir.glob("ghost_pcp_*.md"))
        target_file: Optional[Path] = None
        if files:
            last_file = files[-1]
            raw_last = last_file.read_bytes().decode("utf-8-sig", errors="replace")
            if SESSION_BOUNDARY in raw_last and last_file.stat().st_size < LIVE_VOLUME_MAX_BYTES:
                target_file = last_file
            else:
                nums = [int(n) for f in files for n in re.findall(r"\d+", f.stem)]
                next_num = (max(nums) + 1) if nums else (len(files) + 1)
                target_file = self.data_dir / f"ghost_pcp_{next_num:02d}.md"
        else:
            target_file = self.data_dir / "ghost_pcp_01.md"

        if target_file.exists():
            existing = target_file.read_bytes().decode("utf-8-sig", errors="replace").rstrip()
            updated_text = f"{existing}\n\n{SESSION_BOUNDARY}\n\n{block_text}\n"
        else:
            updated_text = f"{SESSION_BOUNDARY}\n\n{block_text}\n"
        target_file.write_text(updated_text, encoding="utf-8", newline="\n")

        # Chunk directly from the written disk bytes so Windows/POSIX line endings stay exact with load_pcp_chunks().
        written_seg = (
            target_file.read_bytes()
            .decode("utf-8-sig", errors="replace")
            .split(SESSION_BOUNDARY)[-1]
            .strip()
        )
        new_chunks = _chunk_segment(written_seg)

        model = self._get_model()
        new_vecs = np.asarray(
            model.encode(new_chunks, batch_size=8, normalize_embeddings=True),
            dtype=np.float32,
        )

        if self.vectors is None or not self.chunks:
            self.chunks = list(new_chunks)
            self.vectors = new_vecs
        else:
            self.chunks.extend(new_chunks)
            self.vectors = np.vstack([self.vectors, new_vecs]).astype(np.float32, copy=False)

        all_files = sorted(self.data_dir.glob("ghost_pcp_*.md"))
        self.file_count = len(all_files)
        self.fingerprint = _compute_files_fingerprint(all_files)
        self._save_cache()

        return {
            "saved_to": target_file.name,
            "timestamp": ts,
            "passages_added": len(new_chunks),
            "total_volumes": self.file_count,
            "total_passages": len(self.chunks),
        }

    def append_session(self, messages: Sequence[dict], timestamp_str: Optional[str] = None) -> dict:
        """Save a live conversation thread into ghost_pcp_*.md with a timestamp and index it immediately."""
        ts = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"## **[SESSION LOG: {ts}]**", ""]
        turn_count = 0
        for m in messages:
            role = str(m.get("role") or "").lower().strip()
            content = str(m.get("content") or "").strip()
            content = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", content, flags=re.IGNORECASE).strip()
            if not content or role not in ("user", "assistant"):
                continue
            speaker = "USER" if role == "user" else "AI"
            lines.append(f"**{speaker} [{ts}]:** {content}")
            lines.append("")
            turn_count += 1

        if turn_count == 0:
            raise ValueError("No conversation messages to save.")

        result = self._append_text_block("\n".join(lines).strip(), ts)
        result["turns_saved"] = turn_count
        return result

    def append_document(self, filename: str, text: str, timestamp_str: Optional[str] = None) -> dict:
        """Append an extracted document (.pdf/.txt/.md) into ghost_pcp_*.md and index it immediately."""
        clean = str(text or "").strip()
        if not clean:
            raise ValueError("Document contains no readable text to save.")
        ts = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_name = Path(filename).name
        files = sorted(self.data_dir.glob("ghost_pcp_*.md"))
        if files:
            raw_last = files[-1].read_text(encoding="utf-8-sig", errors="replace")
            header_prefix = f"# **DOCUMENT: {safe_name}"
            if header_prefix in raw_last and (
                clean in raw_last
                or (len(clean) >= 400 and clean[:400] in raw_last and clean[-400:] in raw_last)
            ):
                return {
                    "saved_to": files[-1].name,
                    "timestamp": ts,
                    "passages_added": 0,
                    "total_volumes": self.file_count,
                    "total_passages": len(self.chunks),
                    "filename": safe_name,
                }
        doc_block = f"# **DOCUMENT: {safe_name} [{ts}]**\n\n{clean}"
        result = self._append_text_block(doc_block, ts)
        result["filename"] = safe_name
        return result

    def search(self, query: str, limit: int = 4) -> List[str]:
        """Return top-k matching archive excerpts, prioritizing newer timestamped sessions."""
        if not self.ready or self.vectors is None:
            return []
        model = self._get_model()
        prompted = "Represent this sentence for searching relevant passages: " + query
        q_vec = model.encode([prompted], normalize_embeddings=True)[0]
        raw_scores = (self.vectors @ q_vec).astype(np.float32, copy=True)
        scores = raw_scores.copy()

        n_total = len(self.chunks)
        # Exported archive volumes place newest conversations first (index 0) and oldest last;
        # apply a gentle chronological prior so recent persona chapters break close ties.
        scores += np.linspace(0.035, 0.0, n_total, dtype=np.float32)

        # Prioritize uploaded documents (# **DOCUMENT: / [Document:) and recent saved sessions in the tail
        # when their semantic similarity actually clears a relevance threshold.
        tail_window = min(n_total, 2000)
        tail_start = n_total - tail_window
        for idx in range(n_total - 1, tail_start - 1, -1):
            chunk = self.chunks[idx]
            if ("[Document:" in chunk or "# **DOCUMENT:" in chunk) and raw_scores[idx] >= 0.55:
                scores[idx] += np.float32(0.075)
            elif "[SESSION LOG:" in chunk and raw_scores[idx] >= 0.58:
                recency_rank = (idx - tail_start + 1) / max(1, tail_window)
                scores[idx] += np.float32(0.05 * max(0.25, recency_rank))

        k = max(1, int(limit))
        best_indices = [int(i) for i in np.argsort(-scores)[:k]]
        return [self.chunks[i] for i in best_indices]
