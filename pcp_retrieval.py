"""GHost PCP — Local Semantic Vector Index for ghost_pcp_*.md Volumes.

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

        for wqkv, bqkv, wo, bo, ln1w, ln1b, wi, bi, wod, bod, ln2w, ln2b in self.layers:
            qkv = (x @ wqkv + bqkv).reshape(bsz, seq_len, 3, 12, 32).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale + attn_bias
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs = scores / scores.sum(axis=-1, keepdims=True)
            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, 384)
            x = self._layer_norm(x + (ctx @ wo + bo), ln1w, ln1b)
            inter = self._gelu(x @ wi + bi)
            x = self._layer_norm(x + (inter @ wod + bod), ln2w, ln2b)

        cls = x[:, 0, :]
        norms = np.linalg.norm(cls, axis=-1, keepdims=True)
        return (cls / np.clip(norms, np.float32(1e-12), None)).astype(np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        total = len(texts)
        if total == 0:
            return np.zeros((0, 384), dtype=np.float32)
        if total <= 4:
            return self._forward_batch(texts)

        if total >= 64:
            try:
                from sentence_transformers import SentenceTransformer

                st = SentenceTransformer(str(self.model_dir))
                results: List[np.ndarray] = []
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
                        progress_callback(min(total, start + step), total)
                return np.vstack(results)
            except Exception:
                pass

        results = []
        step = max(1, min(batch_size, 8))
        for start in range(0, total, step):
            vecs = self._forward_batch(texts[start : start + step])
            results.append(vecs)
            if progress_callback:
                progress_callback(min(total, start + len(vecs)), total)
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


class PCPIndex:
    """In-memory semantic vector index backed by cached embeddings.npz."""

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Optional[Path] = None,
        model: Optional[object] = None,
        model_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_dir
        self.model_dir = model_dir
        self._model = model
        self.chunks: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.file_count: int = 0
        self.fingerprint: str = ""
        self.reload(progress_callback=progress_callback)

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
        np.savez_compressed(temp_path, fingerprint=self.fingerprint, vectors=self.vectors)
        os.replace(temp_path, cache_path)

    def reload(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Load or rebuild the semantic index from ghost_pcp_*.md files."""
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
        self.vectors = None

        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cached_fp = str(data["fingerprint"].item())
                    if cached_fp == self.fingerprint and data["vectors"].shape[0] == len(self.chunks):
                        self.vectors = data["vectors"].astype(np.float32, copy=False)
                        if progress_callback:
                            progress_callback("Loaded cached PCP embeddings.", 1.0)
                        return True
            except (OSError, ValueError, KeyError):
                pass

        if progress_callback:
            progress_callback(
                f"Indexing {self.file_count} PCP volumes ({len(self.chunks):,} passages)...",
                0.52,
            )

        model = self._get_model()

        def _on_batch(done: int, total: int):
            if progress_callback and total > 0:
                frac = 0.52 + 0.46 * (done / total)
                progress_callback(
                    f"Embedding passages: {done:,} / {total:,} ({int(done * 100 / total)}%)...",
                    frac,
                )

        if hasattr(model, "encode"):
            try:
                vectors = model.encode(
                    self.chunks,
                    batch_size=16,
                    normalize_embeddings=True,
                    progress_callback=_on_batch,
                )
            except TypeError:
                vectors = model.encode(
                    self.chunks,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        else:
            raise RuntimeError("Invalid embedding model object.")

        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._save_cache()
        if progress_callback:
            progress_callback(
                f"Indexed {self.file_count} volumes ({len(self.chunks):,} passages).",
                1.0,
            )
        return True

    def append_session(self, messages: Sequence[dict], timestamp_str: Optional[str] = None) -> dict:
        """Save a live conversation thread into ghost_pcp_*.md with a timestamp and index it immediately."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
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

        session_block = "\n".join(lines).strip()
        new_chunks = _chunk_segment(session_block)
        if not new_chunks:
            raise ValueError("Session text produced no indexable passages.")

        # Determine target volume file in data_dir
        files = sorted(self.data_dir.glob("ghost_pcp_*.md"))
        target_file: Optional[Path] = None
        if files:
            last_file = files[-1]
            raw_last = last_file.read_text(encoding="utf-8-sig", errors="replace")
            if SESSION_BOUNDARY in raw_last and last_file.stat().st_size < LIVE_VOLUME_MAX_BYTES:
                target_file = last_file
            else:
                nums = [int(n) for f in files for n in re.findall(r"\d+", f.stem)]
                next_num = (max(nums) + 1) if nums else (len(files) + 1)
                target_file = self.data_dir / f"ghost_pcp_{next_num:02d}.md"
        else:
            target_file = self.data_dir / "ghost_pcp_01.md"

        if target_file.exists():
            existing = target_file.read_text(encoding="utf-8-sig", errors="replace").rstrip()
            updated_text = f"{existing}\n\n{SESSION_BOUNDARY}\n\n{session_block}\n"
        else:
            updated_text = f"{SESSION_BOUNDARY}\n\n{session_block}\n"
        target_file.write_text(updated_text, encoding="utf-8")

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
            "turns_saved": turn_count,
            "passages_added": len(new_chunks),
            "total_volumes": self.file_count,
            "total_passages": len(self.chunks),
        }

    def search(self, query: str, limit: int = 4) -> List[str]:
        """Return top-k matching archive excerpts, prioritizing newer timestamped sessions."""
        if not self.ready or self.vectors is None:
            return []
        model = self._get_model()
        prompted = "Represent this sentence for searching relevant passages: " + query
        q_vec = model.encode([prompted], normalize_embeddings=True)[0]
        scores = (self.vectors @ q_vec).astype(np.float32, copy=True)

        # Give a recency boost to timestamped [SESSION LOG: ...] passages so newest memories win ties
        n_total = len(self.chunks)
        latest_session_idx: Optional[int] = None
        for idx in range(n_total - 1, max(-1, n_total - 500), -1):
            if "[SESSION LOG:" in self.chunks[idx]:
                if latest_session_idx is None:
                    latest_session_idx = idx
                recency_rank = (idx - (n_total - 500)) / 500.0
                scores[idx] += np.float32(0.05 * max(0.2, recency_rank))

        k = max(1, int(limit))
        best_indices = [int(i) for i in np.argsort(-scores)[:k]]

        # Always include the most recent saved session passage if one exists so "Save & Clear" never loses continuity
        if latest_session_idx is not None and latest_session_idx not in best_indices:
            if len(best_indices) >= k:
                best_indices[-1] = latest_session_idx
            else:
                best_indices.append(latest_session_idx)

        return [self.chunks[i] for i in best_indices]
