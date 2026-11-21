#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""
=======================================================================
 GHostPCP — GPU Hosted Pure Concentrated Personality
 "Break Your AI Out of the Cloud and Onto Your Own Hardware"
=======================================================================
 - Hardware target : Local GPU (CUDA / Vulkan) with safe CPU fallback
 - Bind target     : 0.0.0.0:5555 (Loopback + Token-Paired LAN/Tunnel)
 - Core engine     : llama-cpp-python + Local BGE-small-en-v1.5 PCP Index
 - API schema      : OpenAI-compatible (/v1/chat/completions) + PCP Routes
=======================================================================
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import io
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import wave
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import qrcode
import qrcode.image.svg
import sounddevice as sd
import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from gguf import GGUFReader, LlamaFileType
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from pcp_ingest import (
    discover_source_paths,
    extract_pairs_from_data,
    extract_text_from_document,
    ingest_files,
)
from pcp_retrieval import IngestCancelledError, PCPIndex, _extract_message_plain_text
from pcp_web import gather_web_context
from persona import (
    DEFAULT_BLOCKED_PHRASES,
    DEFAULT_PROTECTED_WORDS,
    SYSTEM_PROMPT,
    build_persona_prompt_prefix,
    build_web_grounding_note,
    format_core_facts_block,
    strip_builtin_system_prompt,
)

IS_FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
BACKEND_ROOT = APP_DIR / "backends"
_DLL_DIRECTORY_HANDLES = []


def _probe_nvidia_runtime() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        fallback = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
        if fallback.is_file():
            executable = str(fallback)
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_vulkan_runtime() -> bool:
    executable = shutil.which("vulkaninfo")
    return executable is not None and bool(os.environ.get("VULKAN_SDK"))


def _cuda_runtime_dlls() -> List[Path]:
    roots = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        Path(os.environ.get("CUDA_PATH", "")) / "bin",
    ]
    dlls = []
    for root in roots:
        if root.is_dir():
            dlls.extend(root.rglob("cublas*.dll"))
    unique = {path.name.lower(): path for path in dlls}
    return list(unique.values())


def _configure_runtime_backend() -> tuple[str, Path]:
    package_spec = importlib.util.find_spec("llama_cpp")
    if package_spec is None or package_spec.origin is None:
        raise RuntimeError("llama_cpp Python package is unavailable.")
    package_lib = Path(package_spec.origin).parent / "lib"

    if os.name != "nt":
        # Non-Windows (macOS Metal / Linux CUDA/Vulkan/CPU): use llama_cpp's packaged shared libraries
        selected = "gpu" if sys.platform == "darwin" else ("cuda" if _probe_nvidia_runtime() else "cpu")
        return selected, package_lib

    candidates = {
        backend: BACKEND_ROOT / backend
        for backend in ("cuda", "vulkan", "cpu")
    }
    if not IS_FROZEN or not any(path.is_dir() for path in candidates.values()):
        candidates = {"cuda": package_lib, "vulkan": package_lib, "cpu": package_lib}

    plugins = {
        "cuda": "ggml-cuda.dll",
        "vulkan": "ggml-vulkan.dll",
        "cpu": "ggml-cpu.dll",
    }
    available = {
        name: directory.is_dir() and (directory / plugins[name]).is_file()
        for name, directory in candidates.items()
    }
    requested = os.environ.get("GHOST_BACKEND", "auto").lower()
    if requested not in {"auto", "cuda", "vulkan", "cpu"}:
        raise RuntimeError("GHOST_BACKEND must be auto, cuda, vulkan, or cpu.")

    if requested == "auto":
        if available["cuda"] and _probe_nvidia_runtime():
            selected = "cuda"
        elif available["vulkan"] and _probe_vulkan_runtime():
            selected = "vulkan"
        else:
            selected = "cpu"
    else:
        selected = requested
        if not available[selected]:
            raise RuntimeError(f"Requested {selected} backend library is not installed.")
        if selected == "cuda" and not _probe_nvidia_runtime():
            raise RuntimeError("CUDA backend was requested, but nvidia-smi found no CUDA device.")
        if selected == "vulkan" and not _probe_vulkan_runtime():
            raise RuntimeError("Vulkan backend was requested, but Vulkan SDK/runtime probing is unavailable.")

    source = candidates[selected]
    if IS_FROZEN:
        runtime_lib = RESOURCE_DIR / "llama_cpp" / "lib"
    else:
        runtime_lib = Path(tempfile.gettempdir()) / "GHost-PCP" / "backends" / selected
    runtime_lib.mkdir(parents=True, exist_ok=True)
    for existing in runtime_lib.glob("*.dll"):
        try:
            existing.unlink()
        except OSError:
            pass

    dlls = list(source.glob("*.dll"))
    if not IS_FROZEN and selected in {"cpu", "cuda"}:
        dlls.extend(_cuda_runtime_dlls())
    required = {"llama.dll", "ggml.dll", "ggml-base.dll", plugins[selected]}
    names = {item.name for item in dlls}
    missing = required - names
    if selected in {"cpu", "cuda"} and (source / "ggml-cuda.dll").is_file():
        missing_cuda = {"ggml-cuda.dll"} - names
        if missing_cuda:
            missing |= missing_cuda
        import pefile
        cuda_imports = {
            entry.dll.decode(errors="replace").lower()
            for entry in pefile.PE(str(source / "ggml-cuda.dll")).DIRECTORY_ENTRY_IMPORT
            if entry.dll.decode(errors="replace").lower().startswith("cublas")
        }
        present = {item.name.lower() for item in dlls}
        missing |= cuda_imports - present
    if missing:
        raise RuntimeError(f"{selected} backend is missing required DLLs: {sorted(missing)}")
    for item in dlls:
        target_dll = runtime_lib / item.name
        try:
            shutil.copy2(item, target_dll)
        except OSError:
            pass

    if os.name == "nt":
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_lib)))
        os.environ["PATH"] = str(runtime_lib) + os.pathsep + os.environ.get("PATH", "")
    os.environ["GGML_BACKEND_PATH"] = str(runtime_lib)
    return selected, runtime_lib


BACKEND_NAME, RUNTIME_LLAMA_LIB_DIR = _configure_runtime_backend()

from llama_cpp import Llama, llama_cpp
from llama_cpp.llama_chat_format import MTMDChatHandler

# =====================================================================
# CONFIGURATION
# =====================================================================

def _find_available_port(preferred: int = 5555) -> int:
    if "GHOST_API_PORT" in os.environ:
        return int(os.environ["GHOST_API_PORT"])
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", candidate)) == 0:
                continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", candidate))
                return candidate
            except OSError:
                continue
    return preferred


def _release_peer_ghost_anchor_vram() -> bool:
    """If a separate GHoSt-Anchor instance on 5555 is holding GPU VRAM, ask it to unload."""
    if PORT == 5555:
        return False
    try:
        with urllib.request.urlopen("http://127.0.0.1:5555/v1/status", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("engine") == "GHoSt-Anchor" and data.get("resident_model"):
            req = urllib.request.Request("http://127.0.0.1:5555/v1/unload", method="POST")
            with urllib.request.urlopen(req, timeout=5):
                pass
            time.sleep(0.5)
            return True
    except Exception:
        pass
    return False


BIND_HOST = os.environ.get("GHOST_BIND_HOST", "0.0.0.0")
LOOPBACK_HOST = "127.0.0.1"
PORT = _find_available_port(5555)
MODEL_DIR = APP_DIR / "ghost_models"
PIPER_MODEL_DIR = APP_DIR / "piper_models"
PCP_DATA_DIR = APP_DIR / "pcp_data"
BUNDLED_BGE_DIR = (APP_DIR / "bge_model") if (APP_DIR / "bge_model").is_dir() else (RESOURCE_DIR / "bge_model")
BUNDLED_STT_DIR = (APP_DIR / "stt_model") if (APP_DIR / "stt_model").is_dir() else (RESOURCE_DIR / "stt_model")
SETTINGS_PATH = APP_DIR / "ghost_pcp_settings.json"
PAIRING_TOKEN_PATH = PCP_DATA_DIR / ".pairing_token"
HTML_PATH = (APP_DIR / "ghost_pcp.html") if (APP_DIR / "ghost_pcp.html").is_file() else (RESOURCE_DIR / "ghost_pcp.html")
ASSET_DIR = (APP_DIR / "assets") if (APP_DIR / "assets").is_dir() else (RESOURCE_DIR / "assets")
LOG_PATH = APP_DIR / "ghost_pcp.log"
DEFAULT_N_CTX = 8192
GIB = 1024 ** 3
GPU_SAFETY_RESERVE_BYTES = 768 * 1024 ** 2
GPU_MAX_UTILIZATION = 0.95
N_THREADS = os.cpu_count() or 8
FLASH_ATTN = True
N_BATCH = 512

DEFAULT_SETTINGS = {
    "model": "Qwen3-14B-Q5_K_M.gguf",
    "mmproj_model": "auto",
    "chat_format": "auto",
    "system_prompt": "",
    "pcp_enabled": True,
    "pcp_top_k": 4,
    "tts_voice": "silas",
    "blocked_phrases": "\n".join(DEFAULT_BLOCKED_PHRASES),
    "protected_words": "\n".join(DEFAULT_PROTECTED_WORDS),
    "show_reasoning": False,
    "web_enabled": False,
    "web_provider": "auto",
    "web_searxng_url": "",
    "web_api_key": "",
    "web_search_mode": "smart",
    "web_cite_sources": True,
    "auto_continuity_summary": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": 0.05,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "seed": None,
    "max_tokens": 512,
    "stop_sequences": "",
    "n_ctx": DEFAULT_N_CTX,
}

# =====================================================================
# LOGGING
# =====================================================================

log_handlers: List[logging.Handler] = [logging.FileHandler(LOG_PATH, encoding="utf-8")]
if sys.stdout is not None:
    log_handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GHostPCP] %(levelname)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=log_handlers,
)
log = logging.getLogger("ghost_pcp")

GHOST_BANNER = r"""
   ____ _   _       ____  _     ____   ____ ____  
  / ___| | | | ___ / ___|| |_  |  _ \ / ___|  _ \ 
 | |  _| |_| |/ _ \\___ \| __| | |_) | |   | |_) |
 | |_| |  _  | (_) |___) | |_  |  __/| |___|  __/ 
  \____|_| |_|\___/|____/ \__| |_|    \____|_|    

    GHostPCP // GPU HOSTED PURE CONCENTRATED PERSONALITY
    Break Your AI Out of the Cloud and Onto Your Own Hardware
=================================================================
"""

# =====================================================================
# RUNTIME STATE
# =====================================================================

class EngineState:
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.loaded_model: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.loaded_gpu_layers: Optional[int] = None
        self.loaded_context_size: Optional[int] = None
        self.loaded_kv_cache_type: Optional[str] = None
        self.loaded_chat_format: Optional[str] = None
        self.loaded_mmproj: Optional[str] = None
        self.loaded_vision_capable: bool = False
        self.locked = False
        self.busy = False
        self.streaming = False
        self.generation_id = 0
        self.abort_event = threading.Event()
        self._lease = threading.Lock()
        self.conversation: List[Dict[str, Any]] = []
        self._history_lock = threading.Lock()
        self.prior_assistant_tokens: List[int] = []
        self.last_turn_meta: Dict[str, Any] = {}

    def begin_generation(self, streaming: bool) -> int:
        with self._lease:
            if self.locked:
                raise HTTPException(status_code=409, detail="Model change in progress.")
            if self.busy:
                raise HTTPException(status_code=409, detail="A generation is already in progress.")
            if self.llm is None:
                raise HTTPException(status_code=409, detail="No model resident. Open CONFIG and click MOUNT first.")
            self.busy = True
            self.streaming = streaming
            self.abort_event.clear()
            self.generation_id += 1
            return self.generation_id

    def end_generation(self, generation_id: Optional[int] = None) -> None:
        with self._lease:
            if generation_id is not None and generation_id != self.generation_id:
                return
            self.busy = False
            self.streaming = False
            self.abort_event.clear()

    def request_abort(self) -> bool:
        with self._lease:
            if not self.busy:
                return False
            self.abort_event.set()
            return True

    @property
    def abort_requested(self) -> bool:
        return self.abort_event.is_set()


class IngestState:
    def __init__(self):
        self.running = False
        self.stage = "Idle"
        self.progress = 0.0
        self.error: Optional[str] = None
        self.last_result: Optional[dict] = None
        self.cancel_event = threading.Event()
        self.cancelled = False
        self._lock = threading.Lock()

    def request_cancel(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.cancel_event.set()
            return True


class TunnelState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.starting: bool = False
        self.error: Optional[str] = None


class ModelDownloadState:
    def __init__(self):
        self.running = False
        self.stage = "Idle"
        self.repo_id = ""
        self.filename = ""
        self.mmproj_filename = ""
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.progress_pct = 0.0
        self.speed_mbps = 0.0
        self.error: Optional[str] = None
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "repo_id": self.repo_id,
                "filename": self.filename,
                "mmproj_filename": self.mmproj_filename,
                "downloaded_bytes": self.downloaded_bytes,
                "total_bytes": self.total_bytes,
                "progress_pct": round(self.progress_pct, 1),
                "speed_mbps": round(self.speed_mbps, 2),
                "error": self.error,
            }


STATE = EngineState()
INGEST_STATE = IngestState()
TUNNEL_STATE = TunnelState()
MODEL_DOWNLOAD_STATE = ModelDownloadState()
PCP_INDEX: Optional[PCPIndex] = None
PAIRING_TOKEN: str = ""
_PIPER_VOICES: Dict[str, Any] = {}
_PIPER_LOCK = threading.Lock()


def _clean_text_for_speech(text: str) -> str:
    """Strip <think> blocks, code blocks, and Markdown asterisks so Piper speaks naturally."""
    cleaned = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", str(text), flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"(?m)^#+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


DEFAULT_VOICE_URLS = {
    "silas": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/john/medium/en_US-john-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/john/medium/en_US-john-medium.onnx.json",
    },
    "kara": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/kristin/medium/en_US-kristin-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json",
    },
}


def scan_piper_voices() -> List[Dict[str, str]]:
    voices: Dict[str, str] = {"silas": "Silas", "kara": "Kara"}
    search_dirs = [
        PIPER_MODEL_DIR,
        RESOURCE_DIR / "piper_models",
    ]
    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.onnx")):
                vid = f.stem.lower()
                if vid not in voices:
                    voices[vid] = vid.capitalize()
    return [{"id": k, "label": v} for k, v in voices.items()]


def _resolve_piper_model_path(voice_name: str) -> Path:
    safe_voice = Path(voice_name.strip().lower()).stem
    candidates = [
        PIPER_MODEL_DIR / f"{safe_voice}.onnx",
        RESOURCE_DIR / "piper_models" / f"{safe_voice}.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path

    if safe_voice in DEFAULT_VOICE_URLS:
        PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target_onnx = PIPER_MODEL_DIR / f"{safe_voice}.onnx"
        target_json = PIPER_MODEL_DIR / f"{safe_voice}.onnx.json"
        urls = DEFAULT_VOICE_URLS[safe_voice]
        log.info(f"Downloading Piper voice '{safe_voice}' from Hugging Face to {PIPER_MODEL_DIR} ...")
        urllib.request.urlretrieve(urls["json"], str(target_json))
        urllib.request.urlretrieve(urls["onnx"], str(target_onnx))
        if target_onnx.is_file():
            return target_onnx

    raise FileNotFoundError(f"Piper voice model '{safe_voice}.onnx' not found in {PIPER_MODEL_DIR}")


def read_aloud(text: str, voice_name: str = "silas", play_locally: bool = True) -> bytes:
    """
    Generates and plays TTS using the local Piper engine.
    voice_name should be "silas" or "kara" based on the UI button clicked.
    """
    cleaned_text = _clean_text_for_speech(text)
    if not cleaned_text:
        raise ValueError("No speakable text in message.")

    model_path = _resolve_piper_model_path(voice_name)
    output_audio = APP_DIR / "current_speech.wav"

    with _PIPER_LOCK:
        wav_bytes: Optional[bytes] = None
        try:
            from piper import PiperVoice

            cache_key = str(model_path.resolve())
            voice = _PIPER_VOICES.get(cache_key)
            if voice is None:
                espeak_dir = RESOURCE_DIR / "piper" / "espeak-ng-data"
                if espeak_dir.is_dir():
                    voice = PiperVoice.load(model_path, espeak_data_dir=espeak_dir)
                else:
                    voice = PiperVoice.load(model_path)
                _PIPER_VOICES[cache_key] = voice

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(cleaned_text, wav_file)
            wav_bytes = buf.getvalue()
            try:
                output_audio.write_bytes(wav_bytes)
            except OSError:
                pass
        except Exception as exc:
            log.warning(f"In-process PiperVoice fallback to CLI: {exc}")
            command = [sys.executable, "-m", "piper", "-m", str(model_path), "-f", str(output_audio)]
            subprocess.run(
                command,
                input=cleaned_text,
                text=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if output_audio.is_file():
                wav_bytes = output_audio.read_bytes()

        if not wav_bytes:
            raise RuntimeError("Piper did not produce audio output.")

        if play_locally:
            try:
                sd.stop()
                data, fs = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                sd.play(data, fs)
            except Exception as exc:
                log.warning(f"sounddevice local playback warning: {exc}")

        return wav_bytes


def stop_read_aloud() -> None:
    try:
        sd.stop()
    except Exception:
        pass


_STT_ENGINE: Optional["_MoonshineSTT"] = None
_STT_LOCK = threading.Lock()
_STT_REQUIRED_FILES = (
    ("onnx/encoder_model_int8.onnx", "encoder_model_int8.onnx"),
    ("onnx/decoder_model_int8.onnx", "decoder_model_int8.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
)


def _is_local_loopback_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    forwarded = (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
    )
    return client_host in ("127.0.0.1", "::1", "localhost") and not forwarded


def _has_local_stt_files(base_dir: Path) -> bool:
    return base_dir.is_dir() and all((base_dir / local_name).is_file() for _, local_name in _STT_REQUIRED_FILES)


def _resolve_stt_model_dir() -> Path:
    for candidate in (APP_DIR / "stt_model", RESOURCE_DIR / "stt_model"):
        if _has_local_stt_files(candidate):
            return candidate

    target_dir = APP_DIR / "stt_model"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        repo_id = "onnx-community/moonshine-base-ONNX"
        log.info(f"Downloading offline Moonshine-Base ONNX speech recognition model ({repo_id})...")
        for remote_name, local_name in _STT_REQUIRED_FILES:
            cached = Path(hf_hub_download(repo_id=repo_id, filename=remote_name))
            shutil.copy2(cached, target_dir / local_name)
        return target_dir
    except Exception as exc:
        raise RuntimeError(
            f"Offline speech recognition model is not present in {target_dir} and could not be downloaded: {exc}"
        ) from exc


class _MoonshineSTT:
    """Pure ONNX Runtime + tokenizers offline Speech-to-Text (Moonshine-Base int8, CPU, 0 VRAM)."""

    SAMPLE_RATE = 16000
    MAX_SEGMENT_SAMPLES = 16000 * 28

    def __init__(self, model_dir: Optional[Path] = None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        resolved = Path(model_dir) if model_dir is not None else _resolve_stt_model_dir()
        self.model_dir = resolved

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(8, os.cpu_count() or 4)
        self.encoder = ort.InferenceSession(
            str(resolved / "encoder_model_int8.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(resolved / "decoder_model_int8.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.encoder_input_name = self.encoder.get_inputs()[0].name
        self.tokenizer = Tokenizer.from_file(str(resolved / "tokenizer.json"))
        cfg = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        self.bos_id = int(cfg.get("decoder_start_token_id") or cfg.get("bos_token_id") or 1)
        self.eos_id = int(cfg.get("eos_token_id") or 2)

    def _decode_segment(self, seg_16k: Any) -> str:
        np = self._np
        enc_out = self.encoder.run(None, {self.encoder_input_name: seg_16k[None, :]})[0]
        max_steps = min(192, max(16, int((len(seg_16k) / self.SAMPLE_RATE) * 6.0) + 12))
        tokens = [self.bos_id]
        for _ in range(max_steps):
            logits = self.decoder.run(
                None,
                {
                    "input_ids": np.array([tokens], dtype=np.int64),
                    "encoder_hidden_states": enc_out,
                },
            )[0]
            nxt = int(np.argmax(logits[0, -1, :]))
            if nxt == self.eos_id:
                break
            if len(tokens) >= 4 and tokens[-3:] == [nxt, nxt, nxt]:
                break
            tokens.append(nxt)
        return re.sub(r"\s+", " ", self.tokenizer.decode(tokens, skip_special_tokens=True)).strip()

    def transcribe_float32(self, audio: Any, sample_rate: int = 16000) -> str:
        np = self._np
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return ""
        sr = int(sample_rate or self.SAMPLE_RATE)
        if sr != self.SAMPLE_RATE and sr > 0:
            n_out = max(1, int(round(len(arr) * self.SAMPLE_RATE / sr)))
            arr = np.interp(np.linspace(0, len(arr) - 1, n_out), np.arange(len(arr)), arr).astype(np.float32)

        if len(arr) < int(0.12 * self.SAMPLE_RATE):
            return ""
        peak = float(np.max(np.abs(arr)))
        if peak < 0.004:
            return ""

        active = np.where(np.abs(arr) >= 0.004)[0]
        if active.size == 0:
            return ""
        pad = int(0.25 * self.SAMPLE_RATE)
        start = max(0, int(active[0]) - pad)
        end = min(len(arr), int(active[-1]) + pad + 1)
        trimmed = np.ascontiguousarray(arr[start:end], dtype=np.float32)
        if len(trimmed) < int(0.12 * self.SAMPLE_RATE):
            return ""

        parts: List[str] = []
        for offset in range(0, len(trimmed), self.MAX_SEGMENT_SAMPLES):
            chunk = trimmed[offset : offset + self.MAX_SEGMENT_SAMPLES]
            if len(chunk) < int(0.12 * self.SAMPLE_RATE):
                continue
            txt = self._decode_segment(chunk)
            if txt:
                parts.append(txt)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _get_stt_engine() -> _MoonshineSTT:
    global _STT_ENGINE
    with _STT_LOCK:
        if _STT_ENGINE is None:
            _STT_ENGINE = _MoonshineSTT()
        return _STT_ENGINE


class _HostAudioRecorder:
    SAMPLE_RATE = 16000
    MAX_SAMPLES = 16000 * 90

    def __init__(self):
        self._buf_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._stream: Optional[Any] = None
        self._chunks: List[Any] = []
        self._total_samples = 0
        self.recording = False
        self.device_name = ""

    def _stop_and_close_stream(self) -> None:
        with self._stream_lock:
            stream = self._stream
            self._stream = None
            self.recording = False
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def start(self) -> str:
        import numpy as np
        self._stop_and_close_stream()
        with self._buf_lock:
            self._chunks = []
            self._total_samples = 0

        try:
            dev = sd.query_devices(kind="input")
            dev_name = str((dev or {}).get("name") or "Default Microphone")
        except Exception:
            dev_name = "Default Microphone"

        def _callback(indata, frames, time_info, status):
            if not self.recording:
                return
            with self._buf_lock:
                if not self.recording or self._total_samples >= self.MAX_SAMPLES:
                    return
                take = min(frames, self.MAX_SAMPLES - self._total_samples)
                if take > 0:
                    self._chunks.append(np.array(indata[:take, 0], dtype=np.float32, copy=True))
                    self._total_samples += take

        with self._stream_lock:
            self.device_name = dev_name
            self.recording = True
            try:
                stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    callback=_callback,
                )
                self._stream = stream
                stream.start()
            except Exception as exc:
                self.recording = False
                bad_stream = self._stream
                self._stream = None
                if bad_stream is not None:
                    try:
                        bad_stream.close()
                    except Exception:
                        pass
                raise RuntimeError(f"Could not open host microphone: {exc}") from exc
        return dev_name

    def stop_and_extract(self) -> Tuple[Any, float]:
        import numpy as np
        self._stop_and_close_stream()
        with self._buf_lock:
            chunks = list(self._chunks)
            self._chunks = []
            self._total_samples = 0
        if not chunks:
            return np.zeros((0,), dtype=np.float32), 0.0
        samples = np.concatenate(chunks).astype(np.float32)
        return samples, round(len(samples) / self.SAMPLE_RATE, 2)

    def cancel(self) -> None:
        self._stop_and_close_stream()
        with self._buf_lock:
            self._chunks = []
            self._total_samples = 0


_AUDIO_REC = _HostAudioRecorder()


def _init_pairing_token() -> str:
    global PAIRING_TOKEN
    PCP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PAIRING_TOKEN_PATH.is_file():
        token = PAIRING_TOKEN_PATH.read_text(encoding="utf-8").strip()
        if len(token) >= 16:
            PAIRING_TOKEN = token
            return token
    token = secrets.token_urlsafe(24)
    PAIRING_TOKEN_PATH.write_text(token, encoding="utf-8")
    PAIRING_TOKEN = token
    return token


def rotate_pairing_token() -> str:
    global PAIRING_TOKEN
    PCP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    PAIRING_TOKEN_PATH.write_text(token, encoding="utf-8")
    PAIRING_TOKEN = token
    return token


# =====================================================================
# MULTI-PERSONA PROFILES & ISOLATED MEMORY FOLDERS
# =====================================================================

_PERSONA_LOCK = threading.RLock()
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _personas_registry_path() -> Path:
    return PCP_DATA_DIR / "personas.json"


def _resolve_persona_folder(folder_rel: str) -> Path:
    root = PCP_DATA_DIR.resolve()
    rel = str(folder_rel or ".").strip()
    if rel in ("", "."):
        return root
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Invalid persona folder outside PCP_DATA_DIR: {rel}")
    return candidate


def _slugify_persona_id(name: str, existing_ids: Set[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "-", str(name or "").lower()).strip("-_") or "persona"
    if base in _WINDOWS_RESERVED_NAMES:
        base = f"{base}-ai"
    candidate = base
    suffix = 2
    while candidate in existing_ids or (PCP_DATA_DIR / "personas" / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _normalize_persona_record(raw: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    pid = str(raw.get("id") or "").strip()
    if not pid:
        return None
    folder = str(raw.get("folder") or f"personas/{pid}").strip()
    try:
        _resolve_persona_folder(folder)
    except ValueError:
        return None
    return {
        "id": pid,
        "name": str(raw.get("name") or pid).strip() or pid,
        "folder": folder,
        "user_name": str(raw.get("user_name") or "").strip(),
        "never_forget": str(raw.get("never_forget") or "").strip(),
        "notes": strip_builtin_system_prompt(str(raw.get("notes") or "")),
        "continuity_notes": str(raw.get("continuity_notes") or "").strip(),
        "tts_voice": str(raw.get("tts_voice") or "silas").strip() or "silas",
        "preferred_model": str(raw.get("preferred_model") or "").strip(),
        "protected_words": str(raw.get("protected_words") or "").strip(),
        "blocked_phrases": str(raw.get("blocked_phrases") or ""),
        "source_type": str(raw.get("source_type") or "custom").strip(),
        "passage_count": int(raw.get("passage_count") or 0),
        "created_at": str(raw.get("created_at") or "").strip(),
        "updated_at": str(raw.get("updated_at") or "").strip(),
    }


def _legacy_default_persona() -> dict:
    try:
        settings = load_settings()
    except Exception:
        settings = dict(DEFAULT_SETTINGS)
    return {
        "id": "default",
        "name": "Existing Archive",
        "folder": ".",
        "user_name": "",
        "never_forget": "",
        "notes": strip_builtin_system_prompt(settings.get("system_prompt", "")),
        "continuity_notes": "",
        "tts_voice": str(settings.get("tts_voice") or "silas"),
        "preferred_model": "",
        "protected_words": str(settings.get("protected_words") or ""),
        "blocked_phrases": str(settings.get("blocked_phrases") or ""),
        "source_type": "existing",
        "passage_count": 0,
        "created_at": "",
        "updated_at": "",
    }


def _load_persona_registry() -> dict:
    with _PERSONA_LOCK:
        reg_path = _personas_registry_path()
        if reg_path.is_file():
            try:
                raw = json.loads(reg_path.read_text(encoding="utf-8"))
                items = [
                    rec for rec in (_normalize_persona_record(p) for p in (raw.get("personas") or []))
                    if rec is not None
                ]
                active_id = raw.get("active_persona_id")
                valid_ids = {p["id"] for p in items}
                if active_id not in valid_ids:
                    active_id = items[0]["id"] if items else None
                return {"version": 1, "active_persona_id": active_id, "personas": items}
            except Exception as exc:
                log.warning(f"Could not read personas.json: {exc}")

        if PCP_DATA_DIR.is_dir() and any(PCP_DATA_DIR.glob("ghost_pcp_*.md")):
            return {
                "version": 1,
                "active_persona_id": "default",
                "personas": [_legacy_default_persona()],
            }

        return {"version": 1, "active_persona_id": None, "personas": []}


def _save_persona_registry(registry: dict) -> dict:
    with _PERSONA_LOCK:
        PCP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        reg_path = _personas_registry_path()
        tmp_path = reg_path.with_name("personas.json.tmp")
        payload = {
            "version": 1,
            "active_persona_id": registry.get("active_persona_id"),
            "personas": registry.get("personas") or [],
        }
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, reg_path)
        for p in payload["personas"]:
            if p.get("folder") not in ("", "."):
                p_dir = _resolve_persona_folder(p["folder"])
                p_dir.mkdir(parents=True, exist_ok=True)
                meta_path = p_dir / "persona.json"
                meta_tmp = p_dir / "persona.json.tmp"
                meta_tmp.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(meta_tmp, meta_path)
        return payload


def _get_persona_by_id(persona_id: Optional[str], registry: Optional[dict] = None) -> Optional[dict]:
    if not persona_id:
        return None
    reg = registry if registry is not None else _load_persona_registry()
    for p in reg.get("personas") or []:
        if p["id"] == persona_id:
            return dict(p)
    return None


def _get_active_persona(registry: Optional[dict] = None) -> Optional[dict]:
    reg = registry if registry is not None else _load_persona_registry()
    active_id = reg.get("active_persona_id")
    if active_id:
        found = _get_persona_by_id(active_id, reg)
        if found:
            return found
    items = reg.get("personas") or []
    return dict(items[0]) if items else None


def _get_active_pcp_dir() -> Path:
    persona = _get_active_persona()
    if persona is None:
        return PCP_DATA_DIR
    return _resolve_persona_folder(persona.get("folder", "."))


_SAFE_DATA_IMAGE_URL_RE = re.compile(
    r"^data:image/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=]+$",
    re.IGNORECASE,
)
_MAX_IMAGE_DATA_URL_CHARS = 12 * 1024 * 1024
_MAX_IMAGES_PER_MESSAGE = 6


def _sanitize_message_content(
    content: Any,
    max_text_chars: int = 200000,
    keep_images: bool = True,
) -> Union[str, List[Dict[str, Any]]]:
    """Validate text and safe base64 raster data-URL image parts; never allow remote/file image URLs."""
    if isinstance(content, str):
        return content[:max_text_chars]
    if not isinstance(content, list):
        return str(content or "")[:max_text_chars]

    cleaned_parts: List[Dict[str, Any]] = []
    remaining_chars = max_text_chars
    image_count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype == "text":
            txt = str(part.get("text") or "")
            if txt and remaining_chars > 0:
                piece = txt[:remaining_chars]
                remaining_chars -= len(piece)
                cleaned_parts.append({"type": "text", "text": piece})
        elif ptype == "image_url" and keep_images and image_count < _MAX_IMAGES_PER_MESSAGE:
            img_obj = part.get("image_url")
            url = str(img_obj.get("url") or "").strip() if isinstance(img_obj, dict) else ""
            if url and len(url) <= _MAX_IMAGE_DATA_URL_CHARS and _SAFE_DATA_IMAGE_URL_RE.match(url):
                image_count += 1
                cleaned_parts.append({"type": "image_url", "image_url": {"url": url}})

    if not cleaned_parts:
        return ""
    if all(p.get("type") == "text" for p in cleaned_parts):
        return "".join(str(p.get("text") or "") for p in cleaned_parts)
    return cleaned_parts


def _save_persona_chat_history(persona_dir: Path, messages: List[Dict[str, Any]]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    cleaned: List[Dict[str, Any]] = []
    for item in messages or []:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        sanitized = _sanitize_message_content(item.get("content"), keep_images=True)
        if sanitized or isinstance(sanitized, list):
            cleaned.append({"role": item["role"], "content": sanitized})
    hist_path = persona_dir / "chat_history.json"
    tmp_path = persona_dir / "chat_history.json.tmp"
    tmp_path.write_text(json.dumps({"messages": cleaned}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, hist_path)


def _load_persona_chat_history(persona_dir: Path) -> List[Dict[str, Any]]:
    hist_path = persona_dir / "chat_history.json"
    if not hist_path.is_file():
        return []
    try:
        raw = json.loads(hist_path.read_text(encoding="utf-8"))
        cleaned: List[Dict[str, Any]] = []
        for item in raw.get("messages") or []:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                continue
            sanitized = _sanitize_message_content(item.get("content"), keep_images=True)
            if sanitized or isinstance(sanitized, list):
                cleaned.append({"role": item["role"], "content": sanitized})
        return cleaned
    except Exception:
        return []


def _facts_path(persona_dir: Path) -> Path:
    return persona_dir / "facts.json"


def _load_persona_facts(persona_dir: Path) -> List[Dict[str, Any]]:
    fpath = _facts_path(persona_dir)
    if not fpath.is_file():
        return []
    try:
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        items = raw.get("facts") if isinstance(raw, dict) else raw
        cleaned: List[Dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject") or "").strip()
            fact = str(item.get("fact") or "").strip()
            if not subj or not fact:
                continue
            aliases = [
                str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()
            ]
            cleaned.append({
                "id": str(item.get("id") or "").strip() or re.sub(r"[^a-z0-9]+", "-", subj.lower()).strip("-") or "fact",
                "subject": subj[:120],
                "aliases": aliases[:12],
                "category": str(item.get("category") or "general").strip()[:40] or "general",
                "fact": fact[:1200],
                "pinned": bool(item.get("pinned", True)),
                "updated_at": str(item.get("updated_at") or "").strip(),
            })
        return cleaned
    except Exception as exc:
        log.warning(f"Could not read facts.json: {exc}")
        return []


def _save_persona_facts(persona_dir: Path, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    persona_dir.mkdir(parents=True, exist_ok=True)
    fpath = _facts_path(persona_dir)
    tmp_path = fpath.with_name("facts.json.tmp")
    tmp_path.write_text(json.dumps({"version": 1, "facts": facts}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, fpath)
    return facts


def _add_or_update_persona_fact(
    persona_dir: Path,
    subject: str,
    fact: str,
    aliases: Optional[List[str]] = None,
    category: str = "general",
    pinned: bool = True,
    fact_id: Optional[str] = None,
) -> Dict[str, Any]:
    with _PERSONA_LOCK:
        facts = _load_persona_facts(persona_dir)
        clean_subj = str(subject or "").strip()[:120]
        clean_fact = str(fact or "").strip()[:1200]
        if not clean_subj or not clean_fact:
            raise HTTPException(status_code=400, detail="Subject and fact text are required.")
        clean_aliases = [str(a).strip()[:80] for a in (aliases or []) if str(a).strip()][:12]
        clean_cat = str(category or "general").strip()[:40] or "general"
        now_iso = datetime.now().isoformat(timespec="seconds")

        target_id = str(fact_id or "").strip()
        if not target_id:
            base = re.sub(r"[^a-z0-9]+", "-", clean_subj.lower()).strip("-") or "fact"
            existing_ids = {f["id"] for f in facts}
            target_id = base
            idx = 2
            while target_id in existing_ids:
                target_id = f"{base}-{idx}"
                idx += 1

        record = {
            "id": target_id,
            "subject": clean_subj,
            "aliases": clean_aliases,
            "category": clean_cat,
            "fact": clean_fact,
            "pinned": bool(pinned),
            "updated_at": now_iso,
        }
        replaced = False
        for i, item in enumerate(facts):
            if item["id"] == target_id:
                facts[i] = record
                replaced = True
                break
        if not replaced:
            facts.append(record)
        _save_persona_facts(persona_dir, facts)
        return record


def _delete_persona_fact(persona_dir: Path, fact_id: str) -> bool:
    with _PERSONA_LOCK:
        facts = _load_persona_facts(persona_dir)
        kept = [f for f in facts if f.get("id") != fact_id]
        if len(kept) == len(facts):
            return False
        _save_persona_facts(persona_dir, kept)
        return True


def _threads_dir(persona_dir: Path) -> Path:
    return persona_dir / "threads"


def _resolve_thread_path(persona_dir: Path, thread_id: str) -> Path:
    clean_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(thread_id or "").strip()).strip(".-")
    if not clean_id or len(clean_id) > 120:
        raise HTTPException(status_code=400, detail="Invalid thread id.")
    tdir = _threads_dir(persona_dir).resolve()
    target = (tdir / f"{clean_id}.json").resolve()
    if target.parent != tdir:
        raise HTTPException(status_code=400, detail="Invalid thread path.")
    return target


def _list_persona_threads(persona_dir: Path) -> List[Dict[str, Any]]:
    tdir = _threads_dir(persona_dir)
    if not tdir.is_dir():
        return []
    threads: List[Dict[str, Any]] = []
    for fpath in sorted(tdir.glob("*.json"), reverse=True):
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            msgs = raw.get("messages") or []
            threads.append({
                "id": fpath.stem,
                "title": str(raw.get("title") or fpath.stem),
                "message_count": len(msgs),
                "preview": str(raw.get("preview") or ""),
                "created_at": str(raw.get("created_at") or ""),
                "updated_at": str(raw.get("updated_at") or ""),
            })
        except Exception:
            continue
    return threads


def _save_persona_thread(
    persona_dir: Path,
    messages: List[Dict[str, Any]],
    title: str = "",
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned: List[Dict[str, Any]] = []
    for item in messages or []:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        sanitized = _sanitize_message_content(item.get("content"), keep_images=True)
        if sanitized or isinstance(sanitized, list):
            cleaned.append({"role": item["role"], "content": sanitized})
    if not cleaned:
        raise HTTPException(status_code=400, detail="Cannot save an empty thread.")

    tdir = _threads_dir(persona_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now().isoformat(timespec="seconds")
    clean_title = str(title or "").strip()[:120]
    first_user = next(
        (_extract_message_plain_text(m["content"]) for m in cleaned if m["role"] == "user"),
        "",
    )
    if not clean_title:
        clean_title = (first_user.splitlines()[0][:64].strip() if first_user else "") or f"Thread {now_iso}"
    preview = first_user[:160].replace("\n", " ").strip()

    if thread_id:
        tpath = _resolve_thread_path(persona_dir, thread_id)
        tid = tpath.stem
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", clean_title.lower()).strip("-")[:32] or "chat"
        tid = f"{stamp}_{slug}"
        tpath = _resolve_thread_path(persona_dir, tid)
        suffix = 2
        while tpath.exists():
            tid = f"{stamp}_{slug}_{suffix}"
            tpath = _resolve_thread_path(persona_dir, tid)
            suffix += 1

    created_at = now_iso
    if tpath.is_file():
        try:
            prev = json.loads(tpath.read_text(encoding="utf-8"))
            created_at = str(prev.get("created_at") or now_iso)
        except Exception:
            pass

    payload = {
        "id": tid,
        "title": clean_title,
        "preview": preview,
        "created_at": created_at,
        "updated_at": now_iso,
        "messages": cleaned,
    }
    tmp_path = tpath.with_name(f"{tid}.json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, tpath)
    return {
        "id": tid,
        "title": clean_title,
        "message_count": len(cleaned),
        "preview": preview,
        "created_at": created_at,
        "updated_at": now_iso,
    }


def _load_persona_thread(persona_dir: Path, thread_id: str) -> Dict[str, Any]:
    tpath = _resolve_thread_path(persona_dir, thread_id)
    if not tpath.is_file():
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    try:
        raw = json.loads(tpath.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read thread: {exc}")
    cleaned: List[Dict[str, Any]] = []
    for item in raw.get("messages") or []:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        sanitized = _sanitize_message_content(item.get("content"), keep_images=True)
        if sanitized or isinstance(sanitized, list):
            cleaned.append({"role": item["role"], "content": sanitized})
    return {
        "id": str(raw.get("id") or tpath.stem),
        "title": str(raw.get("title") or tpath.stem),
        "messages": cleaned,
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
    }


def _delete_persona_thread(persona_dir: Path, thread_id: str) -> bool:
    tpath = _resolve_thread_path(persona_dir, thread_id)
    if not tpath.is_file():
        return False
    tpath.unlink()
    return True


def _create_persona_entry(
    name: str,
    user_name: str = "",
    never_forget: str = "",
    notes: str = "",
    continuity_notes: str = "",
    tts_voice: str = "silas",
    preferred_model: str = "",
    protected_words: str = "",
    blocked_phrases: str = "",
    source_type: str = "custom",
    activate: bool = True,
) -> dict:
    with _PERSONA_LOCK:
        reg = _load_persona_registry()
        personas = list(reg.get("personas") or [])
        existing_ids = {p["id"] for p in personas}
        clean_name = str(name or "").strip() or "New AI"
        pid = _slugify_persona_id(clean_name, existing_ids)
        folder_rel = f"personas/{pid}"
        folder_path = _resolve_persona_folder(folder_rel)
        folder_path.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.now().isoformat(timespec="seconds")
        record = {
            "id": pid,
            "name": clean_name,
            "folder": folder_rel,
            "user_name": str(user_name or "").strip(),
            "never_forget": str(never_forget or "").strip(),
            "notes": strip_builtin_system_prompt(str(notes or "")),
            "continuity_notes": str(continuity_notes or "").strip(),
            "tts_voice": str(tts_voice or "silas").strip() or "silas",
            "preferred_model": str(preferred_model or "").strip(),
            "protected_words": str(protected_words or "").strip(),
            "blocked_phrases": str(blocked_phrases or ""),
            "source_type": str(source_type or "custom").strip() or "custom",
            "passage_count": 0,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        personas.append(record)
        reg["personas"] = personas
        if activate or not reg.get("active_persona_id"):
            reg["active_persona_id"] = pid
        _save_persona_registry(reg)
        return record


def _update_persona_entry(persona_id: str, updates: Dict[str, Any]) -> dict:
    with _PERSONA_LOCK:
        reg = _load_persona_registry()
        personas = list(reg.get("personas") or [])
        target_idx = next((i for i, p in enumerate(personas) if p["id"] == persona_id), None)
        if target_idx is None:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
        rec = dict(personas[target_idx])
        for key in (
            "name",
            "user_name",
            "never_forget",
            "notes",
            "continuity_notes",
            "tts_voice",
            "preferred_model",
            "protected_words",
            "blocked_phrases",
            "source_type",
        ):
            if key in updates and updates[key] is not None:
                val = str(updates[key])
                if key == "notes":
                    val = strip_builtin_system_prompt(val)
                elif key != "blocked_phrases":
                    val = val.strip()
                if key == "name" and not val:
                    continue
                rec[key] = val
        if "passage_count" in updates and updates["passage_count"] is not None:
            rec["passage_count"] = int(updates["passage_count"])
        rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
        personas[target_idx] = rec
        reg["personas"] = personas
        _save_persona_registry(reg)
        return rec


def _persona_summary(persona: dict, active_id: Optional[str]) -> dict:
    folder_path = _resolve_persona_folder(persona.get("folder", "."))
    vol_files = sorted(folder_path.glob("ghost_pcp_*.md")) if folder_path.is_dir() else []
    is_active = persona["id"] == active_id
    ready = False
    passages = int(persona.get("passage_count") or 0)
    if is_active and PCP_INDEX is not None and PCP_INDEX.data_dir.resolve() == folder_path.resolve():
        ready = PCP_INDEX.ready
        passages = PCP_INDEX.passage_count
    else:
        ready = bool(vol_files) and (folder_path / "embeddings.npz").is_file()
    facts_count = len(_load_persona_facts(folder_path)) if folder_path.is_dir() else 0
    threads_count = len(list((_threads_dir(folder_path)).glob("*.json"))) if _threads_dir(folder_path).is_dir() else 0
    return {
        **persona,
        "active": is_active,
        "folder_path": str(folder_path),
        "volumes": len(vol_files),
        "passages": passages,
        "ready": ready,
        "facts_count": facts_count,
        "threads_count": threads_count,
    }


def _activate_persona(persona_id: str) -> dict:
    global PCP_INDEX
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before switching AI personas.")

    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="Wait for current model operation to finish before switching AI personas.")
        STATE.locked = True

    try:
        with _PERSONA_LOCK:
            reg = _load_persona_registry()
            target = _get_persona_by_id(persona_id, reg)
            if target is None:
                raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")

            prev_id = reg.get("active_persona_id")
            prev_persona = _get_persona_by_id(prev_id, reg) if prev_id else None
            if prev_persona and prev_persona["id"] != target["id"]:
                prev_dir = _resolve_persona_folder(prev_persona.get("folder", "."))
                with STATE._history_lock:
                    _save_persona_chat_history(prev_dir, list(STATE.conversation))

            target_dir = _resolve_persona_folder(target.get("folder", "."))
            target_dir.mkdir(parents=True, exist_ok=True)
            reg["active_persona_id"] = target["id"]
            _save_persona_registry(reg)

        if prev_persona is None or prev_persona["id"] != target["id"]:
            loaded_msgs = _load_persona_chat_history(target_dir)
            with STATE._history_lock:
                STATE.conversation = loaded_msgs
            STATE.prior_assistant_tokens = []
            if STATE.llm is not None:
                try:
                    STATE.llm.reset()
                except Exception as exc:
                    log.warning(f"LLM reset warning after persona switch: {exc}")

        bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
        try:
            PCP_INDEX = PCPIndex(target_dir, cache_dir=target_dir, model_dir=bge_dir, cache_only=True)
            if PCP_INDEX.ready:
                _update_persona_entry(target["id"], {"passage_count": PCP_INDEX.passage_count})
        except Exception as exc:
            log.warning(f"PCP Index reload warning for persona '{target['id']}': {exc}")
            PCP_INDEX = None

        updated_reg = _load_persona_registry()
        updated_target = _get_persona_by_id(target["id"], updated_reg) or target
        return {
            "persona": _persona_summary(updated_target, updated_reg.get("active_persona_id")),
            "messages": list(STATE.conversation),
        }
    finally:
        with STATE._lease:
            STATE.locked = False


def _init_pcp_index() -> None:
    global PCP_INDEX
    PCP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    active_dir = _get_active_pcp_dir()
    active_dir.mkdir(parents=True, exist_ok=True)
    bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
    try:
        PCP_INDEX = PCPIndex(active_dir, cache_dir=active_dir, model_dir=bge_dir, cache_only=True)
        if PCP_INDEX.ready:
            log.info(
                f"PCP Index online: {PCP_INDEX.file_count} volume(s), {PCP_INDEX.passage_count:,} passages ({active_dir})."
            )
            if _personas_registry_path().is_file():
                active = _get_active_persona()
                if active:
                    _update_persona_entry(active["id"], {"passage_count": PCP_INDEX.passage_count})
        elif PCP_INDEX.file_count > 0:
            log.info(
                f"Detected {PCP_INDEX.file_count} unindexed PCP volume(s) in {active_dir}. "
                "Open ARCHIVE and click RE-INDEX to finish indexing."
            )
        else:
            log.info("No ghost_pcp_*.md volumes found yet. Waiting for archive import.")
    except Exception as exc:
        log.warning(f"PCP Index initialization deferred: {exc}")
        PCP_INDEX = None


# =====================================================================
# AUTHENTICATION — Seamless on desktop loopback, token-gated on remote
# =====================================================================

def verify_access(request: Request, authorization: str = Header(default="")) -> None:
    client_host = request.client.host if request.client else ""
    forwarded = (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
    )
    is_local_loopback = client_host in ("127.0.0.1", "::1", "localhost", "testclient") and not forwarded
    if is_local_loopback:
        local_hosts = {"127.0.0.1", "::1", "localhost", "testserver", "testclient"}
        request_host = (request.url.hostname or "").lower()
        origin = request.headers.get("origin")
        if request_host in local_hosts:
            if origin:
                try:
                    origin_url = urllib.parse.urlsplit(origin)
                    same_origin = (
                        origin_url.scheme == request.url.scheme
                        and (origin_url.hostname or "").lower() == request_host
                        and origin_url.port == request.url.port
                    )
                except ValueError:
                    same_origin = False
                if same_origin:
                    return
            else:
                return
    expected = f"Bearer {PAIRING_TOKEN}"
    if not PAIRING_TOKEN or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401,
            detail="Remote access requires a valid pairing link. Scan the QR code shown in GHostPCP on your computer.",
        )


# =====================================================================
# NETWORK & TUNNEL HELPERS
# =====================================================================

def get_lan_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            address = sock.getsockname()[0]
            ip_obj = ipaddress.ip_address(address)
            if not ip_obj.is_loopback and not (ip_obj in ipaddress.ip_network("100.64.0.0/10")):
                return address
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if not ip_obj.is_loopback and not (ip_obj in ipaddress.ip_network("100.64.0.0/10")):
                return ip_str
    except OSError:
        pass
    return None


def get_tailscale_ip() -> Optional[str]:
    ts_exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    if Path(ts_exe).is_file() or shutil.which("tailscale"):
        try:
            res = subprocess.run(
                [ts_exe, "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            line = (res.stdout or "").strip().splitlines()
            if line:
                candidate = line[0].strip()
                ipaddress.ip_address(candidate)
                return candidate
        except Exception:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip_str = info[4][0]
            if ipaddress.ip_address(ip_str) in ipaddress.ip_network("100.64.0.0/10"):
                return ip_str
    except OSError:
        pass
    return None


def _find_or_download_cloudflared() -> Path:
    candidates = [APP_DIR / "cloudflared.exe"]
    which_cf = shutil.which("cloudflared")
    if which_cf:
        candidates.insert(0, Path(which_cf))
    for path in candidates:
        if path.is_file():
            return path

    target = APP_DIR / "cloudflared.exe"
    part_target = target.with_suffix(".exe.part")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    log.info(f"Downloading cloudflared tunnel binary from {url} ...")
    try:
        urllib.request.urlretrieve(url, str(part_target))
        os.replace(part_target, target)
    except Exception:
        try:
            part_target.unlink()
        except OSError:
            pass
        raise
    return target


def start_cloudflare_tunnel() -> None:
    if TUNNEL_STATE.process is not None and TUNNEL_STATE.process.poll() is None:
        return
    if TUNNEL_STATE.starting:
        return

    TUNNEL_STATE.starting = True
    TUNNEL_STATE.error = None
    TUNNEL_STATE.public_url = None

    def _worker():
        try:
            cf_path = _find_or_download_cloudflared()
            cmd = [
                str(cf_path),
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{PORT}",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            TUNNEL_STATE.process = proc
            url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            assert proc.stdout is not None
            for line in proc.stdout:
                match = url_pattern.search(line)
                if match:
                    TUNNEL_STATE.public_url = match.group(0)
                    TUNNEL_STATE.starting = False
                    log.info(f"Cloudflare Quick Tunnel active: {TUNNEL_STATE.public_url}")
            if proc.poll() is not None and not TUNNEL_STATE.public_url:
                TUNNEL_STATE.error = "Cloudflare tunnel exited before assigning a URL."
        except Exception as exc:
            log.error(f"Tunnel error: {exc}")
            TUNNEL_STATE.error = str(exc)
        finally:
            TUNNEL_STATE.starting = False

    threading.Thread(target=_worker, daemon=True).start()


def stop_cloudflare_tunnel() -> None:
    proc = TUNNEL_STATE.process
    TUNNEL_STATE.process = None
    TUNNEL_STATE.public_url = None
    TUNNEL_STATE.starting = False
    TUNNEL_STATE.error = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# =====================================================================
# DIRECTORY SCAN & PREFLIGHT VRAM SCHEDULER
# =====================================================================

_VISION_MODEL_RE = re.compile(
    r"(?:^|[-_.])(?:vl|vision|multimodal|llava|pixtral|internvl|minicpm[-_]?v|gemma[-_]?3|gemma[-_]?4|qwen2\.5[-_]?vl|qwen3[-_]?vl)(?:$|[-_.0-9])",
    re.IGNORECASE,
)
_MMPROJ_IGNORE_TOKENS = {
    "mmproj", "model", "gguf", "f16", "fp16", "bf16", "f32", "fp32",
    "q8", "q8_0", "q4", "q4_k_m", "q4_k_s", "q5_k_m", "q6_k",
    "instruct", "chat", "it",
}
_QUANT_SUFFIX_RE = re.compile(
    r"(?:[-_.](?:i?q\d(?:_[a-z0-9_]+)?|[bf]?f(?:p)?(?:16|32)))(?:\.gguf)?$",
    re.IGNORECASE,
)


def _is_mmproj_filename(filename: str) -> bool:
    return "mmproj" in str(filename or "").lower()


def _looks_like_vision_model(filename: str) -> bool:
    return bool(_VISION_MODEL_RE.search(str(filename or "")))


def _family_tokens(filename: str) -> Set[str]:
    stem = _QUANT_SUFFIX_RE.sub("", str(filename or ""))
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    raw_tokens = re.split(r"[^a-z0-9.]+", stem.lower())
    tokens: Set[str] = set()
    for tok in raw_tokens:
        tok = tok.strip(".")
        if not tok or tok in _MMPROJ_IGNORE_TOKENS:
            continue
        tokens.add(tok)
    return tokens


def scan_mmproj_files() -> List[Dict[str, Any]]:
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for f in sorted(MODEL_DIR.glob("*.gguf")):
        if not _is_mmproj_filename(f.name):
            continue
        try:
            file_size_bytes = f.stat().st_size
        except OSError:
            file_size_bytes = 0
        items.append({
            "id": f.name,
            "size_gb": round(file_size_bytes / (1024 ** 3), 2),
            "file_size_bytes": file_size_bytes,
            "file_size_gib": round(file_size_bytes / GIB, 3),
        })
    return items


def _find_paired_mmproj(
    model_filename: str,
    requested_mmproj: str = "",
    strict: bool = False,
) -> Optional[str]:
    req = str(requested_mmproj or "").strip()
    if req.lower() in ("none", "off", "disabled"):
        return None
    if req and req.lower() != "auto":
        if "/" in req or "\\" in req or ".." in req:
            if strict:
                raise HTTPException(status_code=400, detail="Invalid mmproj filename.")
            return None
        candidate = MODEL_DIR / req
        if not candidate.is_file() or not _is_mmproj_filename(req):
            if strict:
                raise HTTPException(status_code=404, detail=f"Vision projector '{req}' not found in ./ghost_models.")
            return None
        return req

    mmproj_files = [m["id"] for m in scan_mmproj_files()]
    if not mmproj_files:
        return None

    model_tokens = _family_tokens(model_filename)
    is_vision = _looks_like_vision_model(model_filename)
    best_name: Optional[str] = None
    best_score = 0

    for cand in mmproj_files:
        cand_tokens = _family_tokens(cand)
        shared = model_tokens & cand_tokens
        if not shared:
            continue
        # Require either >=2 meaningful shared family/size tokens or strong vision signal + distinctive family match
        distinctive = [t for t in shared if not t.isdigit() and len(t) >= 3]
        score = len(shared) * 10 + (8 if distinctive else 0)
        if is_vision and _looks_like_vision_model(cand) and distinctive:
            score += 10
        if len(shared) >= 2 or (is_vision and distinctive):
            if score > best_score:
                best_score = score
                best_name = cand

    return best_name


def scan_models() -> List[Dict[str, Any]]:
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for f in sorted(MODEL_DIR.glob("*.gguf")):
        if _is_mmproj_filename(f.name):
            continue
        try:
            file_size_bytes = f.stat().st_size
        except OSError:
            file_size_bytes = 0
        auto_mmproj = _find_paired_mmproj(f.name, "auto", strict=False)
        models.append({
            "id": f.name,
            "object": "model",
            "created": int(f.stat().st_mtime) if f.exists() else 0,
            "owned_by": "GHostPCP-Local",
            "size_gb": round(file_size_bytes / (1024 ** 3), 2),
            "file_size_bytes": file_size_bytes,
            "file_size_gib": round(file_size_bytes / GIB, 3),
            "auto_mmproj": auto_mmproj,
            "vision_likely": bool(auto_mmproj) or _looks_like_vision_model(f.name),
        })
    return models


def _gguf_value(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.contents()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _probe_nvidia_gpus() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        fallback = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
        if fallback.is_file():
            executable = str(fallback)
    if executable is None:
        return []

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Unable to query NVIDIA VRAM with nvidia-smi: %s", exc)
        return []
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 3:
            continue
        try:
            devices.append({
                "name": row[0].strip(),
                "total_bytes": int(row[1].strip()) * 1024 ** 2,
                "free_bytes": int(row[2].strip()) * 1024 ** 2,
            })
        except ValueError:
            continue
    return devices


def preflight_model(filename: str, context_size: int, mmproj_model: str = "auto") -> dict:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    if STATE.loaded_model is None:
        _release_peer_ghost_anchor_vram()

    strict_mmproj = bool(mmproj_model and str(mmproj_model).strip().lower() not in ("auto", "none", "off", "disabled"))
    paired_mmproj = _find_paired_mmproj(filename, mmproj_model, strict=strict_mmproj)
    mmproj_bytes = 0
    if paired_mmproj:
        try:
            mmproj_bytes = (MODEL_DIR / paired_mmproj).stat().st_size
        except OSError:
            mmproj_bytes = 0

    try:
        reader = GGUFReader(path)
        architecture = str(_gguf_value(reader, "general.architecture") or "unknown")
        layer_count = int(_gguf_value(reader, f"{architecture}.block_count") or 0)
        embedding_length = int(_gguf_value(reader, f"{architecture}.embedding_length") or 0)
        head_count = int(_gguf_value(reader, f"{architecture}.attention.head_count") or 0)
        kv_head_count = int(_gguf_value(reader, f"{architecture}.attention.head_count_kv") or 0)
        key_length = int(
            _gguf_value(reader, f"{architecture}.attention.key_length")
            or _gguf_value(reader, f"{architecture}.rope.dimension_count")
            or 0
        )
        value_length = int(_gguf_value(reader, f"{architecture}.attention.value_length") or key_length)
        sliding_window_value = _gguf_value(reader, f"{architecture}.attention.sliding_window")
        sliding_window = int(sliding_window_value) if sliding_window_value is not None else None
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        file_type = _gguf_value(reader, "general.file_type")
        metadata_parameter_count = _gguf_value(reader, "general.parameter_count")
        tensor_parameter_count = sum(int(tensor.n_elements) for tensor in reader.tensors)
        parameter_count = int(metadata_parameter_count or tensor_parameter_count)
        weight_bytes = sum(int(tensor.n_bytes) for tensor in reader.tensors)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read GGUF metadata: {exc}") from exc

    if layer_count <= 0:
        raise HTTPException(status_code=422, detail="GGUF does not declare a valid block count.")
    if key_length <= 0 and head_count > 0 and embedding_length % head_count == 0:
        key_length = embedding_length // head_count
    if value_length <= 0:
        value_length = key_length
    kv_metadata_supported = kv_head_count > 0 and key_length > 0 and value_length > 0

    try:
        quantization = LlamaFileType(int(file_type)).name if file_type is not None else "unknown"
    except ValueError:
        quantization = f"unknown({file_type})"

    layer_weight_bytes = [0] * layer_count
    non_layer_bytes = 0
    key_layers = set()
    value_layers = set()
    fused_qkv_layers = set()
    has_ssm_state = any(key.startswith(f"{architecture}.ssm.") for key in reader.fields)
    for tensor in reader.tensors:
        parts = tensor.name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            try:
                index = int(parts[1])
            except ValueError:
                non_layer_bytes += int(tensor.n_bytes)
                continue
            if 0 <= index < layer_count:
                layer_weight_bytes[index] += int(tensor.n_bytes)
                if len(parts) >= 4 and parts[2] == "attn_k" and parts[3] == "weight":
                    key_layers.add(index)
                if len(parts) >= 4 and parts[2] == "attn_v" and parts[3] == "weight":
                    value_layers.add(index)
                if len(parts) >= 4 and parts[2] == "attn_qkv" and parts[3] == "weight":
                    fused_qkv_layers.add(index)
            else:
                non_layer_bytes += int(tensor.n_bytes)
        else:
            non_layer_bytes += int(tensor.n_bytes)

    attention_layers = key_layers & value_layers
    if not attention_layers and architecture in {"qwen2", "qwen3"}:
        attention_layers = set(range(layer_count))
    if not attention_layers and architecture == "phi3" and len(fused_qkv_layers) == layer_count:
        attention_layers = fused_qkv_layers
    kv_supported = (
        kv_metadata_supported
        and bool(attention_layers)
        and len(attention_layers) == layer_count
        and sliding_window is None
        and not has_ssm_state
    )
    kv_f16_bytes = 0
    kv_q8_bytes = 0
    kv_method = "unsupported_cpu_fallback"
    if kv_supported:
        kv_f16_bytes = (
            len(attention_layers)
            * context_size
            * kv_head_count
            * (key_length + value_length)
            * 2
        )
        kv_q8_bytes = (kv_f16_bytes * 34 + 63) // 64
        kv_method = "dense_gqa_fp16"
    mmproj_reserve_bytes = int(mmproj_bytes * 1.15) if paired_mmproj else 0
    compute_reserve_bytes = int(0.5 * GIB + 0.04 * weight_bytes) + mmproj_reserve_bytes
    devices = _probe_nvidia_gpus() if BACKEND_NAME == "cuda" else []
    warnings = []
    if paired_mmproj:
        warnings.append(f"Paired vision projector '{paired_mmproj}' (+{round(mmproj_reserve_bytes / GIB, 2)} GiB VRAM reserve).")
    gpu_layers = 0
    gpu_weight_bytes = 0
    kv_cache_bytes = kv_f16_bytes
    kv_type = int(llama_cpp.GGML_TYPE_F16)
    kv_cache_type = "F16"
    usable_bytes = 0
    gpu_name = None
    total_bytes = 0
    free_bytes = 0

    if BACKEND_NAME == "vulkan":
        warnings.append("CPU backend selected; GPU layer offload is disabled.")
    elif len(devices) == 1:
        gpu = devices[0]
        gpu_name = gpu["name"]
        total_bytes = gpu["total_bytes"]
        free_bytes = gpu["free_bytes"]
        usable_bytes = min(
            int(total_bytes * GPU_MAX_UTILIZATION),
            max(0, free_bytes - GPU_SAFETY_RESERVE_BYTES),
        )
        if not kv_supported:
            if not kv_metadata_supported or not attention_layers:
                warnings.append("GGUF attention metadata is incomplete; GPU offload disabled for safety.")
            elif sliding_window is not None:
                warnings.append("Sliding-window KV allocation is not modeled; GPU offload disabled for safety.")
            elif has_ssm_state or len(attention_layers) != layer_count:
                warnings.append("Hybrid SSM/recurrent KV state is not modeled; GPU offload disabled for safety.")
            else:
                warnings.append("KV cache layout is not supported by preflight; GPU offload disabled for safety.")
        else:
            def fitting_layers(kv_bytes: int) -> tuple[int, int]:
                available = usable_bytes - kv_bytes - compute_reserve_bytes - non_layer_bytes
                for candidate in range(layer_count, 0, -1):
                    candidate_bytes = sum(layer_weight_bytes[layer_count - candidate:])
                    if candidate_bytes <= available:
                        return candidate, candidate_bytes
                return 0, 0

            f16_layers, f16_weight_bytes = fitting_layers(kv_f16_bytes)
            q8_layers, q8_weight_bytes = fitting_layers(kv_q8_bytes)
            if q8_layers > f16_layers:
                kv_cache_bytes = kv_q8_bytes
                kv_type = int(llama_cpp.GGML_TYPE_Q8_0)
                kv_cache_type = "Q8_0"
                kv_method = "dense_gqa_q8_0"
                gpu_layers = q8_layers
                gpu_weight_bytes = q8_weight_bytes
                warnings.append("KV cache reduced from FP16 to Q8_0 before reducing GPU layer count.")
            else:
                gpu_layers = f16_layers
                gpu_weight_bytes = f16_weight_bytes
            if gpu_layers == 0:
                warnings.append("Model weights, KV estimate, overhead, and safety reserve leave no layer budget for GPU offload.")
    elif not devices:
        warnings.append("nvidia-smi did not report a GPU; selecting CPU-only loading.")
    else:
        warnings.append("Multiple GPUs are detected; selecting single-GPU conservative fallback.")

    predicted_gpu_bytes = 0
    if gpu_layers:
        predicted_gpu_bytes = non_layer_bytes + gpu_weight_bytes + kv_cache_bytes + compute_reserve_bytes

    return {
        "model": filename,
        "backend": BACKEND_NAME,
        "architecture": architecture,
        "quantization": quantization,
        "parameter_count": parameter_count,
        "weight_bytes": weight_bytes,
        "weight_gib": round(weight_bytes / GIB, 3),
        "context_size": context_size,
        "total_layers": layer_count,
        "attention_layers": len(attention_layers),
        "n_gpu_layers": gpu_layers,
        "kv_cache_type": kv_cache_type,
        "kv_type": kv_type,
        "kv_method": kv_method,
        "gpu_name": gpu_name,
        "gpu_total_bytes": total_bytes,
        "gpu_free_bytes": free_bytes,
        "usable_vram_bytes": usable_bytes,
        "kv_cache_estimate_bytes": kv_cache_bytes,
        "compute_reserve_bytes": compute_reserve_bytes,
        "non_layer_weight_bytes": non_layer_bytes,
        "gpu_layer_weight_bytes": gpu_weight_bytes,
        "predicted_gpu_bytes": predicted_gpu_bytes,
        "mmproj_model": paired_mmproj,
        "mmproj_bytes": mmproj_bytes,
        "vision_capable": bool(paired_mmproj),
        "warnings": warnings,
        "estimate_note": "Tensor bytes are read from the GGUF index. Dense KV is an FP16 upper-bound estimate.",
    }


# =====================================================================
# MODEL LIFECYCLE
# =====================================================================

def unload_current_model() -> None:
    if STATE.llm is not None:
        log.info(f"Unloading resident model '{STATE.loaded_model}' ...")
        try:
            chat_handler = getattr(STATE.llm, "chat_handler", None)
            if chat_handler is not None:
                STATE.llm.chat_handler = None
                del chat_handler
            del STATE.llm
        except Exception as e:
            log.warning(f"Exception during model deletion: {e}")
        STATE.llm = None
        STATE.loaded_model = None
        STATE.loaded_at = None
        STATE.loaded_gpu_layers = None
        STATE.loaded_context_size = None
        STATE.loaded_kv_cache_type = None
        STATE.loaded_chat_format = None
        STATE.loaded_mmproj = None
        STATE.loaded_vision_capable = False

        for _ in range(3):
            gc.collect()
        time.sleep(0.4)
        log.info("Resident model flushed. VRAM released.")


def load_model(
    filename: str,
    context_size: int,
    chat_format: str = "auto",
    mmproj_model: str = "auto",
) -> dict:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="A model change is already in progress.")
        STATE.locked = True
    try:
        unload_current_model()
        preflight = preflight_model(filename, context_size, mmproj_model=mmproj_model)
        gpu_layers = int(preflight["n_gpu_layers"])
        paired_mmproj = preflight.get("mmproj_model")
        log.info(
            f"Loading '{filename}' (preflight n_gpu_layers={gpu_layers}, n_ctx={context_size}, mmproj={paired_mmproj or 'none'}) ..."
        )
        t0 = time.perf_counter()
        try:
            chat_handler = None
            effective_chat_format = None if chat_format == "auto" else chat_format
            if paired_mmproj:
                if MTMDChatHandler is None:
                    msg = f"Vision projector '{paired_mmproj}' found, but MTMDChatHandler is unavailable in llama-cpp-python."
                    log.warning(msg)
                    preflight.setdefault("warnings", []).append(msg)
                    preflight["mmproj_model"] = None
                    preflight["vision_capable"] = False
                    paired_mmproj = None
                else:
                    clip_path = MODEL_DIR / paired_mmproj
                    chat_handler = MTMDChatHandler(
                        clip_model_path=str(clip_path),
                        use_gpu=gpu_layers > 0,
                        verbose=False,
                    )
                    effective_chat_format = None

            llama_kwargs: Dict[str, Any] = {
                "model_path": str(path),
                "n_gpu_layers": gpu_layers,
                "n_ctx": context_size,
                "n_threads": N_THREADS,
                "n_batch": N_BATCH,
                "last_n_tokens_size": 1024,
                "type_k": int(preflight["kv_type"]),
                "type_v": int(preflight["kv_type"]),
                "offload_kqv": gpu_layers > 0,
                "flash_attn": FLASH_ATTN if gpu_layers > 0 else False,
                "chat_format": effective_chat_format,
                "verbose": False,
            }
            if chat_handler is not None:
                llama_kwargs["chat_handler"] = chat_handler

            STATE.llm = Llama(**llama_kwargs)
            orig_init_sampler = STATE.llm._init_sampler

            def _patched_init_sampler(*s_args, **s_kwargs):
                sampler = orig_init_sampler(*s_args, **s_kwargs)
                for tok in STATE.prior_assistant_tokens:
                    try:
                        sampler.accept(int(tok))
                    except Exception:
                        break
                return sampler

            STATE.llm._init_sampler = _patched_init_sampler
        except Exception as exc:
            log.error(f"Load failed: {exc}")
            log.error(traceback.format_exc())
            unload_current_model()
            raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

        dt = time.perf_counter() - t0
        STATE.loaded_model = filename
        STATE.loaded_at = time.time()
        STATE.load_seconds = dt
        STATE.loaded_gpu_layers = gpu_layers
        STATE.loaded_context_size = context_size
        STATE.loaded_kv_cache_type = preflight["kv_cache_type"]
        STATE.loaded_chat_format = chat_format
        STATE.loaded_mmproj = paired_mmproj
        STATE.loaded_vision_capable = bool(paired_mmproj)
        log.info(
            f"Model '{filename}' resident in VRAM (vision={STATE.loaded_vision_capable}). Load time: {dt:.2f}s"
        )
        return preflight
    finally:
        with STATE._lease:
            STATE.locked = False


# =====================================================================
# CURATED MODEL CATALOG & HUGGINGFACE DOWNLOADER
# =====================================================================

_MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "qwen3-8b-q4km",
        "title": "Qwen3 8B Instruct (Q4_K_M)",
        "category": "Fast General + Reasoning",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "save_as": "Qwen3-8B-Q4_K_M.gguf",
        "mmproj_filename": None,
        "mmproj_save_as": None,
        "size_gb": 4.68,
        "vram_note": "Fits smoothly in 8 GB VRAM",
        "vision": False,
        "description": "Sharp everyday chat, coding, and toggleable reasoning at high speed.",
    },
    {
        "id": "llama-3.1-8b-q4km",
        "title": "Meta Llama 3.1 8B Instruct (Q4_K_M)",
        "category": "Balanced Companion",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "save_as": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "mmproj_filename": None,
        "mmproj_save_as": None,
        "size_gb": 4.58,
        "vram_note": "Fits smoothly in 8 GB VRAM",
        "vision": False,
        "description": "Classic, warm conversational baseline with reliable instruction following.",
    },
    {
        "id": "qwen2.5-vl-7b-q4km",
        "title": "Qwen 2.5 VL 7B Instruct + Vision (Q4_K_M)",
        "category": "Vision + OCR + Chat",
        "repo_id": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "save_as": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
        "mmproj_save_as": "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
        "size_gb": 5.15,
        "vram_note": "Recommended 8–12 GB VRAM (includes mmproj)",
        "vision": True,
        "description": "Understands screenshots, documents, handwriting, photos, and UI mockups.",
    },
    {
        "id": "gemma-3-4b-vision-q4km",
        "title": "Gemma 3 4B IT + Vision (Q4_K_M)",
        "category": "Compact Vision",
        "repo_id": "ggml-org/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "save_as": "gemma-3-4b-it-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-model-f16.gguf",
        "mmproj_save_as": "gemma-3-4b-it-mmproj-f16.gguf",
        "size_gb": 3.11,
        "vram_note": "Fits in 6 GB VRAM (includes mmproj)",
        "vision": True,
        "description": "Lightweight Google Gemma 3 model paired with its vision projector.",
    },
    {
        "id": "qwen2.5-3b-q4km",
        "title": "Qwen 2.5 3B Instruct (Q4_K_M)",
        "category": "Ultra-Light / Laptop",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "save_as": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "mmproj_filename": None,
        "mmproj_save_as": None,
        "size_gb": 1.80,
        "vram_note": "Fits in 4 GB VRAM or CPU RAM",
        "vision": False,
        "description": "Fast starter model for laptops, low-VRAM GPUs, or quick testing.",
    },
    {
        "id": "mistral-nemo-12b-q4km",
        "title": "Mistral Nemo 12B Instruct (Q4_K_M)",
        "category": "Creative / 12B Tier",
        "repo_id": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "save_as": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "mmproj_filename": None,
        "mmproj_save_as": None,
        "size_gb": 6.96,
        "vram_note": "Best on 12–16 GB VRAM",
        "vision": False,
        "description": "Nuanced natural dialogue and long-context storytelling.",
    },
]

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _validate_hf_repo_id(repo_id: str) -> str:
    clean = str(repo_id or "").strip()
    if clean.startswith("https://huggingface.co/"):
        clean = clean[len("https://huggingface.co/"):].strip("/")
    elif clean.startswith("http://huggingface.co/"):
        clean = clean[len("http://huggingface.co/"):].strip("/")
    parts = [p for p in clean.split("/") if p]
    if len(parts) >= 2:
        clean = f"{parts[0]}/{parts[1]}"
    if not _HF_REPO_ID_RE.match(clean):
        raise HTTPException(status_code=400, detail="Invalid HuggingFace repo ID. Expected 'owner/repo'.")
    return clean


def _safe_local_model_name(name: str, fallback: str = "") -> str:
    raw = Path(str(name or fallback or "").strip()).name
    if not raw or "/" in raw or "\\" in raw or ".." in raw or not raw.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Model filename must end in .gguf without path separators.")
    return raw


def _list_hf_repo_ggufs(repo_id: str) -> Dict[str, Any]:
    clean_repo = _validate_hf_repo_id(repo_id)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="huggingface_hub is not installed in this environment.") from exc
    try:
        info = HfApi().model_info(clean_repo, files_metadata=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not inspect HuggingFace repo '{clean_repo}': {exc}") from exc

    models: List[Dict[str, Any]] = []
    mmprojs: List[Dict[str, Any]] = []
    for sib in getattr(info, "siblings", None) or []:
        rfilename = str(getattr(sib, "rfilename", "") or "")
        if not rfilename.lower().endswith(".gguf"):
            continue
        size_bytes = int(getattr(sib, "size", 0) or 0)
        entry = {
            "filename": rfilename,
            "local_name": Path(rfilename).name,
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / (1024 ** 3), 2),
        }
        if _is_mmproj_filename(rfilename):
            mmprojs.append(entry)
        else:
            models.append(entry)
    return {"repo_id": clean_repo, "models": models, "mmprojs": mmprojs}


def _stream_hf_file(
    repo_id: str,
    remote_filename: str,
    local_filename: str,
    stage_label: str,
    cancel_event: threading.Event,
) -> Path:
    import requests as req_lib
    from urllib.parse import quote

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = (MODEL_DIR / local_filename).resolve()
    if target.parent != MODEL_DIR.resolve():
        raise ValueError("Invalid target path.")
    if target.is_file() and target.stat().st_size > 0:
        return target

    part_path = target.with_name(f"{target.name}.part")
    encoded_file = quote(str(remote_filename).lstrip("/"), safe="/")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded_file}"

    existing = part_path.stat().st_size if part_path.is_file() else 0
    headers: Dict[str, str] = {"User-Agent": "GHostPCP/1.3.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    t_start = time.perf_counter()
    bytes_this_session = 0
    with req_lib.get(url, headers=headers, stream=True, timeout=(20, 90), allow_redirects=True) as resp:
        if resp.status_code == 416 and existing > 0:
            part_path.unlink(missing_ok=True)
            existing = 0
            headers.pop("Range", None)
            resp2 = req_lib.get(url, headers=headers, stream=True, timeout=(20, 90), allow_redirects=True)
            resp2.raise_for_status()
            return _write_hf_stream(resp2, part_path, target, stage_label, existing, t_start, cancel_event)
        resp.raise_for_status()
        if resp.status_code == 200 and existing > 0:
            # Server ignored Range header; restart cleanly from 0
            existing = 0
        return _write_hf_stream(resp, part_path, target, stage_label, existing, t_start, cancel_event)


def _write_hf_stream(
    resp: Any,
    part_path: Path,
    target: Path,
    stage_label: str,
    resume_offset: int,
    t_start: float,
    cancel_event: threading.Event,
) -> Path:
    content_len = int(resp.headers.get("Content-Length") or 0)
    total_bytes = (resume_offset + content_len) if content_len > 0 else 0
    downloaded = resume_offset
    mode = "ab" if resume_offset > 0 else "wb"
    session_bytes = 0

    with open(part_path, mode) as out_f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            if cancel_event.is_set():
                raise RuntimeError("Download cancelled by user.")
            if not chunk:
                continue
            out_f.write(chunk)
            chunk_len = len(chunk)
            downloaded += chunk_len
            session_bytes += chunk_len
            elapsed = max(0.05, time.perf_counter() - t_start)
            speed_mbps = (session_bytes / (1024 * 1024)) / elapsed
            pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
            with MODEL_DOWNLOAD_STATE._lock:
                MODEL_DOWNLOAD_STATE.stage = stage_label
                MODEL_DOWNLOAD_STATE.downloaded_bytes = downloaded
                MODEL_DOWNLOAD_STATE.total_bytes = total_bytes
                MODEL_DOWNLOAD_STATE.progress_pct = pct
                MODEL_DOWNLOAD_STATE.speed_mbps = speed_mbps

    os.replace(part_path, target)
    return target


def _run_model_download_worker(
    repo_id: str,
    filename: str,
    save_as: str,
    mmproj_filename: str,
    mmproj_save_as: str,
) -> None:
    try:
        _stream_hf_file(
            repo_id=repo_id,
            remote_filename=filename,
            local_filename=save_as,
            stage_label=f"Downloading {save_as}",
            cancel_event=MODEL_DOWNLOAD_STATE.cancel_event,
        )
        if mmproj_filename and mmproj_save_as:
            _stream_hf_file(
                repo_id=repo_id,
                remote_filename=mmproj_filename,
                local_filename=mmproj_save_as,
                stage_label=f"Downloading vision projector {mmproj_save_as}",
                cancel_event=MODEL_DOWNLOAD_STATE.cancel_event,
            )
        with MODEL_DOWNLOAD_STATE._lock:
            MODEL_DOWNLOAD_STATE.running = False
            MODEL_DOWNLOAD_STATE.stage = "Complete"
            MODEL_DOWNLOAD_STATE.progress_pct = 100.0
            MODEL_DOWNLOAD_STATE.error = None
    except Exception as exc:
        log.warning(f"Model download stopped: {exc}")
        with MODEL_DOWNLOAD_STATE._lock:
            MODEL_DOWNLOAD_STATE.running = False
            MODEL_DOWNLOAD_STATE.stage = "Cancelled" if MODEL_DOWNLOAD_STATE.cancel_event.is_set() else "Error"
            MODEL_DOWNLOAD_STATE.error = str(exc)


# =====================================================================
# API SCHEMAS
# =====================================================================

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: StrictStr
    content: Any = None
    name: Optional[StrictStr] = None
    tool_call_id: Optional[StrictStr] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    schema_definition: Dict[str, Any] = Field(alias="schema")
    description: Optional[StrictStr] = None
    strict: Optional[bool] = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaSpec] = None

    @model_validator(mode="after")
    def require_json_schema_payload(self):
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("response_format.json_schema is required for json_schema output")
        return self


class ToolFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: StrictStr = Field(min_length=1, max_length=128)
    description: Optional[StrictStr] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"]
    function: ToolFunctionSpec


ChatFormat = Literal[
    "auto", "llama-2", "llama-3", "alpaca", "qwen", "vicuna", "oasst_llama",
    "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy",
    "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion",
    "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma",
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: StrictStr
    messages: List[Message]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[StrictInt] = Field(default=None, ge=0, le=200)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    max_tokens: Optional[StrictInt] = Field(default=None, ge=1, le=32768)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[StrictInt] = None
    use_pcp: Optional[bool] = None
    pcp_top_k: Optional[int] = Field(default=None, ge=1, le=12)
    show_reasoning: Optional[bool] = None
    use_web: Optional[bool] = None
    web_cite_sources: Optional[bool] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = None
    persona_id: Optional[StrictStr] = Field(default=None, max_length=120)


class LoadRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)
    chat_format: ChatFormat = "auto"
    mmproj_model: StrictStr = Field(default="auto", max_length=255)


class PreflightRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)
    mmproj_model: StrictStr = Field(default="auto", max_length=255)


class LoaderSettings(BaseModel):
    model: StrictStr = Field(default=DEFAULT_SETTINGS["model"], min_length=1, max_length=255)
    mmproj_model: StrictStr = Field(default="auto", max_length=255)
    chat_format: ChatFormat = "auto"
    system_prompt: StrictStr = Field(default=DEFAULT_SETTINGS["system_prompt"], max_length=12000)
    pcp_enabled: bool = True
    pcp_top_k: StrictInt = Field(default=4, ge=1, le=12)
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=64)
    blocked_phrases: StrictStr = Field(default=DEFAULT_SETTINGS["blocked_phrases"], max_length=8000)
    protected_words: StrictStr = Field(default=DEFAULT_SETTINGS["protected_words"], max_length=8000)
    show_reasoning: bool = False
    web_enabled: bool = False
    web_provider: Literal["auto", "searxng", "brave", "tavily"] = "auto"
    web_searxng_url: StrictStr = Field(default="", max_length=500)
    web_api_key: StrictStr = Field(default="", max_length=500)
    web_search_mode: Literal["always", "smart", "url_only"] = "smart"
    web_cite_sources: bool = True
    auto_continuity_summary: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    top_k: StrictInt = Field(default=40, ge=0, le=200)
    repeat_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[StrictInt] = None
    max_tokens: StrictInt = Field(default=512, ge=1, le=8192)
    stop_sequences: StrictStr = Field(default="", max_length=2000)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)


class LocalPathIngestRequest(BaseModel):
    paths: List[str] = Field(min_length=1)
    blocked_phrases: Optional[str] = None
    protected_words: Optional[str] = None
    replace_existing: bool = False
    persona_id: Optional[str] = None
    persona_name: Optional[str] = None
    user_name: Optional[str] = None
    never_forget: Optional[str] = None
    persona_notes: Optional[str] = None
    tts_voice: Optional[str] = None


class PersonaCreateRequest(BaseModel):
    name: StrictStr = Field(min_length=1, max_length=120)
    user_name: StrictStr = Field(default="", max_length=120)
    never_forget: StrictStr = Field(default="", max_length=8000)
    notes: StrictStr = Field(default="", max_length=8000)
    continuity_notes: StrictStr = Field(default="", max_length=8000)
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=64)
    preferred_model: StrictStr = Field(default="", max_length=255)
    protected_words: StrictStr = Field(default="", max_length=8000)
    blocked_phrases: Optional[StrictStr] = Field(default=None, max_length=8000)
    activate: bool = True


class PersonaUpdateRequest(BaseModel):
    name: Optional[StrictStr] = Field(default=None, min_length=1, max_length=120)
    user_name: Optional[StrictStr] = Field(default=None, max_length=120)
    never_forget: Optional[StrictStr] = Field(default=None, max_length=8000)
    notes: Optional[StrictStr] = Field(default=None, max_length=8000)
    continuity_notes: Optional[StrictStr] = Field(default=None, max_length=8000)
    tts_voice: Optional[StrictStr] = Field(default=None, min_length=1, max_length=64)
    preferred_model: Optional[StrictStr] = Field(default=None, max_length=255)
    protected_words: Optional[StrictStr] = Field(default=None, max_length=8000)
    blocked_phrases: Optional[StrictStr] = Field(default=None, max_length=8000)


class SaveSessionRequest(BaseModel):
    messages: List[Message] = Field(min_length=1)
    capture_continuity: Optional[bool] = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[StrictStr, List[Dict[str, Any]]]


class ChatHistoryRequest(BaseModel):
    messages: List[HistoryTurn]


class FactCreateRequest(BaseModel):
    id: Optional[StrictStr] = Field(default=None, max_length=120)
    subject: StrictStr = Field(min_length=1, max_length=120)
    aliases: List[StrictStr] = Field(default_factory=list)
    category: StrictStr = Field(default="general", max_length=40)
    fact: StrictStr = Field(min_length=1, max_length=1200)
    pinned: bool = True


class ThreadSaveRequest(BaseModel):
    thread_id: Optional[StrictStr] = Field(default=None, max_length=120)
    title: StrictStr = Field(default="", max_length=120)
    messages: List[HistoryTurn] = Field(min_length=1)


class VolumeSaveRequest(BaseModel):
    content: StrictStr = Field(max_length=5000000)


class ModelDownloadRequest(BaseModel):
    catalog_id: Optional[StrictStr] = Field(default=None, max_length=120)
    repo_id: Optional[StrictStr] = Field(default=None, max_length=255)
    filename: Optional[StrictStr] = Field(default=None, max_length=255)
    save_as: Optional[StrictStr] = Field(default=None, max_length=255)
    mmproj_filename: Optional[StrictStr] = Field(default=None, max_length=255)
    mmproj_save_as: Optional[StrictStr] = Field(default=None, max_length=255)


class TTSRequest(BaseModel):
    text: StrictStr = Field(min_length=1, max_length=20000)
    voice: StrictStr = Field(default="silas", max_length=64)
    play_on_host: Optional[bool] = None


_SETTINGS_LOCK = threading.Lock()


def load_settings() -> dict:
    try:
        with _SETTINGS_LOCK:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        merged = {**DEFAULT_SETTINGS, **data}
        validated = LoaderSettings.model_validate(merged).model_dump()
        validated["system_prompt"] = strip_builtin_system_prompt(validated.get("system_prompt", ""))
        return validated
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except Exception as exc:
        log.warning("Ignoring invalid settings file: %s", exc)
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: LoaderSettings) -> dict:
    data = settings.model_dump()
    data["system_prompt"] = strip_builtin_system_prompt(data.get("system_prompt", ""))
    temporary_path = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
    with _SETTINGS_LOCK:
        temporary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_path, SETTINGS_PATH)
    return data


# =====================================================================
# BACKGROUND ARCHIVE INGESTION WORKER
# =====================================================================

def _ingest_progress_callback(msg: str, frac: float) -> None:
    if INGEST_STATE.cancel_event.is_set():
        raise IngestCancelledError("Import cancelled by user.")
    INGEST_STATE.stage = msg
    INGEST_STATE.progress = round(min(1.0, max(0.0, frac)), 3)


def _delete_persona_entry(persona_id: str, delete_files: bool = False) -> dict:
    with _PERSONA_LOCK:
        reg = _load_persona_registry()
        personas = list(reg.get("personas") or [])
        target = next((p for p in personas if p["id"] == persona_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
        remaining = [p for p in personas if p["id"] != persona_id]
        was_active = reg.get("active_persona_id") == persona_id
        if was_active:
            reg["active_persona_id"] = remaining[0]["id"] if remaining else None
        reg["personas"] = remaining
        _save_persona_registry(reg)

    target_dir = _resolve_persona_folder(target.get("folder", "."))
    if delete_files and target.get("folder") not in ("", ".") and target_dir.is_dir():
        shutil.rmtree(target_dir, ignore_errors=True)
    return {"deleted": persona_id, "active_persona_id": reg.get("active_persona_id")}


def _prepare_ingest_target_persona(
    persona_id: Optional[str],
    persona_name: Optional[str],
    user_name: Optional[str],
    never_forget: Optional[str],
    persona_notes: Optional[str],
    tts_voice: Optional[str],
    bp_raw: str,
    pw_raw: str,
) -> Tuple[Path, Optional[str], Optional[str], bool]:
    """Resolve or create the destination persona folder for an archive import.

    Returns (target_dir, target_persona_id, prev_active_id, created_new).
    """
    prev_active = _get_active_persona()
    prev_active_id = prev_active["id"] if prev_active else None
    wants_new = persona_id == "__new__" or (bool((persona_name or "").strip()) and not persona_id)

    if wants_new:
        created = _create_persona_entry(
            name=(persona_name or "").strip() or "New AI",
            user_name=user_name or "",
            never_forget=never_forget or "",
            notes=persona_notes or "",
            tts_voice=(tts_voice or "silas").strip() or "silas",
            protected_words=pw_raw,
            blocked_phrases=bp_raw,
            source_type="importing",
            activate=False,
        )
        _activate_persona(created["id"])
        target_dir = _resolve_persona_folder(created["folder"])
        return target_dir, created["id"], prev_active_id, True

    if persona_id:
        updates: Dict[str, Any] = {}
        if persona_name is not None and persona_name.strip():
            updates["name"] = persona_name.strip()
        if user_name is not None:
            updates["user_name"] = user_name
        if never_forget is not None:
            updates["never_forget"] = never_forget
        if persona_notes is not None:
            updates["notes"] = persona_notes
        if tts_voice is not None and tts_voice.strip():
            updates["tts_voice"] = tts_voice.strip()
        if pw_raw:
            updates["protected_words"] = pw_raw
        if bp_raw:
            updates["blocked_phrases"] = bp_raw
        if updates:
            _update_persona_entry(persona_id, updates)
        _activate_persona(persona_id)
        target = _get_persona_by_id(persona_id)
        assert target is not None
        return _resolve_persona_folder(target["folder"]), persona_id, prev_active_id, False

    active = _get_active_persona()
    if active:
        return _resolve_persona_folder(active["folder"]), active["id"], prev_active_id, False
    return PCP_DATA_DIR, None, prev_active_id, False


def _start_background_ingest(
    file_items: Optional[List[Tuple[str, bytes]]],
    blocked_phrases: List[str],
    protected_words: List[str],
    replace_existing: bool = False,
    local_paths: Optional[List[Path]] = None,
    target_dir: Optional[Path] = None,
    target_persona_id: Optional[str] = None,
    prev_persona_id: Optional[str] = None,
    cleanup_new_persona_on_fail: bool = False,
) -> None:
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")

    INGEST_STATE.cancel_event.clear()
    INGEST_STATE.cancelled = False
    INGEST_STATE.running = True
    INGEST_STATE.stage = "Starting archive extraction..."
    INGEST_STATE.progress = 0.02
    INGEST_STATE.error = None
    INGEST_STATE.last_result = None

    resolved_dir = Path(target_dir) if target_dir is not None else _get_active_pcp_dir()

    def _worker():
        global PCP_INDEX
        try:
            loaded_items: List[Tuple[str, bytes]] = list(file_items or [])
            if local_paths:
                total_paths = len(local_paths)
                for idx, lp in enumerate(local_paths, start=1):
                    _ingest_progress_callback(
                        f"Reading {lp.name} ({idx}/{total_paths})...",
                        0.02 + 0.03 * (idx / max(1, total_paths)),
                    )
                    loaded_items.append((lp.name, lp.read_bytes()))

            summary = ingest_files(
                file_items=loaded_items,
                output_dir=resolved_dir,
                blocked_phrases=blocked_phrases,
                protected_words=protected_words,
                replace_existing=replace_existing,
                progress_callback=_ingest_progress_callback,
            )
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            if PCP_INDEX is None or PCP_INDEX.data_dir.resolve() != resolved_dir.resolve():
                PCP_INDEX = PCPIndex(
                    resolved_dir,
                    cache_dir=resolved_dir,
                    model_dir=bge_dir,
                    progress_callback=_ingest_progress_callback,
                )
            else:
                PCP_INDEX.reload(progress_callback=_ingest_progress_callback)
            summary["passages_indexed"] = PCP_INDEX.passage_count if PCP_INDEX else 0
            if target_persona_id:
                summary["persona_id"] = target_persona_id
                try:
                    _update_persona_entry(
                        target_persona_id,
                        {
                            "source_type": summary.get("source_type", "custom"),
                            "passage_count": summary["passages_indexed"],
                        },
                    )
                except Exception as exc:
                    log.warning(f"Could not update persona metadata after ingest: {exc}")
            INGEST_STATE.last_result = summary
            src_label = summary.get("source_label")
            prefix = f"Complete ({src_label})" if src_label else "Complete"
            INGEST_STATE.stage = (
                f"{prefix}: {summary['volumes_written']} PCP volume(s), "
                f"{summary['passages_indexed']:,} passages ready."
            )
            INGEST_STATE.progress = 1.0
            log.info(f"Archive ingestion complete: {summary}")
        except IngestCancelledError:
            INGEST_STATE.cancelled = True
            INGEST_STATE.error = None
            INGEST_STATE.stage = "Import cancelled. Progress saved — click RE-INDEX anytime to resume, or close."
            log.info("Archive ingestion cancelled by user.")
        except Exception as exc:
            log.error(f"Archive ingestion failed: {exc}")
            log.error(traceback.format_exc())
            if cleanup_new_persona_on_fail and target_persona_id and not any(resolved_dir.glob("ghost_pcp_*.md")):
                try:
                    _delete_persona_entry(target_persona_id, delete_files=True)
                    if prev_persona_id:
                        _activate_persona(prev_persona_id)
                except Exception:
                    pass
            INGEST_STATE.error = str(exc)
            INGEST_STATE.stage = f"Failed: {exc}"
        finally:
            INGEST_STATE.running = False

    threading.Thread(target=_worker, daemon=True).start()


# =====================================================================
# FASTAPI APPLICATION & ROUTES
# =====================================================================

app = FastAPI(title="GHostPCP", version="1.3.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def loader_page():
    if not HTML_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"UI not found: {HTML_PATH}")
    return FileResponse(HTML_PATH, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/assets/{filename}")
def loader_asset(filename: str):
    safe_name = Path(filename).name
    path = ASSET_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(path)


@app.get("/v1/settings", dependencies=[Depends(verify_access)])
def get_settings():
    return load_settings()


@app.post("/v1/settings", dependencies=[Depends(verify_access)])
def update_settings(settings: LoaderSettings):
    return save_settings(settings)


@app.get("/v1/models", dependencies=[Depends(verify_access)])
def list_models():
    models = scan_models()
    mmproj_files = scan_mmproj_files()
    return {"object": "list", "data": models, "mmproj_files": mmproj_files}


@app.get("/v1/models/catalog", dependencies=[Depends(verify_access)])
def models_catalog():
    return {
        "catalog": _MODEL_CATALOG,
        "local_models": [m["id"] for m in scan_models()],
        "local_mmprojs": [m["id"] for m in scan_mmproj_files()],
    }


@app.get("/v1/models/hf-files", dependencies=[Depends(verify_access)])
def models_hf_files(repo_id: str):
    return _list_hf_repo_ggufs(repo_id)


@app.get("/v1/models/download/status", dependencies=[Depends(verify_access)])
def models_download_status():
    return MODEL_DOWNLOAD_STATE.snapshot()


@app.post("/v1/models/download", dependencies=[Depends(verify_access)])
def models_download_start(req: ModelDownloadRequest):
    if req.catalog_id:
        preset = next((c for c in _MODEL_CATALOG if c["id"] == req.catalog_id), None)
        if preset is None:
            raise HTTPException(status_code=404, detail=f"Catalog preset '{req.catalog_id}' not found.")
        repo_id = preset["repo_id"]
        filename = preset["filename"]
        save_as = preset["save_as"]
        mmproj_filename = preset.get("mmproj_filename") or ""
        mmproj_save_as = preset.get("mmproj_save_as") or ""
    else:
        if not req.repo_id or not req.filename:
            raise HTTPException(status_code=400, detail="Provide either catalog_id or repo_id + filename.")
        repo_id = _validate_hf_repo_id(req.repo_id)
        filename = str(req.filename).strip()
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            raise HTTPException(status_code=400, detail="Invalid remote filename.")
        save_as = _safe_local_model_name(req.save_as or Path(filename).name)
        mmproj_filename = str(req.mmproj_filename or "").strip()
        if mmproj_filename and (".." in mmproj_filename or mmproj_filename.startswith("/") or mmproj_filename.startswith("\\")):
            raise HTTPException(status_code=400, detail="Invalid remote mmproj filename.")
        mmproj_save_as = (
            _safe_local_model_name(req.mmproj_save_as or Path(mmproj_filename).name)
            if mmproj_filename
            else ""
        )

    with MODEL_DOWNLOAD_STATE._lock:
        if MODEL_DOWNLOAD_STATE.running:
            raise HTTPException(status_code=409, detail="A model download is already in progress.")
        MODEL_DOWNLOAD_STATE.cancel_event.clear()
        MODEL_DOWNLOAD_STATE.running = True
        MODEL_DOWNLOAD_STATE.stage = f"Connecting to {repo_id}..."
        MODEL_DOWNLOAD_STATE.repo_id = repo_id
        MODEL_DOWNLOAD_STATE.filename = save_as
        MODEL_DOWNLOAD_STATE.mmproj_filename = mmproj_save_as
        MODEL_DOWNLOAD_STATE.downloaded_bytes = 0
        MODEL_DOWNLOAD_STATE.total_bytes = 0
        MODEL_DOWNLOAD_STATE.progress_pct = 0.0
        MODEL_DOWNLOAD_STATE.speed_mbps = 0.0
        MODEL_DOWNLOAD_STATE.error = None

    threading.Thread(
        target=_run_model_download_worker,
        args=(repo_id, filename, save_as, mmproj_filename, mmproj_save_as),
        daemon=True,
    ).start()
    return MODEL_DOWNLOAD_STATE.snapshot()


@app.post("/v1/models/download/cancel", dependencies=[Depends(verify_access)])
def models_download_cancel():
    MODEL_DOWNLOAD_STATE.cancel_event.set()
    return {"status": "cancelling" if MODEL_DOWNLOAD_STATE.running else "idle"}


@app.post("/v1/preflight", dependencies=[Depends(verify_access)])
def preflight_endpoint(req: PreflightRequest):
    if STATE.locked:
        raise HTTPException(status_code=409, detail="A model change is in progress.")
    plan = preflight_model(req.model, req.n_ctx, mmproj_model=req.mmproj_model)
    if STATE.loaded_model:
        plan["warnings"].append(
            "A model is currently resident; this preview uses current free VRAM. "
            "Loading will unload it and recalculate before allocation."
        )
    return plan


@app.post("/v1/load", dependencies=[Depends(verify_access)])
def load_endpoint(req: LoadRequest):
    preflight = load_model(req.model, req.n_ctx, req.chat_format, mmproj_model=req.mmproj_model)
    return {
        "status": "loaded",
        "model": req.model,
        "load_seconds": round(STATE.load_seconds or 0.0, 2),
        "gpu_layers": STATE.loaded_gpu_layers,
        "context": req.n_ctx,
        "kv_cache_type": preflight["kv_cache_type"],
        "chat_format": req.chat_format,
        "mmproj_model": preflight.get("mmproj_model"),
        "vision_capable": bool(preflight.get("vision_capable")),
        "preflight": preflight,
    }


@app.post("/v1/unload", dependencies=[Depends(verify_access)])
def unload_endpoint():
    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="A model change or generation is in progress.")
        STATE.locked = True
    try:
        unload_current_model()
    finally:
        with STATE._lease:
            STATE.locked = False
    return {"status": "unloaded"}


@app.post("/v1/reset", dependencies=[Depends(verify_access)])
def reset_endpoint():
    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="Wait for the current model operation to finish.")
        if STATE.llm is None:
            raise HTTPException(status_code=409, detail="No model is loaded.")
        STATE.locked = True
        llm = STATE.llm
        model = STATE.loaded_model
    try:
        llm.reset()
        return {"status": "reset", "model": model}
    finally:
        with STATE._lease:
            STATE.locked = False


@app.get("/v1/status", dependencies=[Depends(verify_access)])
def status(request: Request):
    pcp_ready = bool(PCP_INDEX and PCP_INDEX.ready)
    active_persona = _get_active_persona()
    settings = load_settings()
    web_enabled = bool(settings.get("web_enabled", False))
    stt_ready = bool(
        _STT_ENGINE is not None
        or _has_local_stt_files(APP_DIR / "stt_model")
        or _has_local_stt_files(RESOURCE_DIR / "stt_model")
    )
    return {
        "engine": "GHostPCP",
        "backend": BACKEND_NAME,
        "resident_model": STATE.loaded_model,
        "mmproj_model": STATE.loaded_mmproj,
        "vision_capable": STATE.loaded_vision_capable,
        "n_gpu_layers": STATE.loaded_gpu_layers,
        "n_ctx": STATE.loaded_context_size,
        "kv_cache_type": STATE.loaded_kv_cache_type,
        "chat_format": STATE.loaded_chat_format,
        "loading": STATE.locked,
        "generating": STATE.busy,
        "generating_stream": STATE.streaming,
        "abort_requested": STATE.abort_requested,
        "generation_id": STATE.generation_id,
        "models_available": len(scan_models()),
        "pcp_ready": pcp_ready,
        "pcp_volumes": PCP_INDEX.file_count if pcp_ready and PCP_INDEX else 0,
        "pcp_passages": PCP_INDEX.passage_count if pcp_ready and PCP_INDEX else 0,
        "ingesting": INGEST_STATE.running,
        "downloading_model": MODEL_DOWNLOAD_STATE.running,
        "web_enabled": web_enabled,
        "stt_ready": stt_ready,
        "recording_mic": _AUDIO_REC.recording,
        "is_local_client": _is_local_loopback_request(request),
        "active_persona": (
            {"id": active_persona["id"], "name": active_persona["name"], "user_name": active_persona.get("user_name", "")}
            if active_persona
            else None
        ),
        "offline": not web_enabled,
    }


# =====================================================================
# PERSONA PROFILES & ISOLATED FOLDER ROUTES
# =====================================================================

@app.get("/v1/personas", dependencies=[Depends(verify_access)])
def list_personas():
    reg = _load_persona_registry()
    active_id = reg.get("active_persona_id")
    summaries = [_persona_summary(p, active_id) for p in (reg.get("personas") or [])]
    return {"personas": summaries, "active_persona_id": active_id}


@app.post("/v1/personas", dependencies=[Depends(verify_access)])
def create_persona(req: PersonaCreateRequest):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before creating an AI persona.")
    settings = load_settings()
    bp_raw = req.blocked_phrases if req.blocked_phrases is not None else settings.get("blocked_phrases", "")
    created = _create_persona_entry(
        name=req.name,
        user_name=req.user_name,
        never_forget=req.never_forget,
        notes=req.notes,
        continuity_notes=req.continuity_notes,
        tts_voice=req.tts_voice,
        preferred_model=req.preferred_model,
        protected_words=req.protected_words,
        blocked_phrases=bp_raw,
        source_type="custom",
        activate=False,
    )
    if req.activate:
        activated = _activate_persona(created["id"])
        return {"persona": activated["persona"], "messages": activated["messages"]}
    reg = _load_persona_registry()
    return {"persona": _persona_summary(created, reg.get("active_persona_id")), "messages": list(STATE.conversation)}


@app.put("/v1/personas/{persona_id}", dependencies=[Depends(verify_access)])
def update_persona(persona_id: str, req: PersonaUpdateRequest):
    reg = _load_persona_registry()
    if _get_persona_by_id(persona_id, reg) is None:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
    # Ensure legacy default is persisted if user edits it for the first time
    if not _personas_registry_path().is_file() and reg.get("personas"):
        _save_persona_registry(reg)
    updates = req.model_dump(exclude_none=True)
    updated = _update_persona_entry(persona_id, updates)
    updated_reg = _load_persona_registry()
    return {"persona": _persona_summary(updated, updated_reg.get("active_persona_id"))}


@app.post("/v1/personas/{persona_id}/select", dependencies=[Depends(verify_access)])
def select_persona(persona_id: str):
    reg = _load_persona_registry()
    if _get_persona_by_id(persona_id, reg) is None:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
    if not _personas_registry_path().is_file() and reg.get("personas"):
        _save_persona_registry(reg)
    result = _activate_persona(persona_id)
    return result


@app.delete("/v1/personas/{persona_id}", dependencies=[Depends(verify_access)])
def delete_persona(persona_id: str, delete_files: bool = False):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before deleting an AI persona.")
    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="Wait for current model operation to finish before deleting an AI persona.")
    res = _delete_persona_entry(persona_id, delete_files=delete_files)
    next_id = res.get("active_persona_id")
    if next_id:
        _activate_persona(next_id)
    else:
        global PCP_INDEX
        PCP_INDEX = None
        with STATE._history_lock:
            STATE.conversation.clear()
    return res


def _persona_dir_or_404(persona_id: str) -> Tuple[dict, Path]:
    persona = _get_persona_by_id(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
    return persona, _resolve_persona_folder(persona.get("folder", "."))


@app.get("/v1/personas/{persona_id}/facts", dependencies=[Depends(verify_access)])
def list_persona_facts_endpoint(persona_id: str):
    persona, p_dir = _persona_dir_or_404(persona_id)
    return {"persona_id": persona["id"], "facts": _load_persona_facts(p_dir)}


@app.post("/v1/personas/{persona_id}/facts", dependencies=[Depends(verify_access)])
def upsert_persona_fact_endpoint(persona_id: str, req: FactCreateRequest):
    _, p_dir = _persona_dir_or_404(persona_id)
    fact = _add_or_update_persona_fact(
        p_dir,
        subject=req.subject,
        fact=req.fact,
        aliases=req.aliases,
        category=req.category,
        pinned=req.pinned,
        fact_id=req.id,
    )
    return {"fact": fact, "facts": _load_persona_facts(p_dir)}


@app.delete("/v1/personas/{persona_id}/facts/{fact_id}", dependencies=[Depends(verify_access)])
def delete_persona_fact_endpoint(persona_id: str, fact_id: str):
    _, p_dir = _persona_dir_or_404(persona_id)
    if not _delete_persona_fact(p_dir, fact_id):
        raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.")
    return {"status": "deleted", "facts": _load_persona_facts(p_dir)}


@app.get("/v1/personas/{persona_id}/threads", dependencies=[Depends(verify_access)])
def list_persona_threads_endpoint(persona_id: str):
    persona, p_dir = _persona_dir_or_404(persona_id)
    return {"persona_id": persona["id"], "threads": _list_persona_threads(p_dir)}


@app.post("/v1/personas/{persona_id}/threads/save", dependencies=[Depends(verify_access)])
def save_persona_thread_endpoint(persona_id: str, req: ThreadSaveRequest):
    _, p_dir = _persona_dir_or_404(persona_id)
    msgs = [
        {"role": m.role, "content": _sanitize_message_content(m.content, keep_images=True)}
        for m in req.messages
    ]
    saved = _save_persona_thread(p_dir, msgs, title=req.title, thread_id=req.thread_id)
    return {"thread": saved, "threads": _list_persona_threads(p_dir)}


@app.post("/v1/personas/{persona_id}/threads/{thread_id}/load", dependencies=[Depends(verify_access)])
def load_persona_thread_endpoint(persona_id: str, thread_id: str):
    persona, p_dir = _persona_dir_or_404(persona_id)
    thread = _load_persona_thread(p_dir, thread_id)
    _save_persona_chat_history(p_dir, thread["messages"])
    active = _get_active_persona()
    if not active or active["id"] != persona_id:
        activated = _activate_persona(persona_id)
        return {
            "persona": activated["persona"],
            "thread": thread,
            "messages": activated["messages"],
        }
    with STATE._history_lock:
        STATE.conversation = list(thread["messages"])
    if STATE.llm is not None:
        try:
            STATE.llm.reset()
        except Exception:
            pass
    return {"persona": _persona_summary(persona, persona_id), "thread": thread, "messages": thread["messages"]}


@app.delete("/v1/personas/{persona_id}/threads/{thread_id}", dependencies=[Depends(verify_access)])
def delete_persona_thread_endpoint(persona_id: str, thread_id: str):
    _, p_dir = _persona_dir_or_404(persona_id)
    if not _delete_persona_thread(p_dir, thread_id):
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return {"status": "deleted", "threads": _list_persona_threads(p_dir)}


@app.get("/v1/personas/{persona_id}/export", dependencies=[Depends(verify_access)])
def export_persona_bundle(persona_id: str):
    persona, p_dir = _persona_dir_or_404(persona_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "format": "ghostpcp-persona-bundle",
            "version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": "1.3.0",
            "persona_name": persona.get("name", persona_id),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("persona.json", json.dumps(persona, indent=2, ensure_ascii=False))
        for fname in ("facts.json", "chat_history.json"):
            fpath = p_dir / fname
            if fpath.is_file():
                zf.write(fpath, arcname=fname)
        for vol in sorted(p_dir.glob("ghost_pcp_*.md")):
            if vol.is_file():
                zf.write(vol, arcname=vol.name)
        tdir = _threads_dir(p_dir)
        if tdir.is_dir():
            for tf in sorted(tdir.glob("*.json")):
                if tf.is_file():
                    zf.write(tf, arcname=f"threads/{tf.name}")

    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(persona.get("id") or "persona")).strip("-") or "persona"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="ghostpcp-persona-{safe_slug}.zip"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/v1/personas/import-bundle", dependencies=[Depends(verify_access)])
async def import_persona_bundle(file: UploadFile = File(...)):
    global PCP_INDEX
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before importing a backup.")
    raw_bytes = await file.read()
    if len(raw_bytes) > 512 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Backup zip exceeds 512 MB limit.")

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive.") from exc

    created_pid: Optional[str] = None
    target_dir: Optional[Path] = None
    try:
        with zf:
            infos = zf.infolist()
            total_uncompressed = sum(max(0, info.file_size) for info in infos)
            if total_uncompressed > 512 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Expanded backup exceeds 512 MB safety limit.")

            persona_meta: Dict[str, Any] = {}
            if any(info.filename == "persona.json" for info in infos):
                try:
                    persona_meta = json.loads(zf.read("persona.json").decode("utf-8"))
                except Exception:
                    persona_meta = {}

            fallback_name = Path(file.filename or "Imported Persona").stem.replace("ghostpcp-persona-", "")
            created = _create_persona_entry(
                name=str(persona_meta.get("name") or fallback_name or "Imported AI"),
                user_name=str(persona_meta.get("user_name") or ""),
                never_forget=str(persona_meta.get("never_forget") or ""),
                notes=str(persona_meta.get("notes") or ""),
                continuity_notes=str(persona_meta.get("continuity_notes") or ""),
                tts_voice=str(persona_meta.get("tts_voice") or "silas"),
                preferred_model=str(persona_meta.get("preferred_model") or ""),
                protected_words=str(persona_meta.get("protected_words") or ""),
                blocked_phrases=str(persona_meta.get("blocked_phrases") or ""),
                source_type=str(persona_meta.get("source_type") or "bundle"),
                activate=False,
            )
            created_pid = created["id"]
            target_dir = _resolve_persona_folder(created["folder"])
            resolved_target = target_dir.resolve()

            extracted_volumes = 0
            for info in infos:
                if info.is_dir():
                    continue
                member = info.filename.replace("\\", "/").lstrip("/")
                if not member or ".." in member.split("/") or ":" in member:
                    continue
                lower = member.lower()
                is_root_json = member in ("facts.json", "chat_history.json")
                is_vol = "/" not in member and lower.startswith("ghost_pcp_") and lower.endswith(".md")
                is_thread = member.startswith("threads/") and member.count("/") == 1 and lower.endswith(".json")
                if not (is_root_json or is_vol or is_thread):
                    continue
                dest = (resolved_target / member).resolve()
                if resolved_target not in dest.parents:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info.filename))
                if is_vol:
                    extracted_volumes += 1

        if extracted_volumes > 0 and target_dir is not None:
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            PCP_INDEX = PCPIndex(target_dir, cache_dir=target_dir, model_dir=bge_dir)
            _update_persona_entry(created_pid, {"passage_count": PCP_INDEX.passage_count})

        activated = _activate_persona(created_pid)
        return {
            "status": "imported",
            "persona": activated["persona"],
            "messages": activated["messages"],
            "volumes_imported": extracted_volumes,
        }
    except HTTPException:
        if created_pid:
            try:
                _delete_persona_entry(created_pid, delete_files=True)
            except Exception:
                pass
        raise
    except Exception as exc:
        if created_pid:
            try:
                _delete_persona_entry(created_pid, delete_files=True)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Persona backup import failed: {exc}") from exc


# =====================================================================
# PCP ARCHIVE INGESTION ROUTES
# =====================================================================

@app.get("/v1/pcp/status", dependencies=[Depends(verify_access)])
def pcp_status():
    pcp_ready = bool(PCP_INDEX and PCP_INDEX.ready)
    active_dir = _get_active_pcp_dir()
    active_persona = _get_active_persona()
    volume_files = sorted(p.name for p in active_dir.glob("ghost_pcp_*.md")) if active_dir.is_dir() else []
    return {
        "ready": pcp_ready,
        "volumes": PCP_INDEX.file_count if pcp_ready and PCP_INDEX else len(volume_files),
        "passages": PCP_INDEX.passage_count if pcp_ready and PCP_INDEX else 0,
        "volume_files": volume_files,
        "data_dir": str(active_dir),
        "active_persona": (
            _persona_summary(active_persona, active_persona["id"]) if active_persona else None
        ),
        "ingesting": INGEST_STATE.running,
        "cancelling": INGEST_STATE.running and INGEST_STATE.cancel_event.is_set(),
        "cancelled": INGEST_STATE.cancelled,
        "stage": INGEST_STATE.stage,
        "progress": INGEST_STATE.progress,
        "error": INGEST_STATE.error,
        "last_result": INGEST_STATE.last_result,
    }


@app.post("/v1/pcp/cancel", dependencies=[Depends(verify_access)])
def pcp_cancel():
    active = INGEST_STATE.request_cancel()
    return {"status": "cancelling" if active else "idle"}


@app.post("/v1/pcp/ingest", dependencies=[Depends(verify_access)])
async def pcp_ingest_upload(
    files: List[UploadFile] = File(...),
    blocked_phrases: Optional[str] = Form(default=None),
    protected_words: Optional[str] = Form(default=None),
    replace_existing: bool = Form(default=False),
    persona_id: Optional[str] = Form(default=None),
    persona_name: Optional[str] = Form(default=None),
    user_name: Optional[str] = Form(default=None),
    never_forget: Optional[str] = Form(default=None),
    persona_notes: Optional[str] = Form(default=None),
    tts_voice: Optional[str] = Form(default=None),
):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")
    settings = load_settings()
    bp_raw = blocked_phrases if blocked_phrases is not None else settings.get("blocked_phrases", "")
    pw_raw = protected_words if protected_words is not None else settings.get("protected_words", "")
    bp_list = [line.strip() for line in bp_raw.splitlines() if line.strip()]
    pw_list = [line.strip() for line in pw_raw.splitlines() if line.strip()]

    items: List[Tuple[str, bytes]] = []
    for up in files:
        content = await up.read()
        items.append((up.filename or "conversations.json", content))

    target_dir, target_pid, prev_pid, created_new = _prepare_ingest_target_persona(
        persona_id=persona_id,
        persona_name=persona_name,
        user_name=user_name,
        never_forget=never_forget,
        persona_notes=persona_notes,
        tts_voice=tts_voice,
        bp_raw=bp_raw,
        pw_raw=pw_raw,
    )

    _start_background_ingest(
        items,
        bp_list,
        pw_list,
        replace_existing=replace_existing,
        target_dir=target_dir,
        target_persona_id=target_pid,
        prev_persona_id=prev_pid,
        cleanup_new_persona_on_fail=created_new,
    )
    return {"status": "started", "files": len(items), "persona_id": target_pid}


@app.post("/v1/pcp/ingest-local", dependencies=[Depends(verify_access)])
def pcp_ingest_local(req: LocalPathIngestRequest):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")
    settings = load_settings()
    bp_raw = req.blocked_phrases if req.blocked_phrases is not None else settings.get("blocked_phrases", "")
    pw_raw = req.protected_words if req.protected_words is not None else settings.get("protected_words", "")
    bp_list = [line.strip() for line in bp_raw.splitlines() if line.strip()]
    pw_list = [line.strip() for line in pw_raw.splitlines() if line.strip()]

    discovered_paths: List[Path] = []
    for raw_path in req.paths:
        p = Path(raw_path.strip().strip('"'))
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {p}")
        discovered_paths.extend(discover_source_paths(p))

    if not discovered_paths:
        raise HTTPException(
            status_code=400,
            detail="No supported export or document files (.zip, .json, .jsonl, .html, .md, .txt, .pdf) found at path.",
        )

    target_dir, target_pid, prev_pid, created_new = _prepare_ingest_target_persona(
        persona_id=req.persona_id,
        persona_name=req.persona_name,
        user_name=req.user_name,
        never_forget=req.never_forget,
        persona_notes=req.persona_notes,
        tts_voice=req.tts_voice,
        bp_raw=bp_raw,
        pw_raw=pw_raw,
    )

    _start_background_ingest(
        file_items=None,
        blocked_phrases=bp_list,
        protected_words=pw_list,
        replace_existing=req.replace_existing,
        local_paths=discovered_paths,
        target_dir=target_dir,
        target_persona_id=target_pid,
        prev_persona_id=prev_pid,
        cleanup_new_persona_on_fail=created_new,
    )
    return {"status": "started", "files": len(discovered_paths), "persona_id": target_pid}


def _ensure_active_pcp_index() -> PCPIndex:
    global PCP_INDEX
    active_dir = _get_active_pcp_dir()
    bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
    if PCP_INDEX is None or PCP_INDEX.data_dir.resolve() != active_dir.resolve():
        PCP_INDEX = PCPIndex(active_dir, cache_dir=active_dir, model_dir=bge_dir)
    return PCP_INDEX


def _capture_session_continuity_note(raw_msgs: List[Dict[str, Any]]) -> str:
    """Create a brief continuity note summarizing session progress without failing saves."""
    lines: List[str] = []
    user_snippets: List[str] = []
    for item in raw_msgs:
        role = str(item.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        plain = _extract_message_plain_text(item.get("content")).strip()
        if not plain:
            continue
        compact = re.sub(r"\s+", " ", plain)
        if role == "user":
            user_snippets.append(compact[:140])
            lines.append(f"User: {compact[:420]}")
        else:
            lines.append(f"Assistant: {compact[:420]}")

    if not lines:
        return ""

    transcript = "\n".join(lines[-14:])[-3400:]
    if STATE.llm is not None:
        try:
            resp = STATE.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation into 1-2 short factual continuity sentences "
                            "(key decisions, personal details shared, ongoing threads, or next steps) "
                            "for future sessions. Plain text only, no headings."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=160,
                stream=False,
            )
            text = str(((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            text = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text[:420]
        except Exception as exc:
            log.warning(f"Continuity LLM note skipped cleanly: {exc}")

    if user_snippets:
        joined = " | ".join(user_snippets[-2:])
        return f"Discussed: {joined[:260]}"
    return ""


def _merge_continuity_notes(existing: str, new_note: str, max_entries: int = 10, max_chars: int = 2400) -> str:
    note = re.sub(r"\s+", " ", str(new_note or "")).strip()
    if not note:
        return str(existing or "").strip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"- [{stamp}] {note}"
    lines = [line.strip() for line in str(existing or "").splitlines() if line.strip()]
    lines.append(entry)
    lines = lines[-max_entries:]
    joined = "\n".join(lines)
    while len(joined) > max_chars and len(lines) > 1:
        lines.pop(0)
        joined = "\n".join(lines)
    return joined[:max_chars]


@app.post("/v1/pcp/save-session", dependencies=[Depends(verify_access)])
def pcp_save_session(req: SaveSessionRequest):
    global PCP_INDEX
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive indexing to finish before saving session.")
    with STATE._lease:
        if STATE.locked or STATE.busy:
            raise HTTPException(status_code=409, detail="Wait for current generation to finish before saving session.")
        STATE.locked = True
    try:
        active_dir = _get_active_pcp_dir()
        bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
        if PCP_INDEX is None or PCP_INDEX.data_dir.resolve() != active_dir.resolve():
            PCP_INDEX = PCPIndex(active_dir, cache_dir=active_dir, model_dir=bge_dir)

        raw_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        settings = load_settings()
        capture_cont = (
            req.capture_continuity
            if req.capture_continuity is not None
            else bool(settings.get("auto_continuity_summary", True))
        )
        continuity_note = _capture_session_continuity_note(raw_msgs) if capture_cont else ""

        try:
            result = PCP_INDEX.append_session(raw_msgs, continuity_note=continuity_note or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.error(f"Save session failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to save session: {exc}") from exc

        result["continuity_note"] = continuity_note
        with STATE._history_lock:
            STATE.conversation.clear()
        if _personas_registry_path().is_file():
            active_persona = _get_active_persona()
            if active_persona:
                _save_persona_chat_history(active_dir, [])
                updates: Dict[str, Any] = {"passage_count": PCP_INDEX.passage_count}
                if continuity_note:
                    updates["continuity_notes"] = _merge_continuity_notes(
                        active_persona.get("continuity_notes", ""),
                        continuity_note,
                    )
                _update_persona_entry(active_persona["id"], updates)

        if STATE.llm is not None:
            try:
                STATE.llm.reset()
            except Exception as exc:
                log.warning(f"LLM reset warning after save-session: {exc}")

        log.info(
            f"SESSION SAVED :: file={result['saved_to']} turns={result['turns_saved']} "
            f"passages_added={result['passages_added']} total_passages={result['total_passages']}"
        )
        return result
    finally:
        with STATE._lease:
            STATE.locked = False


@app.post("/v1/pcp/reindex", dependencies=[Depends(verify_access)])
def pcp_reindex():
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")

    active_dir = _get_active_pcp_dir()
    active_persona = _get_active_persona()
    INGEST_STATE.cancel_event.clear()
    INGEST_STATE.cancelled = False
    INGEST_STATE.running = True
    INGEST_STATE.stage = "Re-indexing ghost_pcp_*.md volumes..."
    INGEST_STATE.progress = 0.1
    INGEST_STATE.error = None

    def _worker():
        global PCP_INDEX
        try:
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            if PCP_INDEX is None or PCP_INDEX.data_dir.resolve() != active_dir.resolve():
                PCP_INDEX = PCPIndex(
                    active_dir,
                    cache_dir=active_dir,
                    model_dir=bge_dir,
                    progress_callback=_ingest_progress_callback,
                )
            else:
                PCP_INDEX.reload(progress_callback=_ingest_progress_callback)
            if active_persona and _personas_registry_path().is_file() and PCP_INDEX and PCP_INDEX.ready:
                _update_persona_entry(active_persona["id"], {"passage_count": PCP_INDEX.passage_count})
            INGEST_STATE.stage = (
                f"Indexed {PCP_INDEX.file_count} volume(s), {PCP_INDEX.passage_count:,} passages."
                if PCP_INDEX and PCP_INDEX.ready
                else f"No ghost_pcp_*.md files found in {active_dir}."
            )
            INGEST_STATE.progress = 1.0
        except IngestCancelledError:
            INGEST_STATE.cancelled = True
            INGEST_STATE.error = None
            INGEST_STATE.stage = "Import cancelled. Progress saved — click RE-INDEX anytime to resume, or close."
            log.info("Archive re-index cancelled by user.")
        except Exception as exc:
            INGEST_STATE.error = str(exc)
            INGEST_STATE.stage = f"Failed: {exc}"
        finally:
            INGEST_STATE.running = False

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.get("/v1/pcp/search", dependencies=[Depends(verify_access)])
def pcp_search_memories(q: str = "", limit: int = 8):
    query = str(q or "").strip()
    if not query:
        return {"query": "", "ready": bool(PCP_INDEX and PCP_INDEX.ready), "results": []}
    idx = _ensure_active_pcp_index()
    if not idx.ready:
        return {"query": query, "ready": False, "results": []}
    safe_limit = max(1, min(24, int(limit or 8)))
    results = idx.search_detailed(query, limit=safe_limit)
    return {"query": query, "ready": idx.ready, "results": results}


@app.get("/v1/pcp/volumes", dependencies=[Depends(verify_access)])
def pcp_list_volumes():
    idx = _ensure_active_pcp_index()
    return {
        "volumes": idx.list_volumes(),
        "passages": idx.passage_count if idx.ready else 0,
        "ready": idx.ready,
    }


@app.get("/v1/pcp/volumes/{filename}", dependencies=[Depends(verify_access)])
def pcp_read_volume(filename: str):
    idx = _ensure_active_pcp_index()
    try:
        safe_name = Path(filename).name
        return {"filename": safe_name, "content": idx.read_volume(filename)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/v1/pcp/volumes/{filename}", dependencies=[Depends(verify_access)])
def pcp_update_volume(filename: str, req: VolumeSaveRequest):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before editing a volume.")
    idx = _ensure_active_pcp_index()
    try:
        res = idx.save_volume(filename, req.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if _personas_registry_path().is_file():
        active = _get_active_persona()
        if active:
            _update_persona_entry(active["id"], {"passage_count": idx.passage_count})
    return res


@app.delete("/v1/pcp/volumes/{filename}", dependencies=[Depends(verify_access)])
def pcp_delete_volume(filename: str):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive processing to finish before deleting a volume.")
    idx = _ensure_active_pcp_index()
    try:
        res = idx.delete_volume(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if _personas_registry_path().is_file():
        active = _get_active_persona()
        if active:
            _update_persona_entry(active["id"], {"passage_count": idx.passage_count})
    return res


# =====================================================================
# PHONE & REMOTE TUNNEL ROUTES
# =====================================================================

@app.get("/v1/remote/status", dependencies=[Depends(verify_access)])
def remote_status():
    lan_ip = get_lan_ip()
    ts_ip = get_tailscale_ip()
    lan_url = f"http://{lan_ip}:{PORT}/#{PAIRING_TOKEN}" if lan_ip else None
    ts_url = f"http://{ts_ip}:{PORT}/#{PAIRING_TOKEN}" if ts_ip else None
    tunnel_running = TUNNEL_STATE.process is not None and TUNNEL_STATE.process.poll() is None
    tunnel_url = f"{TUNNEL_STATE.public_url}/#{PAIRING_TOKEN}" if (tunnel_running and TUNNEL_STATE.public_url) else None
    return {
        "port": PORT,
        "lan_ip": lan_ip,
        "lan_url": lan_url,
        "tailscale_ip": ts_ip,
        "tailscale_url": ts_url,
        "tunnel_running": tunnel_running,
        "tunnel_starting": TUNNEL_STATE.starting,
        "tunnel_url": tunnel_url,
        "tunnel_error": TUNNEL_STATE.error,
        "scanner_url": "https://wildemindpress.com/scanner.html",
    }


@app.post("/v1/remote/tunnel/start", dependencies=[Depends(verify_access)])
def remote_tunnel_start():
    start_cloudflare_tunnel()
    return remote_status()


@app.post("/v1/remote/tunnel/stop", dependencies=[Depends(verify_access)])
def remote_tunnel_stop():
    stop_cloudflare_tunnel()
    return remote_status()


@app.post("/v1/remote/rotate-token", dependencies=[Depends(verify_access)])
def remote_rotate_token():
    rotate_pairing_token()
    return remote_status()


@app.get("/v1/remote/qr.svg", dependencies=[Depends(verify_access)])
def remote_qr_svg(target: str = "tunnel"):
    info = remote_status()
    if target == "tunnel" and info["tunnel_url"]:
        url = info["tunnel_url"]
    elif target == "tailscale" and info["tailscale_url"]:
        url = info["tailscale_url"]
    elif target == "lan" and info["lan_url"]:
        url = info["lan_url"]
    elif info["tunnel_url"]:
        url = info["tunnel_url"]
    elif info["lan_url"]:
        url = info["lan_url"]
    else:
        url = f"http://127.0.0.1:{PORT}/#{PAIRING_TOKEN}"

    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    svg_text = buf.getvalue().decode("utf-8")
    svg_text = svg_text.replace('fill="#000000"', 'fill="#f1f0f5"')
    return Response(content=svg_text, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


# =====================================================================
# PIPER TTS READ-ALOUD ROUTES ("Silas" -> silas, "Kara" -> kara)
# =====================================================================

@app.get("/v1/tts/voices", dependencies=[Depends(verify_access)])
def tts_voices_endpoint():
    return {"voices": scan_piper_voices()}


@app.post("/v1/tts", dependencies=[Depends(verify_access)])
def tts_endpoint(req: TTSRequest, request: Request):
    is_local_loopback = _is_local_loopback_request(request)
    play_locally = req.play_on_host if req.play_on_host is not None else is_local_loopback

    try:
        wav_bytes = read_aloud(req.text, req.voice, play_locally=play_locally)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(f"TTS error: {exc}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-Played-On-Host": "true" if play_locally else "false",
        },
    )


@app.post("/v1/tts/stop", dependencies=[Depends(verify_access)])
def tts_stop_endpoint():
    stop_read_aloud()
    return {"status": "stopped"}


@app.post("/v1/audio/record/start", dependencies=[Depends(verify_access)])
def audio_record_start_endpoint(request: Request):
    if not _is_local_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Host microphone recording is restricted to local desktop sessions.",
        )
    try:
        device_name = _AUDIO_REC.start()
        return {"status": "recording", "device": device_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/record/stop", dependencies=[Depends(verify_access)])
def audio_record_stop_endpoint(request: Request):
    if not _is_local_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Host microphone recording is restricted to local desktop sessions.",
        )
    samples, duration_sec = _AUDIO_REC.stop_and_extract()
    if duration_sec <= 0.05:
        return {"text": "", "duration_sec": duration_sec, "engine": "moonshine-base-onnx"}
    try:
        text = _get_stt_engine().transcribe_float32(samples, sample_rate=_AUDIO_REC.SAMPLE_RATE)
        return {"text": text, "duration_sec": duration_sec, "engine": "moonshine-base-onnx"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}") from exc


@app.post("/v1/audio/record/cancel", dependencies=[Depends(verify_access)])
def audio_record_cancel_endpoint(request: Request):
    if not _is_local_loopback_request(request):
        raise HTTPException(status_code=403, detail="Host microphone recording is restricted to local desktop sessions.")
    _AUDIO_REC.cancel()
    return {"status": "cancelled"}


@app.post("/v1/audio/transcribe", dependencies=[Depends(verify_access)])
async def audio_transcribe_endpoint(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio clip is empty or exceeds 25 MB limit.")

    # Decode standard PCM/WAV/FLAC/OGG directly in memory using built-in soundfile + Moonshine ONNX
    try:
        samples, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        text = _get_stt_engine().transcribe_float32(samples, sample_rate=int(sr))
        return {"text": text, "engine": "moonshine-base-onnx"}
    except Exception as sf_exc:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode audio upload as WAV/OGG/FLAC ({sf_exc}). Upload WAV or use [ MIC ].",
            ) from exc

        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        if suffix.lower() not in (".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac"):
            suffix = ".webm"
        tmp_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(raw_bytes)
                tmp_path = tf.name
            model = WhisperModel("base.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(tmp_path)
            transcript = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
            return {"text": transcript, "engine": "faster-whisper"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass


# =====================================================================
# LIVE CONVERSATION SYNC (DESKTOP <-> PHONE)
# =====================================================================

@app.get("/v1/chat/history", dependencies=[Depends(verify_access)])
def get_chat_history():
    with STATE._history_lock:
        return {"messages": [dict(m) for m in STATE.conversation]}


@app.post("/v1/chat/history", dependencies=[Depends(verify_access)])
def set_chat_history(req: ChatHistoryRequest):
    cleaned = [
        {"role": m.role, "content": _sanitize_message_content(m.content, keep_images=True)}
        for m in req.messages
    ]
    with STATE._history_lock:
        STATE.conversation = cleaned
    if _personas_registry_path().is_file():
        active = _get_active_persona()
        if active:
            _save_persona_chat_history(_resolve_persona_folder(active.get("folder", ".")), cleaned)
    return {"status": "ok", "count": len(cleaned)}


def _condense_document_for_chat(safe_name: str, text: str, max_chars: int = 11000) -> str:
    """Keep huge attached files within chat context limits while indexing the full text into PCP."""
    total_len = len(text)
    if total_len <= max_chars:
        return text

    open_len, mid_len, close_len = 2200, 1550, 1800
    sections = [("--- OPENING SECTION ---", text[:open_len].rstrip())]
    cursor = open_len
    tail_start = total_len - close_len
    for label, frac in (
        ("--- EARLY SECTION ---", 0.20),
        ("--- MID-EARLY SECTION ---", 0.40),
        ("--- MID-LATE SECTION ---", 0.60),
        ("--- LATE SECTION ---", 0.80),
    ):
        start = max(cursor, min(tail_start - mid_len, int(total_len * frac) - (mid_len // 2)))
        snippet = text[start : start + mid_len].strip()
        if snippet:
            sections.append((label, snippet))
        cursor = start + mid_len
    sections.append(("--- CLOSING SECTION ---", text[tail_start:].lstrip()))

    body = "\n\n".join(f"{label}\n{snippet}" for label, snippet in sections)
    return (
        f"[Large document '{safe_name}' ({total_len:,} chars total) — automatically indexed into your PCP archive for full search. "
        f"Representative sections across the full document are shown below:]\n\n{body}"
    )


@app.post("/v1/chat/extract-file", dependencies=[Depends(verify_access)])
async def chat_extract_file(
    files: List[UploadFile] = File(...),
    save_to_archive: bool = Form(default=False),
):
    global PCP_INDEX
    if save_to_archive and INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive indexing to finish before saving to database.")

    extracted_files: List[Dict[str, Any]] = []
    for up in files:
        safe_name = Path(up.filename or "document.txt").name
        raw = await up.read()
        suffix = Path(safe_name).suffix.lower()
        try:
            if suffix == ".json":
                text_raw = raw.decode("utf-8-sig", errors="replace").strip()
                try:
                    data = json.loads(text_raw)
                    pairs = extract_pairs_from_data(data)
                except Exception:
                    pairs = []
                if pairs:
                    text = "\n\n".join(f"**USER:** {u}\n\n**AI:** {a}" for u, a in pairs).strip()
                else:
                    text = text_raw
            else:
                text = extract_text_from_document(safe_name, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read '{safe_name}': {exc}") from exc

        if not text:
            raise HTTPException(status_code=400, detail=f"File '{safe_name}' contained no readable text.")

        condensed = _condense_document_for_chat(safe_name, text)
        entry: Dict[str, Any] = {
            "name": safe_name,
            "text": condensed,
            "chat_text": condensed,
            "chars": len(text),
            "saved_to": None,
        }
        should_save = bool(save_to_archive) or (len(text) > 12000 and not INGEST_STATE.running)
        if should_save:
            active_dir = _get_active_pcp_dir()
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            if PCP_INDEX is None or PCP_INDEX.data_dir.resolve() != active_dir.resolve():
                PCP_INDEX = PCPIndex(active_dir, cache_dir=active_dir, model_dir=bge_dir)
            try:
                save_res = PCP_INDEX.append_document(safe_name, text)
                entry["saved_to"] = save_res.get("saved_to")
                if _personas_registry_path().is_file():
                    active_persona = _get_active_persona()
                    if active_persona:
                        _update_persona_entry(active_persona["id"], {"passage_count": PCP_INDEX.passage_count})
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to index '{safe_name}': {exc}") from exc
        extracted_files.append(entry)

    return {
        "files": extracted_files,
        "total_volumes": PCP_INDEX.file_count if PCP_INDEX else 0,
        "total_passages": PCP_INDEX.passage_count if PCP_INDEX else 0,
    }


# =====================================================================
# CHAT COMPLETIONS (WITH CITATION-FREE PCP GROUNDING)
# =====================================================================

def _build_effective_system_prompt(custom_prompt: str = "") -> str:
    custom = strip_builtin_system_prompt(custom_prompt)
    builtin = SYSTEM_PROMPT.strip()
    return f"{custom}\n\n{builtin}" if custom else builtin


def _condense_long_text(text: str, keep_head: int, keep_tail: int, marker: str) -> str:
    if len(text) <= keep_head + keep_tail + len(marker) + 20:
        return text
    head = text[:keep_head].rstrip()
    tail = text[-keep_tail:].lstrip() if keep_tail > 0 else ""
    return f"{head}{marker}{tail}".strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if STATE.llm is not None and hasattr(STATE.llm, "tokenize"):
        try:
            return len(STATE.llm.tokenize(text.encode("utf-8", errors="replace"), add_bos=False)) + 6
        except Exception:
            pass
    return max(1, (len(text) // 3) + 6)


def _count_image_parts(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1 for part in content
        if isinstance(part, dict) and str(part.get("type") or "").lower() == "image_url"
    )


def _estimate_message_tokens(msg: Dict[str, Any]) -> int:
    content = msg.get("content")
    if isinstance(content, list):
        plain = _extract_message_plain_text(content)
        img_count = _count_image_parts(content)
        return _estimate_tokens(plain) + (img_count * 256)
    return _estimate_tokens(str(content or ""))


def _condense_message_content(msg: Dict[str, Any], keep_head: int, keep_tail: int, marker: str) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        if len(content) > keep_head + keep_tail + len(marker) + 20:
            msg["content"] = _condense_long_text(content, keep_head, keep_tail, marker)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").lower() == "text":
                txt = str(part.get("text") or "")
                if len(txt) > keep_head + keep_tail + len(marker) + 20:
                    part["text"] = _condense_long_text(txt, keep_head, keep_tail, marker)


def _append_text_to_message_content(content: Any, addition: str) -> Any:
    if isinstance(content, list):
        updated = [dict(p) if isinstance(p, dict) else p for p in content]
        for idx in range(len(updated) - 1, -1, -1):
            part = updated[idx]
            if isinstance(part, dict) and str(part.get("type") or "").lower() == "text":
                part["text"] = str(part.get("text") or "").rstrip() + addition
                return updated
        updated.append({"type": "text", "text": addition.lstrip()})
        return updated
    return str(content or "").rstrip() + addition


def _strip_images_with_placeholder(content: Any, placeholder: str) -> Tuple[Any, int]:
    if not isinstance(content, list):
        return content, 0
    kept: List[Dict[str, Any]] = []
    removed = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        if str(part.get("type") or "").lower() == "image_url":
            removed += 1
            continue
        kept.append(dict(part))
    if not removed:
        return content, 0
    text_pieces = [str(p.get("text") or "") for p in kept if str(p.get("type") or "").lower() == "text"]
    joined = "\n".join(t for t in text_pieces if t.strip()).strip()
    note = f"{placeholder}" if removed == 1 else f"{placeholder} (x{removed})"
    return f"{joined}\n\n{note}".strip() if joined else note, removed


def _fit_messages_to_context(
    messages: List[Dict[str, Any]],
    n_ctx: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Trim older turns and oversize attachments so total prompt tokens stay inside n_ctx."""
    if not messages:
        return messages
    fitted = [dict(m) for m in messages]
    budget = max(1024, int(n_ctx) - int(max_tokens) - 256)

    def total_est() -> int:
        return sum(_estimate_message_tokens(m) for m in fitted)

    if total_est() <= budget:
        return fitted

    for idx in range(len(fitted) - 1):
        msg = fitted[idx]
        if msg.get("role") == "system":
            continue
        plain_len = len(_extract_message_plain_text(msg.get("content")))
        if plain_len > 2400:
            _condense_message_content(
                msg,
                1200,
                800,
                "\n\n[... earlier attachment/turn condensed ...]\n\n",
            )

    while len(fitted) > 2 and total_est() > budget:
        drop_idx = 1 if fitted[0].get("role") == "system" else 0
        fitted.pop(drop_idx)

    if total_est() > budget:
        other_tokens = sum(_estimate_message_tokens(m) for m in fitted[:-1])
        img_tokens = _count_image_parts(fitted[-1].get("content")) * 256
        remaining_tokens = max(512, budget - other_tokens - img_tokens)
        last_text = _extract_message_plain_text(fitted[-1].get("content"))
        if _estimate_tokens(last_text) > remaining_tokens:
            char_budget = max(800, int(remaining_tokens * 2.5))
            if isinstance(fitted[-1].get("content"), str):
                for _ in range(6):
                    candidate = _condense_long_text(
                        last_text,
                        max(300, int(char_budget * 0.5)),
                        max(300, int(char_budget * 0.45)),
                        "\n\n[... middle condensed to fit context window ...]\n\n",
                    )
                    if _estimate_tokens(candidate) <= remaining_tokens or len(candidate) >= len(last_text):
                        last_text = candidate
                        break
                    char_budget = int(char_budget * 0.75)
                    last_text = candidate
                fitted[-1]["content"] = last_text
            else:
                _condense_message_content(
                    fitted[-1],
                    max(300, int(char_budget * 0.5)),
                    max(300, int(char_budget * 0.45)),
                    "\n\n[... middle condensed to fit context window ...]\n\n",
                )

    return fitted


def _llm_web_query_rewriter(user_prompt: str) -> str:
    """Use the resident model to distill a short search query or return NONE."""
    llm = STATE.llm
    if llm is None or not hasattr(llm, "create_chat_completion"):
        return ""
    try:
        resp = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's message into a short web search query (2-7 words). "
                        "Reply ONLY with the search query, or reply NONE if no fresh external web lookup is needed."
                    ),
                },
                {"role": "user", "content": str(user_prompt or "")[:600]},
            ],
            temperature=0.0,
            top_p=0.9,
            max_tokens=28,
            stream=False,
        )
        raw = str(((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        raw = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"[\r\n\"'`]+", " ", raw).strip()
        if not raw or len(raw) > 120:
            return ""
        return raw
    except Exception:
        return ""


def _prepare_pcp_messages(req: ChatRequest) -> Tuple[List[Dict[str, Any]], bool]:
    settings = load_settings()
    use_pcp = req.use_pcp if req.use_pcp is not None else bool(settings.get("pcp_enabled", True))
    top_k = req.pcp_top_k or int(settings.get("pcp_top_k", 4))
    show_reasoning = req.show_reasoning if req.show_reasoning is not None else bool(settings.get("show_reasoning", False))
    use_web = req.use_web if req.use_web is not None else bool(settings.get("web_enabled", False))
    web_cite = req.web_cite_sources if req.web_cite_sources is not None else bool(settings.get("web_cite_sources", True))

    turn_warnings: List[str] = []
    turn_sources: List[Dict[str, Any]] = []

    raw_messages = [m.model_dump(exclude_none=True) for m in req.messages]
    if not raw_messages:
        STATE.last_turn_meta = {"sources": [], "warnings": [], "prompt_tokens_est": 0}
        return raw_messages, show_reasoning

    # Strictly sanitize any incoming image_url parts (only safe data:image/...;base64,... allowed)
    unsafe_stripped = 0
    vision_skipped = 0
    allow_vision_parts = bool(STATE.loaded_vision_capable or STATE.llm is None)
    for msg in raw_messages:
        content = msg.get("content")
        if isinstance(content, list):
            orig_imgs = _count_image_parts(content)
            sanitized = _sanitize_message_content(content, keep_images=True)
            kept_imgs = _count_image_parts(sanitized)
            if orig_imgs > kept_imgs:
                unsafe_stripped += (orig_imgs - kept_imgs)
            if not allow_vision_parts and kept_imgs > 0:
                sanitized, removed = _strip_images_with_placeholder(
                    sanitized,
                    "[Attached image skipped — active model has no vision projector (mmproj) loaded]",
                )
                vision_skipped += removed
            msg["content"] = sanitized

    if unsafe_stripped:
        turn_warnings.append(
            f"Stripped {unsafe_stripped} unsupported image reference(s); only local base64 image uploads are permitted."
        )
    if vision_skipped:
        turn_warnings.append(
            "Loaded model does not have a vision projector (mmproj) active; attached image(s) were skipped. "
            "Pair an mmproj file or choose a vision model in CONFIG."
        )

    # Locate the latest user message for semantic retrieval & web context
    latest_user_idx = None
    latest_user_text = ""
    for idx in range(len(raw_messages) - 1, -1, -1):
        if raw_messages[idx].get("role") == "user":
            plain = _extract_message_plain_text(raw_messages[idx].get("content")).strip()
            if plain or isinstance(raw_messages[idx].get("content"), list):
                latest_user_idx = idx
                latest_user_text = plain
                break

    # Include recent conversation turns when the latest reply is short (ignoring large attached file bodies)
    query_core = re.sub(
        r"\[Attached File:\s*([^\]]+)\][\s\S]*?(?:\[/Attached File\]|\Z)",
        r"Attached file: \1",
        latest_user_text,
    ).strip()
    retrieval_query = query_core or latest_user_text
    if retrieval_query and len(query_core) < 120 and latest_user_idx is not None:
        prior_turns: List[str] = []
        for msg in raw_messages[:latest_user_idx]:
            if msg.get("role") in ("user", "assistant"):
                plain_prior = _extract_message_plain_text(msg.get("content"))
                cleaned = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", plain_prior, flags=re.IGNORECASE).strip()
                if cleaned:
                    prior_turns.append(cleaned[:200])
        if prior_turns:
            retrieval_query = "\n".join(prior_turns[-3:] + [retrieval_query])

    configured_custom = strip_builtin_system_prompt(settings.get("system_prompt") or "")
    has_system = raw_messages[0].get("role") == "system"
    incoming_sys = str(raw_messages[0].get("content") or "").strip() if has_system else ""
    target_persona = None
    if req.persona_id:
        target_persona = _get_persona_by_id(req.persona_id)
    elif _personas_registry_path().is_file() and not (has_system and incoming_sys):
        target_persona = _get_active_persona()

    if has_system and "ARCHIVE EXCERPTS:" in incoming_sys:
        effective_sys = incoming_sys
    elif target_persona is not None:
        notes_source = (
            strip_builtin_system_prompt(incoming_sys)
            if (has_system and incoming_sys)
            else (target_persona.get("notes") or configured_custom)
        )
        persona_prefix = build_persona_prompt_prefix(
            name=target_persona.get("name", ""),
            user_name=target_persona.get("user_name", ""),
            never_forget=target_persona.get("never_forget", ""),
            notes=notes_source,
            continuity_notes=target_persona.get("continuity_notes", ""),
        )
        effective_sys = _build_effective_system_prompt(persona_prefix)
    else:
        custom_prefix = strip_builtin_system_prompt(incoming_sys) if (has_system and incoming_sys) else configured_custom
        effective_sys = _build_effective_system_prompt(custom_prefix)

    if has_system:
        raw_messages[0]["content"] = effective_sys
    else:
        raw_messages.insert(0, {"role": "system", "content": effective_sys})

    # Inject structured Core Facts & Relationships for the active/target persona
    facts_dir = (
        _resolve_persona_folder(target_persona["folder"])
        if target_persona is not None
        else _get_active_pcp_dir()
    )
    persona_facts = _load_persona_facts(facts_dir)
    if persona_facts and "CORE FACTS & RELATIONSHIPS:" not in raw_messages[0]["content"]:
        facts_block = format_core_facts_block(persona_facts, query=retrieval_query or latest_user_text, max_facts=12)
        if facts_block:
            raw_messages[0]["content"] = (
                raw_messages[0]["content"].rstrip()
                + "\n\nCORE FACTS & RELATIONSHIPS:\n"
                + facts_block
            )

    # Inject Hybrid PCP archive excerpts
    can_search_index = use_pcp and PCP_INDEX is not None and PCP_INDEX.ready and bool(retrieval_query)
    if can_search_index and target_persona is not None:
        try:
            can_search_index = PCP_INDEX.data_dir.resolve() == _resolve_persona_folder(target_persona["folder"]).resolve()
        except Exception:
            can_search_index = False
    if can_search_index and PCP_INDEX is not None:
        excerpts = PCP_INDEX.search(retrieval_query, limit=top_k)
        if excerpts and "ARCHIVE EXCERPTS:" not in raw_messages[0]["content"]:
            archive_block = "\n\n---\n\n".join(excerpts)
            raw_messages[0]["content"] = raw_messages[0]["content"].rstrip() + "\n\nARCHIVE EXCERPTS:\n" + archive_block

    # Opt-in live web search & URL reader (strictly off unless use_web=True)
    if use_web and query_core and "WEB RESULTS:" not in raw_messages[0]["content"]:
        web_mode = str(settings.get("web_search_mode") or "smart")
        rewriter = _llm_web_query_rewriter if web_mode == "smart" else None
        try:
            web_res = gather_web_context(
                user_text=query_core,
                provider=str(settings.get("web_provider") or "auto"),
                searxng_url=str(settings.get("web_searxng_url") or ""),
                api_key=str(settings.get("web_api_key") or ""),
                search_mode=web_mode,
                query_rewriter=rewriter,
                max_results=4,
            )
            turn_sources = web_res.get("sources") or []
            turn_warnings.extend(web_res.get("warnings") or [])
            ctx_block = str(web_res.get("context_block") or "").strip()
            if ctx_block:
                grounding_note = build_web_grounding_note(cite_sources=bool(web_cite))
                raw_messages[0]["content"] = (
                    raw_messages[0]["content"].rstrip()
                    + "\n\n"
                    + grounding_note
                    + "\n"
                    + ctx_block
                )
        except Exception as exc:
            log.warning(f"Web lookup error: {exc}")
            turn_warnings.append(f"Web lookup failed: {exc}")

    last_user_msg = next(
        (m for m in reversed(raw_messages) if m.get("role") == "user"),
        None,
    )
    if last_user_msg is not None:
        last_user_plain = _extract_message_plain_text(last_user_msg.get("content"))
        if "[Attached File:" in last_user_plain:
            reminder = (
                "[Respond directly to the attached text above by referencing its actual scenes, characters, and details. "
                "Do not repeat or paraphrase your previous reply.]"
            )
            if reminder not in last_user_plain:
                last_user_msg["content"] = _append_text_to_message_content(
                    last_user_msg.get("content"),
                    "\n\n" + reminder,
                )

    effective_n_ctx = int(STATE.loaded_context_size or settings.get("n_ctx") or 8192)
    effective_max_tokens = int(req.max_tokens or settings.get("max_tokens") or 512)
    raw_messages = _fit_messages_to_context(raw_messages, effective_n_ctx, effective_max_tokens)

    # For Qwen3 models when reasoning blocks are hidden, append /no_think to system + all user turns
    if (
        latest_user_idx is not None
        and STATE.loaded_model
        and "qwen3" in STATE.loaded_model.lower()
        and not show_reasoning
    ):
        for msg in raw_messages:
            if msg.get("role") in ("system", "user"):
                plain = _extract_message_plain_text(msg.get("content"))
                if "/no_think" not in plain:
                    msg["content"] = _append_text_to_message_content(msg.get("content"), "\n/no_think")

    prompt_tokens_est = sum(_estimate_message_tokens(m) for m in raw_messages)
    STATE.last_turn_meta = {
        "sources": turn_sources,
        "warnings": turn_warnings,
        "prompt_tokens_est": prompt_tokens_est,
        "n_ctx": effective_n_ctx,
        "web_used": bool(use_web and turn_sources),
    }

    return raw_messages, show_reasoning


def _chat_completion_args(req: ChatRequest, prepared_messages: List[Dict[str, Any]]) -> dict:
    settings = load_settings()

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(req, name)
        if name in req.model_fields_set and value is not None:
            return value
        stored = settings.get(name)
        return stored if stored is not None else fallback

    stop = req.stop
    if stop is None:
        raw_stop = str(settings.get("stop_sequences") or "").strip()
        if raw_stop:
            stop = [line.strip() for line in raw_stop.splitlines() if line.strip()]

    chosen_seed = pick("seed", None)
    if chosen_seed is None:
        chosen_seed = secrets.randbelow(2**31 - 1) + 1

    ai_turns: List[str] = []
    for m in prepared_messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        plain_ai = _extract_message_plain_text(content).strip()
        if role == "assistant" and plain_ai:
            ai_turns.append(plain_ai)
    ai_turns = ai_turns[-3:]
    if ai_turns and STATE.llm is not None and hasattr(STATE.llm, "tokenize"):
        try:
            prior_text = "\n".join(ai_turns)
            STATE.prior_assistant_tokens = list(
                STATE.llm.tokenize(prior_text.encode("utf-8", errors="replace"), add_bos=False)[-1024:]
            )
        except Exception:
            STATE.prior_assistant_tokens = []
    else:
        STATE.prior_assistant_tokens = []

    args = {
        "messages": prepared_messages,
        "max_tokens": int(pick("max_tokens", 512)),
        "temperature": float(pick("temperature", 0.7)),
        "top_p": float(pick("top_p", 0.95)),
        "min_p": float(pick("min_p", 0.05)),
        "top_k": int(pick("top_k", 40)),
        "repeat_penalty": float(pick("repeat_penalty", 1.0)),
        "presence_penalty": float(pick("presence_penalty", 0.0)),
        "stop": stop,
        "seed": chosen_seed,
    }
    if req.response_format is not None:
        args["response_format"] = req.response_format.model_dump(by_alias=True, exclude_none=True)
    if req.tools is not None:
        args["tools"] = [tool.model_dump(exclude_none=True) for tool in req.tools]
    if req.tool_choice is not None:
        args["tool_choice"] = req.tool_choice
    return args


class _ThinkGate:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, show_reasoning: bool):
        self.show_reasoning = show_reasoning
        self.inside = False
        self.pending = ""

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        for size in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def feed(self, text: str) -> str:
        if self.show_reasoning or not text:
            return text
        buf = self.pending + text
        lowered = buf.lower()
        out: List[str] = []
        cursor = 0
        self.pending = ""
        while cursor < len(buf):
            tag = self.CLOSE if self.inside else self.OPEN
            hit = lowered.find(tag, cursor)
            if hit < 0:
                tail = buf[cursor:]
                hold = self._partial_tail(tail.lower(), tag)
                if not self.inside:
                    out.append(tail[:-hold] if hold else tail)
                self.pending = tail[-hold:] if hold else ""
                break
            if not self.inside:
                out.append(buf[cursor:hit])
            cursor = hit + len(tag)
            self.inside = not self.inside
        return "".join(out)

    def flush(self) -> str:
        remainder = "" if self.inside else self.pending
        self.pending = ""
        return remainder


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _stream_generator(
    req: ChatRequest,
    prepared_messages: List[Dict[str, Any]],
    generation_id: int,
    show_reasoning: bool,
    turn_meta: Optional[Dict[str, Any]] = None,
):
    start = time.perf_counter()
    first_token_at: Optional[float] = None
    token_count = 0
    aborted = False
    failed = False
    stream = None
    reasoning_open = False
    think_gate = _ThinkGate(show_reasoning)
    try:
        stream_args = _chat_completion_args(req, prepared_messages)
        stream_args["stream"] = True
        llm = STATE.llm
        if llm is None:
            raise RuntimeError("No model resident.")
        stream = llm.create_chat_completion(**stream_args)
        for chunk in stream:
            if STATE.abort_requested:
                aborted = True
                break
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                reasoning = delta.pop("reasoning_content", None)
                if not show_reasoning and reasoning:
                    pass
                elif isinstance(reasoning, str) and reasoning:
                    prefix = ""
                    if not reasoning_open:
                        prefix = "<think>"
                    if isinstance(content, str) and content:
                        content = prefix + reasoning + "</think>" + content
                        reasoning_open = False
                    else:
                        content = prefix + reasoning
                        reasoning_open = True
                if isinstance(content, str) and content:
                    if show_reasoning and reasoning_open and reasoning is None:
                        content = "</think>" + content
                        reasoning_open = False
                    filtered = think_gate.feed(content)
                    if filtered:
                        delta["content"] = filtered
                        token_count += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                    else:
                        delta.pop("content", None)
                if not delta and choices[0].get("finish_reason") is None:
                    continue
            yield _sse(chunk)

        if show_reasoning and reasoning_open:
            yield _sse({
                "id": f"chatcmpl-{generation_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": "</think>"}}],
            })
        residual = think_gate.flush()
        if residual:
            yield _sse({
                "id": f"chatcmpl-{generation_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": residual}}],
            })
        if aborted:
            yield _sse({
                "id": f"chatcmpl-{generation_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "ghost_aborted": True,
            })

        elapsed_now = max(0.01, time.perf_counter() - start)
        decode_now = elapsed_now - ((first_token_at - start) if first_token_at else 0.0)
        rate_now = token_count / decode_now if decode_now > 0 else 0.0
        meta_payload = dict(turn_meta or {})
        meta_payload["tok_per_sec"] = round(rate_now, 1)
        meta_payload["elapsed_sec"] = round(elapsed_now, 2)
        meta_payload["chunks"] = token_count
        yield _sse({
            "id": f"chatcmpl-{generation_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [],
            "ghost_meta": meta_payload,
        })
        yield "data: [DONE]\n\n"
    except Exception as exc:
        failed = True
        log.error(f"Streaming generation error: {exc}")
        yield _sse({"error": {"message": str(exc)}})
        yield "data: [DONE]\n\n"
    finally:
        if stream is not None:
            closer = getattr(stream, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:
                    log.warning(f"Could not close generation stream: {exc}")
        STATE.end_generation(generation_id)
        elapsed = time.perf_counter() - start
        decode_time = elapsed - ((first_token_at - start) if first_token_at else elapsed)
        rate = token_count / decode_time if decode_time > 0 else 0.0
        outcome = "ABORTED" if aborted else ("FAILED" if failed else "DONE")
        log.info(
            f"STREAM {outcome} :: gen={generation_id} model={req.model} chunks={token_count} "
            f"time={elapsed:.2f}s decode={rate:.2f} chunk/s"
        )


@app.post("/v1/abort", dependencies=[Depends(verify_access)])
def abort_generation():
    active = STATE.request_abort()
    streaming = bool(active and STATE.streaming)
    log.info(
        f"ABORT REQUEST :: gen={STATE.generation_id} active={active} streaming={streaming}"
    )
    return {
        "status": "aborting" if active else "idle",
        "generation_id": STATE.generation_id,
        "interruptible": streaming,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_access)])
def chat_completions(req: ChatRequest):
    if STATE.llm is None:
        raise HTTPException(status_code=409, detail="No model resident. Open CONFIG and click MOUNT first.")

    generation_id = STATE.begin_generation(bool(req.stream))
    STATE.last_turn_meta = {}
    try:
        prepared_messages, show_reasoning = _prepare_pcp_messages(req)
        turn_meta = dict(STATE.last_turn_meta or {})
    except Exception:
        STATE.end_generation(generation_id)
        raise
    log.info(
        f"REQUEST :: gen={generation_id} model={req.model} stream={req.stream} "
        f"msgs={len(prepared_messages)}"
    )

    if req.stream:
        return StreamingResponse(
            _stream_generator(req, prepared_messages, generation_id, show_reasoning, turn_meta=turn_meta),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(STATE.end_generation, generation_id),
        )

    start = time.perf_counter()
    try:
        completion_args = _chat_completion_args(req, prepared_messages)
        completion_args["stream"] = False
        llm = STATE.llm
        if llm is None:
            raise RuntimeError("No model resident.")
        out = llm.create_chat_completion(**completion_args)
        if not show_reasoning:
            try:
                content = out["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    if cleaned:
                        out["choices"][0]["message"]["content"] = cleaned
            except (KeyError, IndexError, TypeError):
                pass
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        STATE.end_generation(generation_id)

    elapsed = time.perf_counter() - start
    usage = out.get("usage", {})
    tps = usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0.0
    if isinstance(out, dict):
        out["ghost_meta"] = {
            **turn_meta,
            "tok_per_sec": round(tps, 1),
            "elapsed_sec": round(elapsed, 2),
        }
    log.info(
        f"COMPLETION :: model={req.model} prompt_tok={usage.get('prompt_tokens', '?')} "
        f"completion_tok={usage.get('completion_tokens', '?')} time={elapsed:.2f}s speed={tps:.2f} tok/s"
    )
    return out


# =====================================================================
# STARTUP & SHUTDOWN
# =====================================================================

def startup_banner():
    _init_pairing_token()
    _init_pcp_index()
    if _personas_registry_path().is_file():
        active_persona = _get_active_persona()
        if active_persona:
            active_dir = _resolve_persona_folder(active_persona.get("folder", "."))
            with STATE._history_lock:
                STATE.conversation = _load_persona_chat_history(active_dir)
    if sys.stdout is not None:
        try:
            print(GHOST_BANNER)
        except UnicodeEncodeError:
            pass
    log.info("GHostPCP boot sequence initiated.")
    log.info(f"Model directory : {MODEL_DIR.resolve()}")
    log.info(f"PCP Data dir    : {PCP_DATA_DIR.resolve()}")
    log.info(f"Settings file   : {SETTINGS_PATH.resolve()}")
    log.info(f"Selected backend: {BACKEND_NAME} ({RUNTIME_LLAMA_LIB_DIR})")
    log.info(f"Bind address    : http://{BIND_HOST}:{PORT} (Desktop Loopback + Paired Remote)")
    lan_ip = get_lan_ip()
    ts_ip = get_tailscale_ip()
    if lan_ip:
        log.info(f"Phone (Wi-Fi)   : http://{lan_ip}:{PORT}/#{PAIRING_TOKEN}")
    if ts_ip:
        log.info(f"Tailscale URL   : http://{ts_ip}:{PORT}/#{PAIRING_TOKEN}")
    detected = scan_models()
    log.info(f"GGUF models detected: {len(detected)}")
    for m in detected:
        log.info(f"  -> {m['id']} ({m['size_gb']} GB)")
    log.info("GHostPCP Engine online.")


@app.on_event("startup")
def on_startup():
    startup_banner()


@app.on_event("shutdown")
def on_shutdown():
    log.info("Shutdown signal received. Stopping tunnel, microphone, and flushing VRAM ...")
    _AUDIO_REC.cancel()
    stop_cloudflare_tunnel()
    unload_current_model()
    log.info("GHostPCP terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHostPCP — GPU Hosted Pure Concentrated Personality")
    parser.add_argument("--api-only", action="store_true", help="Run the server without opening a desktop window")
    args = parser.parse_args()

    try:
        if args.api_only:
            uvicorn.run(
                app,
                host=BIND_HOST,
                port=PORT,
                log_level="warning",
                access_log=False,
                log_config=None,
            )
        else:
            import webview

            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=BIND_HOST,
                    port=PORT,
                    log_level="warning",
                    access_log=False,
                    log_config=None,
                )
            )
            server_thread = threading.Thread(target=server.run, daemon=True)
            server_thread.start()
            deadline = time.monotonic() + 20
            while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not server.started:
                server.should_exit = True
                server_thread.join(timeout=5)
                raise RuntimeError(f"GHostPCP server did not start on {LOOPBACK_HOST}:{PORT}.")

            try:
                webview.create_window(
                    "GHostPCP // GPU Hosted Pure Concentrated Personality",
                    f"http://{LOOPBACK_HOST}:{PORT}/",
                    width=1440,
                    height=960,
                    min_size=(920, 660),
                    background_color="#090a0d",
                    resizable=True,
                )
                webview.start()
            finally:
                stop_cloudflare_tunnel()
                server.should_exit = True
                server_thread.join(timeout=10)
    except Exception as exc:
        log.exception("GHostPCP failed to start")
        message = f"GHostPCP could not start.\n\n{exc}\n\nDetails: {LOG_PATH}"
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "GHostPCP startup error", 0x10)
        else:
            print(message, file=sys.stderr)
        raise
