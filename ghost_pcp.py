#!/usr/bin/env python3
"""
=======================================================================
 GHost PCP — GPU Hosted Pure Concentrated Personality
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
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import qrcode
import qrcode.image.svg
import sounddevice as sd
import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from gguf import GGUFReader, LlamaFileType
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from pcp_ingest import ingest_files
from pcp_retrieval import PCPIndex
from persona import DEFAULT_BLOCKED_PHRASES, DEFAULT_PROTECTED_WORDS, SYSTEM_PROMPT

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
        if not target_dll.exists():
            shutil.copy2(item, target_dll)

    if os.name == "nt":
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_lib)))
        os.environ["PATH"] = str(runtime_lib) + os.pathsep + os.environ.get("PATH", "")
    os.environ["GGML_BACKEND_PATH"] = str(runtime_lib)
    return selected, runtime_lib


BACKEND_NAME, RUNTIME_LLAMA_LIB_DIR = _configure_runtime_backend()

from llama_cpp import Llama, llama_cpp

# =====================================================================
# CONFIGURATION
# =====================================================================

def _find_available_port(preferred: int = 5555) -> int:
    if "GHOST_API_PORT" in os.environ:
        return int(os.environ["GHOST_API_PORT"])
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", candidate))
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
    "chat_format": "auto",
    "system_prompt": SYSTEM_PROMPT.strip(),
    "pcp_enabled": True,
    "pcp_top_k": 4,
    "tts_voice": "silas",
    "blocked_phrases": "\n".join(DEFAULT_BLOCKED_PHRASES),
    "protected_words": "\n".join(DEFAULT_PROTECTED_WORDS),
    "show_reasoning": False,
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": 0.05,
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
    format="%(asctime)s [GHost-PCP] %(levelname)s :: %(message)s",
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

    GHost PCP // GPU HOSTED PURE CONCENTRATED PERSONALITY
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
        self.locked = False
        self.busy = False


class IngestState:
    def __init__(self):
        self.running = False
        self.stage = "Idle"
        self.progress = 0.0
        self.error: Optional[str] = None
        self.last_result: Optional[dict] = None


class TunnelState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self.starting: bool = False
        self.error: Optional[str] = None


STATE = EngineState()
INGEST_STATE = IngestState()
TUNNEL_STATE = TunnelState()
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
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
    },
    "kara": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
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


def _init_pcp_index() -> None:
    global PCP_INDEX
    PCP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
    try:
        PCP_INDEX = PCPIndex(PCP_DATA_DIR, cache_dir=PCP_DATA_DIR, model_dir=bge_dir)
        if PCP_INDEX.ready:
            log.info(
                f"PCP Index online: {PCP_INDEX.file_count} volume(s), {PCP_INDEX.passage_count:,} passages."
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
        return
    expected = f"Bearer {PAIRING_TOKEN}"
    if not PAIRING_TOKEN or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401,
            detail="Remote access requires a valid pairing link. Scan the QR code shown in GHost PCP on your computer.",
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
    candidates = [
        APP_DIR / "cloudflared.exe",
        Path.home() / "OneDrive" / "Desktop" / "EVERYTHING" / "cloudflared.exe",
    ]
    which_cf = shutil.which("cloudflared")
    if which_cf:
        candidates.insert(0, Path(which_cf))
    for path in candidates:
        if path.is_file():
            return path

    target = APP_DIR / "cloudflared.exe"
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    log.info(f"Downloading cloudflared tunnel binary from {url} ...")
    urllib.request.urlretrieve(url, str(target))
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

def scan_models() -> List[Dict[str, Any]]:
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for f in sorted(MODEL_DIR.glob("*.gguf")):
        try:
            file_size_bytes = f.stat().st_size
        except OSError:
            file_size_bytes = 0
        models.append({
            "id": f.name,
            "object": "model",
            "created": int(f.stat().st_mtime) if f.exists() else 0,
            "owned_by": "GHost-PCP-Local",
            "size_gb": round(file_size_bytes / (1024 ** 3), 2),
            "file_size_bytes": file_size_bytes,
            "file_size_gib": round(file_size_bytes / GIB, 3),
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


def preflight_model(filename: str, context_size: int) -> dict:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    if STATE.loaded_model is None:
        _release_peer_ghost_anchor_vram()

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
    compute_reserve_bytes = int(0.5 * GIB + 0.04 * weight_bytes)
    devices = _probe_nvidia_gpus() if BACKEND_NAME == "cuda" else []
    warnings = []
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

        for _ in range(3):
            gc.collect()
        time.sleep(0.4)
        log.info("Resident model flushed. VRAM released.")


def load_model(
    filename: str,
    context_size: int,
    chat_format: str = "auto",
) -> dict:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    if STATE.locked or STATE.busy:
        raise HTTPException(status_code=409, detail="A model change is already in progress.")

    STATE.locked = True
    try:
        unload_current_model()
        preflight = preflight_model(filename, context_size)
        gpu_layers = int(preflight["n_gpu_layers"])
        log.info(f"Loading '{filename}' (preflight n_gpu_layers={gpu_layers}, n_ctx={context_size}) ...")
        t0 = time.perf_counter()
        try:
            STATE.llm = Llama(
                model_path=str(path),
                n_gpu_layers=gpu_layers,
                n_ctx=context_size,
                n_threads=N_THREADS,
                n_batch=N_BATCH,
                type_k=int(preflight["kv_type"]),
                type_v=int(preflight["kv_type"]),
                offload_kqv=gpu_layers > 0,
                flash_attn=FLASH_ATTN if gpu_layers > 0 else False,
                chat_format=None if chat_format == "auto" else chat_format,
                verbose=False,
            )
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
        log.info(f"Model '{filename}' resident in VRAM. Load time: {dt:.2f}s")
        return preflight
    finally:
        STATE.locked = False


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
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=0.05, ge=0.0, le=1.0)
    max_tokens: Optional[StrictInt] = Field(default=512, ge=1, le=32768)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[StrictInt] = None
    use_pcp: Optional[bool] = None
    pcp_top_k: Optional[int] = Field(default=None, ge=1, le=12)
    show_reasoning: Optional[bool] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = None


class LoadRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)
    chat_format: ChatFormat = "auto"


class PreflightRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)


class LoaderSettings(BaseModel):
    model: StrictStr = Field(default=DEFAULT_SETTINGS["model"], min_length=1, max_length=255)
    chat_format: ChatFormat = "auto"
    system_prompt: StrictStr = Field(default=DEFAULT_SETTINGS["system_prompt"], max_length=12000)
    pcp_enabled: bool = True
    pcp_top_k: StrictInt = Field(default=4, ge=1, le=12)
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=64)
    blocked_phrases: StrictStr = Field(default=DEFAULT_SETTINGS["blocked_phrases"], max_length=8000)
    protected_words: StrictStr = Field(default=DEFAULT_SETTINGS["protected_words"], max_length=8000)
    show_reasoning: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    seed: Optional[StrictInt] = None
    max_tokens: StrictInt = Field(default=512, ge=1, le=8192)
    stop_sequences: StrictStr = Field(default="", max_length=2000)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=512, le=32768)


class LocalPathIngestRequest(BaseModel):
    paths: List[str] = Field(min_length=1)
    blocked_phrases: Optional[str] = None
    protected_words: Optional[str] = None
    replace_existing: bool = False


class SaveSessionRequest(BaseModel):
    messages: List[Message] = Field(min_length=1)


class TTSRequest(BaseModel):
    text: StrictStr = Field(min_length=1, max_length=20000)
    voice: StrictStr = Field(default="silas", max_length=64)
    play_on_host: Optional[bool] = None


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        merged = {**DEFAULT_SETTINGS, **data}
        return LoaderSettings.model_validate(merged).model_dump()
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except Exception as exc:
        log.warning("Ignoring invalid settings file: %s", exc)
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: LoaderSettings) -> dict:
    data = settings.model_dump()
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


# =====================================================================
# BACKGROUND ARCHIVE INGESTION WORKER
# =====================================================================

def _start_background_ingest(
    file_items: List[Tuple[str, bytes]],
    blocked_phrases: List[str],
    protected_words: List[str],
    replace_existing: bool = False,
) -> None:
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")

    INGEST_STATE.running = True
    INGEST_STATE.stage = "Starting archive extraction..."
    INGEST_STATE.progress = 0.02
    INGEST_STATE.error = None
    INGEST_STATE.last_result = None

    def _progress(msg: str, frac: float):
        INGEST_STATE.stage = msg
        INGEST_STATE.progress = round(min(1.0, max(0.0, frac)), 3)

    def _worker():
        global PCP_INDEX
        try:
            summary = ingest_files(
                file_items=file_items,
                output_dir=PCP_DATA_DIR,
                blocked_phrases=blocked_phrases,
                protected_words=protected_words,
                replace_existing=replace_existing,
                progress_callback=_progress,
            )
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            if PCP_INDEX is None:
                PCP_INDEX = PCPIndex(
                    PCP_DATA_DIR,
                    cache_dir=PCP_DATA_DIR,
                    model_dir=bge_dir,
                    progress_callback=_progress,
                )
            else:
                PCP_INDEX.reload(progress_callback=_progress)
            summary["passages_indexed"] = PCP_INDEX.passage_count if PCP_INDEX else 0
            INGEST_STATE.last_result = summary
            INGEST_STATE.stage = (
                f"Complete: {summary['volumes_written']} PCP volume(s), "
                f"{summary['passages_indexed']:,} passages ready."
            )
            INGEST_STATE.progress = 1.0
            log.info(f"Archive ingestion complete: {summary}")
        except Exception as exc:
            log.error(f"Archive ingestion failed: {exc}")
            log.error(traceback.format_exc())
            INGEST_STATE.error = str(exc)
            INGEST_STATE.stage = f"Failed: {exc}"
        finally:
            INGEST_STATE.running = False

    threading.Thread(target=_worker, daemon=True).start()


# =====================================================================
# FASTAPI APPLICATION & ROUTES
# =====================================================================

app = FastAPI(title="GHost PCP", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    return {"object": "list", "data": models}


@app.post("/v1/preflight", dependencies=[Depends(verify_access)])
def preflight_endpoint(req: PreflightRequest):
    if STATE.locked:
        raise HTTPException(status_code=409, detail="A model change is in progress.")
    plan = preflight_model(req.model, req.n_ctx)
    if STATE.loaded_model:
        plan["warnings"].append(
            "A model is currently resident; this preview uses current free VRAM. "
            "Loading will unload it and recalculate before allocation."
        )
    return plan


@app.post("/v1/load", dependencies=[Depends(verify_access)])
def load_endpoint(req: LoadRequest):
    preflight = load_model(req.model, req.n_ctx, req.chat_format)
    return {
        "status": "loaded",
        "model": req.model,
        "load_seconds": round(STATE.load_seconds or 0.0, 2),
        "gpu_layers": STATE.loaded_gpu_layers,
        "context": req.n_ctx,
        "kv_cache_type": preflight["kv_cache_type"],
        "chat_format": req.chat_format,
        "preflight": preflight,
    }


@app.post("/v1/unload", dependencies=[Depends(verify_access)])
def unload_endpoint():
    if STATE.locked or STATE.busy:
        raise HTTPException(status_code=409, detail="A model change or generation is in progress.")
    STATE.locked = True
    try:
        unload_current_model()
    finally:
        STATE.locked = False
    return {"status": "unloaded"}


@app.post("/v1/reset", dependencies=[Depends(verify_access)])
def reset_endpoint():
    if STATE.locked or STATE.busy:
        raise HTTPException(status_code=409, detail="Wait for the current model operation to finish.")
    if STATE.llm is None:
        raise HTTPException(status_code=409, detail="No model is loaded.")
    STATE.llm.reset()
    return {"status": "reset", "model": STATE.loaded_model}


@app.get("/v1/status", dependencies=[Depends(verify_access)])
def status():
    pcp_ready = bool(PCP_INDEX and PCP_INDEX.ready)
    return {
        "engine": "GHost-PCP",
        "backend": BACKEND_NAME,
        "resident_model": STATE.loaded_model,
        "n_gpu_layers": STATE.loaded_gpu_layers,
        "n_ctx": STATE.loaded_context_size,
        "kv_cache_type": STATE.loaded_kv_cache_type,
        "chat_format": STATE.loaded_chat_format,
        "loading": STATE.locked,
        "generating": STATE.busy,
        "models_available": len(scan_models()),
        "pcp_ready": pcp_ready,
        "pcp_volumes": PCP_INDEX.file_count if pcp_ready and PCP_INDEX else 0,
        "pcp_passages": PCP_INDEX.passage_count if pcp_ready and PCP_INDEX else 0,
        "ingesting": INGEST_STATE.running,
        "offline": True,
    }


# =====================================================================
# PCP ARCHIVE INGESTION ROUTES
# =====================================================================

@app.get("/v1/pcp/status", dependencies=[Depends(verify_access)])
def pcp_status():
    pcp_ready = bool(PCP_INDEX and PCP_INDEX.ready)
    volume_files = sorted(p.name for p in PCP_DATA_DIR.glob("ghost_pcp_*.md")) if PCP_DATA_DIR.is_dir() else []
    return {
        "ready": pcp_ready,
        "volumes": PCP_INDEX.file_count if pcp_ready and PCP_INDEX else len(volume_files),
        "passages": PCP_INDEX.passage_count if pcp_ready and PCP_INDEX else 0,
        "volume_files": volume_files,
        "data_dir": str(PCP_DATA_DIR),
        "ingesting": INGEST_STATE.running,
        "stage": INGEST_STATE.stage,
        "progress": INGEST_STATE.progress,
        "error": INGEST_STATE.error,
        "last_result": INGEST_STATE.last_result,
    }


@app.post("/v1/pcp/ingest", dependencies=[Depends(verify_access)])
async def pcp_ingest_upload(
    files: List[UploadFile] = File(...),
    blocked_phrases: Optional[str] = Form(default=None),
    protected_words: Optional[str] = Form(default=None),
    replace_existing: bool = Form(default=False),
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

    _start_background_ingest(items, bp_list, pw_list, replace_existing=replace_existing)
    return {"status": "started", "files": len(items)}


@app.post("/v1/pcp/ingest-local", dependencies=[Depends(verify_access)])
def pcp_ingest_local(req: LocalPathIngestRequest):
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")
    settings = load_settings()
    bp_raw = req.blocked_phrases if req.blocked_phrases is not None else settings.get("blocked_phrases", "")
    pw_raw = req.protected_words if req.protected_words is not None else settings.get("protected_words", "")
    bp_list = [line.strip() for line in bp_raw.splitlines() if line.strip()]
    pw_list = [line.strip() for line in pw_raw.splitlines() if line.strip()]

    items: List[Tuple[str, bytes]] = []
    for raw_path in req.paths:
        p = Path(raw_path.strip().strip('"'))
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {p}")
        if p.is_dir():
            for child in sorted(p.glob("*.md")):
                items.append((child.name, child.read_bytes()))
            conv = p / "conversations.json"
            if conv.is_file():
                items.append((conv.name, conv.read_bytes()))
        else:
            items.append((p.name, p.read_bytes()))

    if not items:
        raise HTTPException(status_code=400, detail="No supported export files (.zip, .json, .jsonl, .md) found at path.")

    _start_background_ingest(items, bp_list, pw_list, replace_existing=req.replace_existing)
    return {"status": "started", "files": len(items)}


@app.post("/v1/pcp/save-session", dependencies=[Depends(verify_access)])
def pcp_save_session(req: SaveSessionRequest):
    global PCP_INDEX
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Wait for archive indexing to finish before saving session.")
    if STATE.locked or STATE.busy:
        raise HTTPException(status_code=409, detail="Wait for current generation to finish before saving session.")

    bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
    if PCP_INDEX is None:
        PCP_INDEX = PCPIndex(PCP_DATA_DIR, cache_dir=PCP_DATA_DIR, model_dir=bge_dir)

    raw_msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    try:
        result = PCP_INDEX.append_session(raw_msgs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(f"Save session failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save session: {exc}") from exc

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


@app.post("/v1/pcp/reindex", dependencies=[Depends(verify_access)])
def pcp_reindex():
    if INGEST_STATE.running:
        raise HTTPException(status_code=409, detail="Archive processing is already running.")

    INGEST_STATE.running = True
    INGEST_STATE.stage = "Re-indexing ghost_pcp_*.md volumes..."
    INGEST_STATE.progress = 0.1
    INGEST_STATE.error = None

    def _progress(msg: str, frac: float):
        INGEST_STATE.stage = msg
        INGEST_STATE.progress = round(min(1.0, max(0.0, frac)), 3)

    def _worker():
        global PCP_INDEX
        try:
            bge_dir = BUNDLED_BGE_DIR if BUNDLED_BGE_DIR.is_dir() else None
            if PCP_INDEX is None:
                PCP_INDEX = PCPIndex(
                    PCP_DATA_DIR,
                    cache_dir=PCP_DATA_DIR,
                    model_dir=bge_dir,
                    progress_callback=_progress,
                )
            else:
                PCP_INDEX.reload(progress_callback=_progress)
            INGEST_STATE.stage = (
                f"Indexed {PCP_INDEX.file_count} volume(s), {PCP_INDEX.passage_count:,} passages."
                if PCP_INDEX and PCP_INDEX.ready
                else "No ghost_pcp_*.md files found in pcp_data."
            )
            INGEST_STATE.progress = 1.0
        except Exception as exc:
            INGEST_STATE.error = str(exc)
            INGEST_STATE.stage = f"Failed: {exc}"
        finally:
            INGEST_STATE.running = False

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


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
def remote_qr_svg(target: str = "lan"):
    info = remote_status()
    if target == "tunnel" and info["tunnel_url"]:
        url = info["tunnel_url"]
    elif target == "tailscale" and info["tailscale_url"]:
        url = info["tailscale_url"]
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
    client_host = request.client.host if request.client else ""
    forwarded = (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for")
        or request.headers.get("x-real-ip")
    )
    is_local_loopback = client_host in ("127.0.0.1", "::1", "localhost") and not forwarded
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


# =====================================================================
# CHAT COMPLETIONS (WITH CITATION-FREE PCP GROUNDING)
# =====================================================================

def _prepare_pcp_messages(req: ChatRequest) -> Tuple[List[Dict[str, Any]], bool]:
    settings = load_settings()
    use_pcp = req.use_pcp if req.use_pcp is not None else bool(settings.get("pcp_enabled", True))
    top_k = req.pcp_top_k or int(settings.get("pcp_top_k", 4))
    show_reasoning = req.show_reasoning if req.show_reasoning is not None else bool(settings.get("show_reasoning", False))

    raw_messages = [m.model_dump(exclude_none=True) for m in req.messages]
    if not raw_messages:
        return raw_messages, show_reasoning

    # Locate the latest user message for semantic retrieval
    latest_user_idx = None
    latest_user_text = ""
    for idx in range(len(raw_messages) - 1, -1, -1):
        if raw_messages[idx].get("role") == "user" and isinstance(raw_messages[idx].get("content"), str):
            latest_user_idx = idx
            latest_user_text = raw_messages[idx]["content"].strip()
            break

    if use_pcp and PCP_INDEX is not None and PCP_INDEX.ready and latest_user_text:
        excerpts = PCP_INDEX.search(latest_user_text, limit=top_k)
        if excerpts:
            archive_block = "\n\n---\n\n".join(excerpts)
            base_sys = settings.get("system_prompt") or SYSTEM_PROMPT.strip()
            if raw_messages[0].get("role") == "system":
                existing_sys = str(raw_messages[0].get("content") or base_sys).strip()
                if "ARCHIVE EXCERPTS:" not in existing_sys:
                    raw_messages[0]["content"] = existing_sys + "\n\nARCHIVE EXCERPTS:\n" + archive_block
            else:
                raw_messages.insert(
                    0,
                    {"role": "system", "content": base_sys + "\n\nARCHIVE EXCERPTS:\n" + archive_block},
                )

    # For Qwen3 models when reasoning blocks are hidden, append /no_think to keep replies immediate
    if (
        latest_user_idx is not None
        and STATE.loaded_model
        and "qwen3" in STATE.loaded_model.lower()
        and not show_reasoning
    ):
        # Find latest user message again in case system prompt was inserted at index 0
        for idx in range(len(raw_messages) - 1, -1, -1):
            if raw_messages[idx].get("role") == "user" and isinstance(raw_messages[idx].get("content"), str):
                if "/no_think" not in raw_messages[idx]["content"]:
                    raw_messages[idx]["content"] = raw_messages[idx]["content"].rstrip() + "\n/no_think"
                break

    return raw_messages, show_reasoning


def _chat_completion_args(req: ChatRequest, prepared_messages: List[Dict[str, Any]]) -> dict:
    args = {
        "messages": prepared_messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "min_p": req.min_p,
        "stop": req.stop,
        "seed": req.seed,
    }
    if req.response_format is not None:
        args["response_format"] = req.response_format.model_dump(by_alias=True, exclude_none=True)
    if req.tools is not None:
        args["tools"] = [tool.model_dump(exclude_none=True) for tool in req.tools]
    if req.tool_choice is not None:
        args["tool_choice"] = req.tool_choice
    return args


def _stream_generator(req: ChatRequest, prepared_messages: List[Dict[str, Any]]):
    start = time.perf_counter()
    token_count = 0

    try:
        stream_args = _chat_completion_args(req, prepared_messages)
        stream_args["stream"] = True
        assert STATE.llm is not None
        stream = STATE.llm.create_chat_completion(**stream_args)
        for chunk in stream:
            choice = (chunk.get("choices") or [{}])[0]
            if choice.get("delta", {}).get("content"):
                token_count += 1
            yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
    except Exception as e:
        log.error(f"Streaming generation error: {e}")
        yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
    finally:
        STATE.busy = False

    elapsed = time.perf_counter() - start
    tps = token_count / elapsed if elapsed > 0 else 0.0
    log.info(f"STREAM DONE :: model={req.model} tokens={token_count} time={elapsed:.2f}s speed={tps:.2f} tok/s")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(verify_access)])
def chat_completions(req: ChatRequest):
    if STATE.locked:
        raise HTTPException(status_code=409, detail="Model change in progress.")
    if STATE.busy:
        raise HTTPException(status_code=409, detail="A generation is already in progress.")
    if STATE.llm is None:
        raise HTTPException(status_code=409, detail="No model resident. Open CONFIG and click MOUNT first.")

    prepared_messages, show_reasoning = _prepare_pcp_messages(req)
    log.info(f"REQUEST :: model={req.model} stream={req.stream} msgs={len(prepared_messages)}")

    if req.stream:
        STATE.busy = True
        return StreamingResponse(
            _stream_generator(req, prepared_messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    start = time.perf_counter()
    STATE.busy = True
    try:
        completion_args = _chat_completion_args(req, prepared_messages)
        completion_args["stream"] = False
        out = STATE.llm.create_chat_completion(**completion_args)
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
        STATE.busy = False

    elapsed = time.perf_counter() - start
    usage = out.get("usage", {})
    tps = usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0.0
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
    if sys.stdout is not None:
        try:
            print(GHOST_BANNER)
        except UnicodeEncodeError:
            pass
    log.info("GHost PCP boot sequence initiated.")
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
    log.info("GHost PCP Engine online.")


@app.on_event("startup")
def on_startup():
    startup_banner()


@app.on_event("shutdown")
def on_shutdown():
    log.info("Shutdown signal received. Stopping tunnel and flushing VRAM ...")
    stop_cloudflare_tunnel()
    unload_current_model()
    log.info("GHost PCP terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHost PCP — GPU Hosted Pure Concentrated Personality")
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
                raise RuntimeError(f"GHost PCP server did not start on {LOOPBACK_HOST}:{PORT}.")

            try:
                webview.create_window(
                    "GHost PCP // GPU Hosted Pure Concentrated Personality",
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
        log.exception("GHost PCP failed to start")
        message = f"GHost PCP could not start.\n\n{exc}\n\nDetails: {LOG_PATH}"
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "GHost PCP startup error", 0x10)
        else:
            print(message, file=sys.stderr)
        raise
