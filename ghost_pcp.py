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
import ctypes
import gc
import hashlib
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
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_vulkan_runtime() -> bool:
    """Detect Vulkan driver/runtime availability without requiring the developer SDK."""
    if os.name == "nt":
        try:
            ctypes.WinDLL("vulkan-1.dll")
            return True
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        for lib_name in ("libvulkan.so.1", "libvulkan.so"):
            try:
                ctypes.CDLL(lib_name)
                return True
            except OSError:
                continue
    return shutil.which("vulkaninfo") is not None


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
def _detect_physical_cores() -> int:
    """Return physical CPU core count (avoiding SMT thread oversubscription in llama.cpp)."""
    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            size = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
            if size.value > 0:
                buf = ctypes.create_string_buffer(size.value)
                if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
                    count = 0
                    offset = 0
                    raw = buf.raw[:size.value]
                    while offset + 8 <= len(raw):
                        rel = ctypes.c_uint32.from_buffer_copy(raw, offset).value
                        entry_size = ctypes.c_uint32.from_buffer_copy(raw, offset + 4).value
                        if entry_size == 0:
                            break
                        if rel == 0:
                            count += 1
                        offset += entry_size
                    if count > 0:
                        return max(1, min(64, count))
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            pairs: Set[Tuple[str, str]] = set()
            phys_id = "0"
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    pairs.add((phys_id, line.split(":", 1)[1].strip()))
            if pairs:
                return max(1, min(64, len(pairs)))
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=2).strip()
            val = int(out)
            if val > 0:
                return max(1, min(64, val))
        except Exception:
            pass
    logical = os.cpu_count() or 8
    return max(1, min(64, (logical // 2) if logical >= 4 else logical))


N_THREADS = _detect_physical_cores()
FLASH_ATTN = True
N_BATCH = 512

DEFAULT_SETTINGS = {
    "model": "Qwen3-14B-Q5_K_M.gguf",
    "mmproj_model": "auto",
    "chat_format": "auto",
    "auto_mount_model": True,
    "system_prompt": "",
    "pcp_enabled": True,
    "pcp_top_k": 4,
    "tts_engine": "piper",
    "tts_voice": "silas",
    "tts_api_url": "http://127.0.0.1:3900",
    "tts_api_model": "tts-1",
    "tts_api_key": "",
    "mic_device": "auto",
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

GHOST_ANSI_BANNER = "\n".join([
    "",
    "\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;255;85;255;48;2;0;0;0m▄▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄▄\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m▄▄\033[0;38;2;170;0;170;48;2;0;0;0m▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄▄ ▄▄▄▄▄\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄     \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;255;85;255;48;2;0;0;0m▄▄\033[0;38;2;170;0;170;48;2;0;0;0m▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄▄\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m▄▄▄▄▄\033[0;38;2;170;0;170;48;2;0;0;0m▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄▄▄▄▄\033[48;2;0;0;0m  \033[48;2;0;0;0m      \033[38;2;0;0;0;48;2;34;0;61m▀\033[38;2;0;0;0;48;2;113;29;168m▀\033[38;2;50;0;88;48;2;192;137;227m▀\033[38;2;97;10;148;48;2;232;208;246m▀\033[38;2;125;33;178;48;2;246;235;251m▀\033[38;2;126;34;179;48;2;247;235;252m▀\033[38;2;103;12;155;48;2;234;209;246m▀\033[38;2;60;0;98;48;2;199;141;231m▀\033[38;2;0;0;0;48;2;128;34;182m▀\033[38;2;0;0;0;48;2;43;0;72m▀\033[48;2;0;0;0m      \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;255;85;255;48;2;0;0;0m▄▀▀\033[0;38;2;170;0;170;48;2;0;0;0m       ▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     █\033[0;38;2;85;85;85;48;2;0;0;0m░\033[0;38;2;170;0;170;48;2;0;0;0m    █\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;255;85;255;48;2;0;0;0m▄▀▀\033[0;38;2;170;0;170;48;2;0;0;0m    ▀▀▄  \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;255;85;255;48;2;0;0;0m▄▀▀\033[0;38;2;170;0;170;48;2;0;0;0m      ▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m           █\033[48;2;0;0;0m  \033[48;2;0;0;0m     \033[38;2;34;0;65;48;2;107;40;164m▀\033[38;2;141;65;195;48;2;243;237;248m▀\033[38;2;242;231;249;48;2;253;253;253m▀\033[38;2;252;252;252;48;2;255;255;255m▀\033[38;2;253;253;253;48;2;255;255;255m▀\033[38;2;254;254;254;48;2;255;255;255m▀\033[38;2;254;254;254;48;2;255;254;255m▀\033[38;2;254;254;254;48;2;254;254;255m▀\033[38;2;252;252;252;48;2;255;255;255m▀\033[38;2;242;231;248;48;2;253;253;253m▀\033[38;2;158;70;210;48;2;248;242;251m▀\033[38;2;42;0;71;48;2;123;42;180m▀\033[38;2;0;0;0;48;2;13;1;26m▀\033[48;2;0;0;0m    \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m          ▒\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     █\033[0;38;2;85;85;85;48;2;0;0;0m▒\033[0;38;2;170;0;170;48;2;0;0;0m    █\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m          █ \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m         ▒\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m           █\033[48;2;0;0;0m  \033[38;2;0;0;0;48;2;20;2;53m▀\033[38;2;9;0;31;48;2;138;110;173m▀\033[38;2;12;0;36;48;2;138;107;172m▀\033[38;2;0;0;0;48;2;81;45;127m▀\033[38;2;36;0;77;48;2;83;22;140m▀\033[38;2;195;162;223;48;2;234;222;243m▀\033[38;2;251;251;251;48;2;254;251;255m▀\033[38;2;253;247;255;48;2;212;204;218m▀\033[38;2;199;194;206;48;2;111;99;131m▀\033[38;2;150;142;165;48;2;28;8;67m▀\033[38;2;247;246;248;48;2;129;118;149m▀\033[38;2;254;253;254;48;2;255;255;255m▀\033[38;2;254;254;254;48;2;204;199;211m▀\033[38;2;183;176;191;48;2;98;82;121m▀\033[38;2;139;123;158;48;2;24;0;69m▀\033[38;2;249;249;249;48;2;155;137;175m▀\033[38;2;210;173;233;48;2;254;251;254m▀\033[38;2;61;2;110;48;2;95;16;158m▀\033[38;2;35;3;72;48;2;169;131;204m▀\033[38;2;94;41;146;48;2;238;229;243m▀\033[38;2;103;60;150;48;2;238;220;244m▀\033[38;2;26;3;54;48;2;67;26;115m▀\033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m▐\033[0;38;2;255;85;255;48;2;0;0;0m▌\033[0;38;2;170;0;170;48;2;0;0;0m          ░\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m█     \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;85;85;85;48;2;0;0;0m░\033[0;38;2;170;0;170;48;2;0;0;0m    █\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m     ▐▌\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m▐▌\033[0;38;2;170;0;170;48;2;0;0;0m         ░\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m▀▀▀\033[0;38;2;255;85;255;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m     █▀▀▀\033[48;2;0;0;0m  \033[38;2;52;22;98;48;2;37;12;78m▀\033[38;2;234;216;243;48;2;196;166;221m▀\033[38;2;250;250;250;48;2;245;220;253m▀\033[38;2;197;182;212;48;2;205;192;217m▀\033[38;2;95;44;147;48;2;94;54;141m▀\033[38;2;254;248;255;48;2;232;217;241m▀\033[38;2;254;246;255;48;2;249;237;255m▀\033[38;2;198;186;208;48;2;244;233;251m▀\033[38;2;80;47;120;48;2;255;255;255m▀\033[38;2;103;66;149;48;2;255;255;255m▀\033[38;2;115;95;145;48;2;245;237;249m▀\033[38;2;255;255;255;48;2;255;254;255m▀\033[38;2;202;195;211;48;2;255;251;255m▀\033[38;2;90;56;127;48;2;255;255;255m▀\033[38;2;124;90;164;48;2;255;255;255m▀\033[38;2;155;128;181;48;2;246;232;253m▀\033[38;2;255;255;255;48;2;255;254;255m▀\033[38;2;104;45;156;48;2;133;93;172m▀\033[38;2;175;148;199;48;2;199;178;216m▀\033[38;2;253;253;253;48;2;244;215;252m▀\033[38;2;222;190;238;48;2;177;139;210m▀\033[38;2;68;32;110;48;2;28;4;67m▀\033[0m",
    "\033[0;38;2;255;85;255;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;255;85;255;48;2;0;0;0m▄▄▄\033[0;38;2;170;0;170;48;2;0;0;0m▄\033[0;38;2;255;85;255;48;2;0;0;0m▄\033[0;38;2;170;0;170;48;2;0;0;0m▄▄\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m           █\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     █\033[0;38;2;85;85;85;48;2;0;0;0m▓\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m ▀▄▄   \033[0;38;2;85;85;85;48;2;0;0;0m▀▄▀▀▀\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;255;85;255;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[38;2;5;0;33;48;2;0;0;0m▀\033[38;2;140;105;177;48;2;51;26;94m▀\033[38;2;221;181;252;48;2;205;164;238m▀\033[38;2;244;233;255;48;2;208;170;253m▀\033[38;2;207;190;222;48;2;240;221;255m▀\033[38;2;171;142;201;48;2;243;225;255m▀\033[38;2;244;231;253;48;2;234;217;252m▀\033[38;2;253;248;255;48;2;255;247;255m▀\033[38;2;163;153;177;48;2;141;128;161m▀\033[38;2;186;178;197;48;2;0;0;38m▀\033[38;2;242;242;241;48;2;68;41;106m▀\033[38;2;251;246;250;48;2;70;49;104m▀\033[38;2;244;241;243;48;2;66;43;101m▀\033[38;2;200;191;208;48;2;0;0;42m▀\033[38;2;119;106;140;48;2;88;73;115m▀\033[38;2;236;223;248;48;2;248;232;254m▀\033[38;2;255;250;255;48;2;251;236;255m▀\033[38;2;146;117;177;48;2;144;116;174m▀\033[38;2;172;133;210;48;2;150;94;211m▀\033[38;2;228;187;252;48;2;184;138;220m▀\033[38;2;109;69;151;48;2;23;3;62m▀\033[38;2;1;0;19;48;2;0;0;0m▀\033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m█     \033[0;38;2;255;85;255;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     ░\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;85;85;85;48;2;0;0;0m░\033[0;38;2;170;0;170;48;2;0;0;0m    \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m▓░\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;0;0;0m▄▄▄\033[0;38;2;170;0;170;48;2;0;0;0m ▀▄   \033[0;38;2;85;85;85;48;2;0;0;0m▀▄\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;85;85;85;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[48;2;0;0;0m \033[38;2;0;0;22;48;2;0;0;0m▀\033[38;2;94;61;139;48;2;3;0;32m▀\033[38;2;211;161;250;48;2;92;54;136m▀\033[38;2;197;155;252;48;2;205;154;238m▀\033[38;2;223;198;253;48;2;207;171;252m▀\033[38;2;234;216;255;48;2;229;208;254m▀\033[38;2;237;221;253;48;2;230;211;254m▀\033[38;2;253;244;255;48;2;237;219;255m▀\033[38;2;107;102;129;48;2;233;229;245m▀\033[38;2;134;43;190;48;2;116;58;143m▀\033[38;2;170;67;216;48;2;253;96;255m▀\033[38;2;169;61;210;48;2;228;84;234m▀\033[38;2;83;54;112;48;2;132;98;152m▀\033[38;2;226;217;238;48;2;244;227;255m▀\033[38;2;235;213;254;48;2;219;192;251m▀\033[38;2;245;224;255;48;2;238;208;253m▀\033[38;2;149;118;179;48;2;148;114;183m▀\033[38;2;123;76;168;48;2;18;0;72m▀\033[38;2;57;27;100;48;2;0;0;0m▀  \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m▐▌    \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m     ▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;85;85;85;48;2;0;0;0m▒\033[0;38;2;170;0;170;48;2;0;0;0m    \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m▒\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m▐▌\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;255;85;255;48;2;170;0;170m░\033[0;38;2;170;0;170;48;2;0;0;0m         \033[0;38;2;85;85;85;48;2;0;0;0m▐▌\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[48;2;0;0;0m  \033[38;2;0;0;0;48;2;0;0;10m▀\033[38;2;32;4;81;48;2;106;68;147m▀\033[38;2;206;155;233;48;2;218;164;253m▀\033[38;2;205;160;253;48;2;185;141;252m▀\033[38;2;223;198;253;48;2;219;190;253m▀\033[38;2;224;198;255;48;2;210;175;255m▀\033[38;2;222;200;253;48;2;218;189;255m▀\033[38;2;237;216;255;48;2;219;192;254m▀\033[38;2;176;164;194;48;2;232;205;255m▀\033[38;2;130;76;146;48;2;218;202;246m▀\033[38;2;130;84;146;48;2;220;200;249m▀\033[38;2;191;174;211;48;2;227;196;255m▀\033[38;2;232;205;254;48;2;217;184;255m▀\033[38;2;214;181;251;48;2;208;167;252m▀\033[38;2;232;191;252;48;2;225;174;254m▀\033[38;2;129;95;169;48;2;101;66;143m▀\033[38;2;0;0;14;48;2;0;0;6m▀\033[48;2;0;0;0m   \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m    \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;85;85;85;48;2;0;0;0m░\033[0;38;2;170;0;170;48;2;0;0;0m    \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m          \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▒\033[0;38;2;170;0;170;48;2;0;0;0m         \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[38;2;0;0;0;48;2;22;8;60m▀\033[38;2;0;0;3;48;2;79;46;123m▀\033[38;2;39;15;86;48;2;194;157;222m▀\033[38;2;207;167;229;48;2;175;134;252m▀\033[38;2;179;128;253;48;2;154;103;250m▀\033[38;2;198;162;250;48;2;163;116;252m▀\033[38;2;192;154;253;48;2;157;113;253m▀\033[38;2;196;157;255;48;2;176;136;255m▀\033[38;2;204;170;255;48;2;187;151;255m▀\033[38;2;209;178;255;48;2;192;157;255m▀\033[38;2;208;175;253;48;2;195;161;255m▀\033[38;2;212;177;255;48;2;189;154;253m▀\033[38;2;212;177;255;48;2;172;131;252m▀\033[38;2;206;172;253;48;2;166;125;253m▀\033[38;2;209;173;255;48;2;182;143;253m▀\033[38;2;196;150;252;48;2;183;125;252m▀\033[38;2;216;154;251;48;2;193;127;227m▀\033[38;2;64;33;109;48;2;24;6;66m▀\033[48;2;0;0;0m    \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;85;85;85;48;2;0;0;0m▀▄▄\033[0;38;2;170;0;170;48;2;0;0;0m       \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;85;85;85;48;2;0;0;0m▀▄▄\033[0;38;2;170;0;170;48;2;0;0;0m    \033[0;38;2;85;85;85;48;2;0;0;0m▄▄▀\033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;170;0;170m▓\033[0;38;2;170;0;170;48;2;0;0;0m      \033[0;38;2;85;85;85;48;2;0;0;0m▄▄▀\033[0;38;2;170;0;170;48;2;0;0;0m  \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m█\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[38;2;22;4;71;48;2;0;0;0m▀\033[38;2;169;104;203;48;2;23;1;66m▀\033[38;2;230;146;253;48;2;87;35;138m▀\033[38;2;181;105;253;48;2;123;66;164m▀\033[38;2;200;128;255;48;2;103;57;146m▀\033[38;2;198;136;250;48;2;58;32;112m▀\033[38;2;197;131;254;48;2;146;86;181m▀\033[38;2;154;106;253;48;2;207;125;253m▀\033[38;2;170;130;254;48;2;145;90;254m▀\033[38;2;179;141;254;48;2;154;99;254m▀\033[38;2;165;124;252;48;2;193;122;252m▀\033[38;2;179;125;253;48;2;197;134;226m▀\033[38;2;204;134;254;48;2;81;43;131m▀\033[38;2;202;125;254;48;2;104;48;150m▀\033[38;2;151;92;249;48;2;231;131;253m▀\033[38;2;202;126;251;48;2;198;113;233m▀\033[38;2;130;74;170;48;2;34;10;83m▀\033[38;2;0;0;21;48;2;0;0;0m▀    \033[0m",
    "\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀▀▀▀▀\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀▀\033[0;38;2;170;0;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀▀▀▀\033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀▀▀▀\033[0;38;2;170;0;170;48;2;0;0;0m     \033[0;38;2;170;170;170;48;2;0;0;0m \033[0;38;2;170;0;170;48;2;0;0;0m   \033[0;38;2;85;85;85;48;2;0;0;0m▀▀▀▀▀▀▀\033[0;38;2;170;0;170;48;2;0;0;0m   \033[48;2;0;0;0m  \033[48;2;0;0;0m   \033[38;2;0;0;16;48;2;0;0;0m▀\033[38;2;0;0;10;48;2;0;0;0m▀ \033[38;2;16;0;68;48;2;0;0;0m▀\033[38;2;168;92;207;48;2;22;3;68m▀\033[38;2;203;115;249;48;2;70;31;117m▀\033[38;2;197;117;239;48;2;51;21;95m▀\033[38;2;149;91;184;48;2;2;0;37m▀\033[38;2;32;9;83;48;2;0;0;0m▀\033[38;2;0;0;8;48;2;0;0;0m▀\033[38;2;7;0;39;48;2;0;0;0m▀\033[38;2;106;50;153;48;2;0;0;13m▀\033[38;2;63;25;113;48;2;0;0;0m▀\033[38;2;1;0;16;48;2;0;0;0m▀     \033[0m",
    "",
    "\033[38;2;255;85;255m    GHost\033[38;2;154;123;255mPCP\033[38;2;170;170;170m // GPU HOSTED PURE CONCENTRATED PERSONALITY\033[0m",
    "\033[38;2;130;130;145m    Break Your AI Out of the Cloud and Onto Your Own Hardware\033[0m",
    "\033[38;2;170;0;170m=============================================================================================\033[0m",
    "",
])


def _enable_windows_ansi_console(stream: Any = None) -> bool:
    """Enable Windows VT100 sequences and UTF-8 encoding on interactive TTY streams."""
    out = sys.stdout if stream is None else stream
    if out is None or not bool(getattr(out, "isatty", lambda: False)()):
        return False
    if os.name == "nt" and out is sys.stdout and hasattr(out, "fileno"):
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if handle and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            if hasattr(out, "reconfigure"):
                out.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return True


def _select_startup_banner(stream: Any = None) -> str:
    """Choose the TrueColor ANSI banner for capable TTY terminals, else plain ASCII fallback."""
    out = sys.stdout if stream is None else stream
    if out is None or not _enable_windows_ansi_console(out):
        return GHOST_BANNER
    enc = getattr(out, "encoding", None) or "utf-8"
    try:
        GHOST_ANSI_BANNER.encode(enc)
        return GHOST_ANSI_BANNER
    except (UnicodeEncodeError, LookupError):
        return GHOST_BANNER

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
        self.loaded_vram_bytes: int = 0
        self.loading_model: Optional[str] = None
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
        self.auto_archived_signatures: Set[str] = set()

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


_HARMONY_OPEN_RE = re.compile(
    r"(?:<\|start\|>[a-z0-9_:-]*)?<\|channel\|>(analysis|commentary)(?:<\|constrain\|>[^\s<]+)?(?:<\|message\|>)?",
    re.IGNORECASE,
)
_HARMONY_FINAL_RE = re.compile(
    r"(?:<\|end\|>)?(?:<\|start\|>[a-z0-9_:-]*)?<\|channel\|>final(?:<\|message\|>)?",
    re.IGNORECASE,
)
_HARMONY_CLOSE_RE = re.compile(r"<\|end\|>|<\|return\|>|<\|start\|>[a-z0-9_:-]*|<\|message\|>", re.IGNORECASE)


def _normalize_reasoning_tags(text: str) -> str:
    """Normalize Harmony (<|channel|>), <thought>, and [THINK] markers into standard <think> blocks."""
    out = str(text or "")
    if not out:
        return ""
    out = re.sub(r"<thought>|\[THINK\]", "<think>", out, flags=re.IGNORECASE)
    out = re.sub(r"</thought>|\[/THINK\]", "</think>", out, flags=re.IGNORECASE)
    if "<|" in out:
        out = _HARMONY_OPEN_RE.sub("<think>", out)
        out = _HARMONY_FINAL_RE.sub("</think>", out)
        out = _HARMONY_CLOSE_RE.sub("", out)
        # Ensure balanced </think> only if preceded by <think> (drop orphan leading </think> from direct final channel)
        parts = out.split("<think>")
        if len(parts) == 1:
            out = out.replace("</think>", "")
        else:
            rebuilt = [parts[0].replace("</think>", "")]
            for idx, seg in enumerate(parts[1:], start=1):
                if idx < len(parts) - 1 and "</think>" not in seg:
                    seg = seg + "</think>"
                rebuilt.append("<think>" + seg)
            out = "".join(rebuilt)
    return out


def _strip_reasoning_blocks(text: str) -> str:
    """Remove complete or unclosed reasoning blocks across all model formats."""
    normalized = _normalize_reasoning_tags(text)
    cleaned = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", normalized, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _clean_text_for_speech(text: str) -> str:
    """Strip reasoning blocks, code blocks, and Markdown asterisks so Piper speaks naturally."""
    cleaned = _strip_reasoning_blocks(str(text))
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


def _play_wav_on_host(wav_bytes: bytes) -> bool:
    """Play WAV bytes on the host default output device, resampling to native rate if needed."""
    try:
        import numpy as np

        sd.stop()
        data, fs = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        try:
            sd.play(data, fs)
            return True
        except Exception as rate_exc:
            out_dev = sd.query_devices(kind="output") or {}
            native_fs = int(round(float(out_dev.get("default_samplerate") or 48000)))
            if native_fs > 0 and native_fs != int(fs):
                n_out = max(1, int(round(len(data) * native_fs / fs)))
                if data.ndim == 1:
                    resampled = np.interp(
                        np.linspace(0, len(data) - 1, n_out),
                        np.arange(len(data)),
                        data,
                    ).astype(np.float32)
                else:
                    resampled = np.stack(
                        [
                            np.interp(
                                np.linspace(0, len(data) - 1, n_out),
                                np.arange(len(data)),
                                data[:, ch],
                            )
                            for ch in range(data.shape[1])
                        ],
                        axis=1,
                    ).astype(np.float32)
                sd.play(resampled, native_fs)
                return True
            raise rate_exc
    except Exception as exc:
        log.warning(f"sounddevice local playback warning: {exc}")
        return False


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
            import piper
            from piper import PiperVoice

            cache_key = str(model_path.resolve())
            voice = _PIPER_VOICES.get(cache_key)
            if voice is None:
                espeak_candidates = [
                    RESOURCE_DIR / "piper" / "espeak-ng-data",
                    Path(piper.__file__).resolve().parent / "espeak-ng-data",
                ]
                espeak_dir = next((d for d in espeak_candidates if d.is_dir()), None)
                if espeak_dir is not None:
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
            if IS_FROZEN:
                raise RuntimeError(f"Piper voice synthesis failed: {exc}") from exc
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
            _play_wav_on_host(wav_bytes)

        return wav_bytes


def stop_read_aloud() -> None:
    try:
        sd.stop()
    except Exception:
        pass


def _normalize_openai_tts_urls(raw_url: str) -> Tuple[str, List[str]]:
    """Return (speech_post_url, voice_list_candidate_urls) for an OpenAI-compatible TTS server."""
    raw = str(raw_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("TTS server URL is empty.")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("TTS server URL must start with http:// or https://")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/audio/speech"):
        speech_url = raw
        v1_base = raw[: -len("/audio/speech")]
    elif path.endswith("/audio/speech"):
        speech_url = raw
        v1_base = raw[: -len("/audio/speech")]
    elif path.endswith("/v1"):
        speech_url = f"{raw}/audio/speech"
        v1_base = raw
    else:
        speech_url = f"{raw}/v1/audio/speech"
        v1_base = f"{raw}/v1"

    voice_urls = [f"{v1_base}/audio/voices", f"{v1_base}/voices"]
    return speech_url, voice_urls


def _split_text_for_tts(text: str, max_chars: int = 3000) -> List[str]:
    """Split long speech text along sentence/newline boundaries without exceeding server limits."""
    clean = str(text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    sentences = re.split(r"(?<=[.!?])\s+|\n+", clean)
    chunks: List[str] = []
    cur = ""
    for sent in sentences:
        part = sent.strip()
        if not part:
            continue
        while len(part) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            cut = part[:max_chars].rsplit(" ", 1)[0].strip() or part[:max_chars]
            chunks.append(cut)
            part = part[len(cut):].strip()
        if not part:
            continue
        cand = f"{cur} {part}".strip() if cur else part
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = part
    if cur:
        chunks.append(cur)
    return chunks


def scan_openai_tts_voices(
    api_url: str,
    api_key: str = "",
    timeout: float = 4.0,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Query an OpenAI/VoiceStudio-compatible local TTS server for available voice profiles."""
    import requests

    try:
        _speech_url, voice_urls = _normalize_openai_tts_urls(api_url)
    except ValueError as exc:
        return [], str(exc)

    headers: Dict[str, str] = {}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_err: Optional[str] = None
    for vurl in voice_urls:
        try:
            resp = requests.get(vurl, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            payload = resp.json()
            raw_list = payload.get("voices") if isinstance(payload, dict) else payload
            if not isinstance(raw_list, list):
                continue

            profiles: List[Dict[str, str]] = []
            presets: List[Dict[str, str]] = []
            seen_ids: Set[str] = set()
            for item in raw_list:
                if isinstance(item, str):
                    vid = item.strip()
                    if vid and vid not in seen_ids:
                        seen_ids.add(vid)
                        presets.append({"id": vid, "label": vid})
                elif isinstance(item, dict):
                    vid = str(item.get("voice_id") or item.get("id") or item.get("name") or "").strip()
                    name = str(item.get("name") or vid).strip()
                    vtype = str(item.get("type") or "").strip().lower()
                    if not vid or vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                    if vtype == "profile":
                        profiles.append({"id": vid, "label": f"{name} (Cloned Profile)"})
                    else:
                        label = f"{name} (Built-in)" if vtype == "openai_alias" else name
                        presets.append({"id": vid, "label": label})
            return profiles + presets, None
        except requests.exceptions.ConnectionError:
            last_err = f"Could not reach TTS server at {api_url} — open VoiceStudio (or start your TTS server) first."
        except Exception as exc:
            last_err = f"Voice lookup failed: {exc}"

    return [], last_err or f"No /v1/audio/voices endpoint found at {api_url}"


def read_aloud_openai_api(
    text: str,
    voice_name: str,
    api_url: str,
    api_model: str = "tts-1",
    api_key: str = "",
) -> bytes:
    """Synthesize speech via an OpenAI-compatible local TTS endpoint (e.g. VoiceStudio on :3900)."""
    import numpy as np
    import requests

    cleaned_text = _clean_text_for_speech(text)
    if not cleaned_text:
        raise ValueError("No speakable text in message.")

    speech_url, _ = _normalize_openai_tts_urls(api_url)
    chunks = _split_text_for_tts(cleaned_text, max_chars=3000)
    if not chunks:
        raise ValueError("No speakable text in message.")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    chosen_model = str(api_model or "tts-1").strip() or "tts-1"
    chosen_voice = str(voice_name or "default").strip() or "default"

    arrays: List[Any] = []
    target_sr: Optional[int] = None
    raw_first: Optional[bytes] = None

    for chunk in chunks:
        payload = {
            "model": chosen_model,
            "input": chunk,
            "voice": chosen_voice,
            "response_format": "wav",
        }
        try:
            resp = requests.post(speech_url, json=payload, headers=headers, timeout=(6, 180))
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not connect to external TTS server ({speech_url}). Make sure VoiceStudio is running."
            ) from exc

        if not resp.ok:
            msg = f"External TTS returned HTTP {resp.status_code}"
            try:
                err_json = resp.json()
                if isinstance(err_json, dict):
                    err_obj = err_json.get("error")
                    if isinstance(err_obj, dict) and err_obj.get("message"):
                        msg = str(err_obj["message"])
                    elif err_json.get("detail"):
                        msg = str(err_json["detail"])
            except Exception:
                pass
            raise RuntimeError(msg)

        if not resp.content:
            raise RuntimeError("External TTS server returned empty audio.")
        if raw_first is None:
            raw_first = resp.content

        data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        if target_sr is None:
            target_sr = int(sr)
        elif int(sr) != target_sr and data.size > 0:
            n_out = max(1, int(round(len(data) * target_sr / int(sr))))
            data = np.interp(np.linspace(0, len(data) - 1, n_out), np.arange(len(data)), data).astype(np.float32)
        arrays.append(data)

    if len(chunks) == 1 and raw_first is not None and raw_first[:4] == b"RIFF":
        return raw_first

    pause = np.zeros(int((target_sr or 24000) * 0.15), dtype=np.float32)
    stitched: List[Any] = []
    for idx, arr in enumerate(arrays):
        if idx > 0:
            stitched.append(pause)
        stitched.append(arr)
    combined = np.concatenate(stitched) if stitched else np.zeros((0,), dtype=np.float32)
    out_buf = io.BytesIO()
    sf.write(out_buf, combined, target_sr or 24000, format="WAV", subtype="PCM_16")
    return out_buf.getvalue()


_STT_ENGINE: Optional["_WhisperSTT"] = None
_STT_LOCK = threading.Lock()
_STT_REQUIRED_FILES = (
    ("onnx/encoder_model_int8.onnx", "encoder_model_int8.onnx"),
    ("onnx/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
    ("generation_config.json", "generation_config.json"),
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
        repo_id = "onnx-community/whisper-base.en"
        log.info(f"Downloading offline Whisper-Base.en ONNX speech recognition model ({repo_id})...")
        for remote_name, local_name in _STT_REQUIRED_FILES:
            cached = Path(hf_hub_download(repo_id=repo_id, filename=remote_name))
            shutil.copy2(cached, target_dir / local_name)
        return target_dir
    except Exception as exc:
        raise RuntimeError(
            f"Offline speech recognition model is not present in {target_dir} and could not be downloaded: {exc}"
        ) from exc


class _WhisperSTT:
    """Pure ONNX Runtime + tokenizers offline Speech-to-Text (Whisper-Base.en int8 with KV cache, CPU, 0 VRAM)."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 16000 * 30  # 30s Whisper encoder window

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
            str(resolved / "decoder_model_merged_int8.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._enc_input_name = self.encoder.get_inputs()[0].name
        self._dec_out_names = [o.name for o in self.decoder.get_outputs()]
        self.tokenizer = Tokenizer.from_file(str(resolved / "tokenizer.json"))

        cfg = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        gen_cfg = json.loads((resolved / "generation_config.json").read_text(encoding="utf-8"))
        self.n_mels = int(cfg.get("num_mel_bins") or 80)
        self.n_layers = int(cfg.get("decoder_layers") or cfg.get("num_hidden_layers") or 6)
        self.n_heads = int(cfg.get("decoder_attention_heads") or 8)
        d_model = int(cfg.get("d_model") or 512)
        self.head_dim = max(1, d_model // self.n_heads)
        self.max_steps = min(224, max(64, int(cfg.get("max_target_positions") or 448) - 8))

        self.start_id = int(gen_cfg.get("decoder_start_token_id") or cfg.get("decoder_start_token_id") or 50257)
        self.eos_id = int(gen_cfg.get("eos_token_id") or cfg.get("eos_token_id") or 50256)
        self.no_ts_id = int(gen_cfg.get("no_timestamps_token_id") or 50362)
        self.init_ids = [self.start_id, self.no_ts_id]

        suppress = [int(t) for t in (gen_cfg.get("suppress_tokens") or [])]
        begin_suppress = [int(t) for t in (gen_cfg.get("begin_suppress_tokens") or [])]
        self.suppress_ids = np.array(suppress, dtype=np.int64) if suppress else np.zeros((0,), dtype=np.int64)
        self.begin_suppress_ids = np.array(begin_suppress, dtype=np.int64) if begin_suppress else np.zeros((0,), dtype=np.int64)

        self._mel_filters = self._build_mel_filters(self.SAMPLE_RATE, 400, self.n_mels)
        self._hann = np.hanning(401)[:-1].astype(np.float32)

    def _build_mel_filters(self, sr: int, n_fft: int, n_mels: int) -> Any:
        np = self._np

        def _hz_to_mel(freq: Any) -> Any:
            f_min, f_sp = 0.0, 200.0 / 3.0
            min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
            logstep = np.log(6.4) / 27.0
            if np.ndim(freq) == 0:
                return min_log_mel + np.log(freq / min_log_hz) / logstep if freq >= min_log_hz else (freq - f_min) / f_sp
            out = (freq - f_min) / f_sp
            mask = freq >= min_log_hz
            out[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
            return out

        def _mel_to_hz(mel: Any) -> Any:
            f_min, f_sp = 0.0, 200.0 / 3.0
            min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
            logstep = np.log(6.4) / 27.0
            out = f_min + f_sp * mel
            mask = mel >= min_log_mel
            out[mask] = min_log_hz * np.exp(logstep * (mel[mask] - min_log_mel))
            return out

        fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
        mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), n_mels + 2)
        f_pts = _mel_to_hz(mels)
        slopes = f_pts[None, :] - fft_freqs[:, None]
        down = -slopes[:, :-2] / (f_pts[1:-1] - f_pts[:-2])[None, :]
        up = slopes[:, 2:] / (f_pts[2:] - f_pts[1:-1])[None, :]
        fb = np.maximum(0.0, np.minimum(down, up))
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm[None, :]
        return fb.T.astype(np.float32)

    def _compute_log_mel(self, audio_16k: Any) -> Any:
        np = self._np
        padded = np.zeros(self.CHUNK_SAMPLES, dtype=np.float32)
        take = min(len(audio_16k), self.CHUNK_SAMPLES)
        padded[:take] = audio_16k[:take]
        x = np.pad(padded, (200, 200), mode="reflect")
        n_frames = 3000
        strides = (x.strides[0] * 160, x.strides[0])
        frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, 400), strides=strides)
        mag = np.abs(np.fft.rfft(frames * self._hann[None, :], n=400, axis=1)) ** 2
        mel = self._mel_filters @ mag.T
        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return np.ascontiguousarray(log_spec[None, :, :], dtype=np.float32)

    def _decode_window(self, window_16k: Any) -> str:
        np = self._np
        feats = self._compute_log_mel(window_16k)
        enc_hidden = self.encoder.run(None, {self._enc_input_name: feats})[0]

        empty_kv = np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)
        past: Dict[str, Any] = {}
        for idx in range(self.n_layers):
            past[f"past_key_values.{idx}.decoder.key"] = empty_kv
            past[f"past_key_values.{idx}.decoder.value"] = empty_kv
            past[f"past_key_values.{idx}.encoder.key"] = empty_kv
            past[f"past_key_values.{idx}.encoder.value"] = empty_kv

        cur_ids = np.array([self.init_ids], dtype=np.int64)
        use_cache = np.array([False], dtype=bool)
        tokens: List[int] = []
        for step in range(self.max_steps):
            outs = self.decoder.run(
                None,
                {
                    "input_ids": cur_ids,
                    "encoder_hidden_states": enc_hidden,
                    "use_cache_branch": use_cache,
                    **past,
                },
            )
            logits = outs[0][0, -1, :].copy()
            if self.suppress_ids.size > 0:
                logits[self.suppress_ids] = -np.inf
            if step == 0 and self.begin_suppress_ids.size > 0:
                logits[self.begin_suppress_ids] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == self.eos_id:
                break
            if len(tokens) >= 4 and tokens[-4:] == [nxt, nxt, nxt, nxt]:
                break
            tokens.append(nxt)

            for out_idx, out_name in enumerate(self._dec_out_names[1:], start=1):
                past_name = out_name.replace("present.", "past_key_values.", 1)
                if step == 0 or ".decoder." in out_name:
                    past[past_name] = outs[out_idx]
            cur_ids = np.array([[nxt]], dtype=np.int64)
            use_cache = np.array([True], dtype=bool)

        return re.sub(r"\s+", " ", self.tokenizer.decode(tokens, skip_special_tokens=True)).strip()

    def _preprocess_audio(self, audio: Any, sample_rate: int) -> Any:
        np = self._np
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return np.zeros((0,), dtype=np.float32)

        sr = max(8000, int(sample_rate or self.SAMPLE_RATE))
        # Zero-phase raised-cosine high-pass removes DC offset and tabletop/fan rumble below 95Hz completely
        if len(arr) >= 64:
            spec = np.fft.rfft(arr)
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / float(sr))
            gain = np.ones_like(freqs, dtype=np.float32)
            gain[freqs <= 65.0] = 0.0
            trans = (freqs > 65.0) & (freqs < 95.0)
            gain[trans] = 0.5 * (1.0 - np.cos(np.pi * (freqs[trans] - 65.0) / 30.0))
            arr = np.fft.irfft(spec * gain, n=len(arr)).astype(np.float32)
        else:
            arr = (arr - float(np.median(arr))).astype(np.float32)

        # Anti-alias low-pass before decimation when capturing at 44.1kHz / 48kHz
        if sr != self.SAMPLE_RATE:
            if sr > 17500 and len(arr) >= 31:
                fc = 7200.0 / float(sr)
                taps = np.arange(-15, 16, dtype=np.float32)
                sinc = (2.0 * fc) * np.sinc(2.0 * fc * taps).astype(np.float32)
                lp = (sinc * np.hanning(31)).astype(np.float32)
                lp /= np.sum(lp)
                arr = np.convolve(arr, lp, mode="same").astype(np.float32)
            n_out = max(1, int(round(len(arr) * self.SAMPLE_RATE / sr)))
            arr = np.interp(np.linspace(0, len(arr) - 1, n_out), np.arange(len(arr)), arr).astype(np.float32)
        return arr

    def transcribe_float32(self, audio: Any, sample_rate: int = 16000) -> str:
        np = self._np
        arr = self._preprocess_audio(audio, sample_rate)
        if len(arr) < int(0.12 * self.SAMPLE_RATE):
            return ""

        peak = float(np.max(np.abs(arr)))
        if peak < 0.002:
            return ""

        # Compute 50ms frame RMS to reject flat stationary room hum before calling Whisper
        hop = 800
        n_frames = max(1, len(arr) // hop)
        trimmed_frames = arr[: n_frames * hop].reshape(n_frames, hop)
        frame_rms = np.sqrt(np.mean(trimmed_frames * trimmed_frames, axis=1))
        p95_rms = float(np.percentile(frame_rms, 95))
        p10_rms = float(np.percentile(frame_rms, 10))
        if p95_rms < 0.0028 or (p95_rms < 0.006 and (p95_rms / max(1e-6, p10_rms)) < 1.55):
            return ""

        gate = max(0.0022, p95_rms * 0.12)
        active = np.where(np.abs(arr) >= gate)[0]
        if active.size == 0:
            return ""
        pad = int(0.35 * self.SAMPLE_RATE)
        start = max(0, int(active[0]) - pad)
        end = min(len(arr), int(active[-1]) + pad + 1)
        trimmed = np.ascontiguousarray(arr[start:end], dtype=np.float32)
        if len(trimmed) < int(0.12 * self.SAMPLE_RATE):
            return ""

        # RMS-based gain normalization boosts quiet speech without letting key taps suppress voice
        if p95_rms < 0.08:
            gain = min(12.0, 0.09 / max(1e-5, p95_rms))
            trimmed = np.ascontiguousarray(np.clip(trimmed * gain, -0.98, 0.98), dtype=np.float32)

        parts: List[str] = []
        pos = 0
        total = len(trimmed)
        while pos < total:
            if total - pos <= self.CHUNK_SAMPLES:
                window = trimmed[pos:total]
                pos = total
            else:
                # Split cleanly at the quietest pause between 23s and 29.5s
                lo = pos + int(23.0 * self.SAMPLE_RATE)
                hi = min(total, pos + int(29.5 * self.SAMPLE_RATE))
                seg = trimmed[lo:hi]
                if len(seg) >= hop:
                    nf = len(seg) // hop
                    f_e = np.mean(seg[: nf * hop].reshape(nf, hop) ** 2, axis=1)
                    split_idx = lo + int(np.argmin(f_e)) * hop
                else:
                    split_idx = pos + self.CHUNK_SAMPLES
                window = trimmed[pos:split_idx]
                pos = split_idx
            if len(window) < int(0.12 * self.SAMPLE_RATE):
                continue
            txt = self._decode_window(window)
            if txt:
                parts.append(txt)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _get_stt_engine() -> _WhisperSTT:
    global _STT_ENGINE
    with _STT_LOCK:
        if _STT_ENGINE is None:
            _STT_ENGINE = _WhisperSTT()
        return _STT_ENGINE


def scan_input_devices() -> List[Dict[str, Any]]:
    """Return available PC microphone input devices, deduped across host APIs."""
    devices_out: List[Dict[str, Any]] = []
    try:
        all_devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    except Exception as exc:
        log.warning(f"Could not query audio input devices: {exc}")
        return devices_out

    seen_names: Set[str] = set()
    # Prefer MME (0) then WASAPI / DirectSound over exotic WDM-KS raw endpoints
    ordered: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, dev in enumerate(all_devs):
        if int(dev.get("max_input_channels") or 0) <= 0:
            continue
        api_idx = int(dev.get("hostapi") or 0)
        api_name = str(hostapis[api_idx].get("name") or "") if 0 <= api_idx < len(hostapis) else ""
        prio = 0 if idx == default_in else (1 if "MME" in api_name else (2 if "WASAPI" in api_name else 3))
        ordered.append((prio, idx, dev))
    ordered.sort(key=lambda t: (t[0], t[1]))

    for _prio, idx, dev in ordered:
        raw_name = re.sub(r"\s+", " ", str(dev.get("name") or f"Microphone {idx}")).strip()
        if not raw_name or raw_name.lower() in ("input ()", "microsoft sound mapper - input", "primary sound capture driver"):
            continue
        key = raw_name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        devices_out.append({
            "id": str(idx),
            "name": raw_name,
            "default": idx == default_in,
            "default_samplerate": int(round(float(dev.get("default_samplerate") or 44100))),
            "channels": int(dev.get("max_input_channels") or 1),
        })
    return devices_out


def _resolve_input_device(device_pref: str = "auto") -> Tuple[Optional[int], Dict[str, Any]]:
    """Resolve 'auto', numeric device index, or device name substring into (device_idx, info)."""
    pref = str(device_pref or "auto").strip()
    try:
        all_devs = sd.query_devices()
    except Exception:
        return None, {"name": "Default Microphone", "default_samplerate": 16000, "max_input_channels": 1}

    if pref and pref.lower() != "auto":
        if pref.isdigit():
            idx = int(pref)
            if 0 <= idx < len(all_devs) and int(all_devs[idx].get("max_input_channels") or 0) > 0:
                return idx, dict(all_devs[idx])
        pref_low = pref.lower()
        for idx, dev in enumerate(all_devs):
            if int(dev.get("max_input_channels") or 0) > 0:
                dname = str(dev.get("name") or "").strip().lower()
                if dname == pref_low:
                    return idx, dict(dev)
        for idx, dev in enumerate(all_devs):
            if int(dev.get("max_input_channels") or 0) > 0:
                dname = str(dev.get("name") or "").strip().lower()
                if pref_low in dname:
                    return idx, dict(dev)

    try:
        dev = sd.query_devices(kind="input") or {}
        default_idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        idx_val = int(default_idx) if isinstance(default_idx, int) and default_idx >= 0 else None
        return idx_val, dict(dev)
    except Exception:
        return None, {"name": "Default Microphone", "default_samplerate": 16000, "max_input_channels": 1}


class _HostAudioRecorder:
    SAMPLE_RATE = 16000
    MAX_SECONDS = 90

    def __init__(self):
        self._buf_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._stream: Optional[Any] = None
        self._chunks: List[Any] = []
        self._total_samples = 0
        self._max_samples = self.SAMPLE_RATE * self.MAX_SECONDS
        self.recording = False
        self.device_name = ""
        self.capture_sample_rate = self.SAMPLE_RATE

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

    def start(self, device_pref: str = "auto") -> str:
        import numpy as np
        self._stop_and_close_stream()
        with self._buf_lock:
            self._chunks = []
            self._total_samples = 0

        dev_idx, dev_info = _resolve_input_device(device_pref)
        dev_name = str((dev_info or {}).get("name") or "Default Microphone").strip()
        native_sr = int(round(float((dev_info or {}).get("default_samplerate") or 44100)))
        max_ch = max(1, int((dev_info or {}).get("max_input_channels") or 1))

        # Prefer device native sample rate first (avoids Windows MME 16kHz driver quirks), then standard rates
        sr_candidates: List[int] = []
        for sr in (native_sr, 48000, 44100, 16000):
            if sr and sr >= 8000 and sr not in sr_candidates:
                sr_candidates.append(sr)
        ch_candidates = [1] + ([2] if max_ch >= 2 else [])

        def _callback(indata, frames, time_info, status):
            if not self.recording:
                return
            with self._buf_lock:
                if not self.recording or self._total_samples >= self._max_samples:
                    return
                take = min(frames, self._max_samples - self._total_samples)
                if take > 0:
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = np.mean(indata[:take], axis=1, dtype=np.float32)
                    else:
                        mono = np.array(indata[:take, 0], dtype=np.float32, copy=True)
                    self._chunks.append(mono)
                    self._total_samples += take

        with self._stream_lock:
            self.device_name = dev_name
            last_exc: Optional[Exception] = None
            opened = False
            for ch in ch_candidates:
                if opened:
                    break
                for sr in sr_candidates:
                    self.capture_sample_rate = sr
                    self._max_samples = sr * self.MAX_SECONDS
                    self.recording = True
                    try:
                        stream = sd.InputStream(
                            device=dev_idx,
                            samplerate=sr,
                            channels=ch,
                            dtype="float32",
                            callback=_callback,
                        )
                        self._stream = stream
                        stream.start()
                        opened = True
                        log.info(f"Mic start: device={dev_name!r} sr={sr}Hz ch={ch}")
                        break
                    except Exception as exc:
                        last_exc = exc
                        self.recording = False
                        bad_stream = self._stream
                        self._stream = None
                        if bad_stream is not None:
                            try:
                                bad_stream.close()
                            except Exception:
                                pass
            if not opened:
                raise RuntimeError(f"Could not open host microphone ({dev_name}): {last_exc}") from last_exc
        return dev_name

    def stop_and_extract(self) -> Tuple[Any, float]:
        import numpy as np
        self._stop_and_close_stream()
        with self._buf_lock:
            chunks = list(self._chunks)
            self._chunks = []
            self._total_samples = 0
        sr = int(self.capture_sample_rate or self.SAMPLE_RATE)
        if not chunks:
            return np.zeros((0,), dtype=np.float32), 0.0
        samples = np.concatenate(chunks).astype(np.float32)
        return samples, round(len(samples) / max(1, sr), 2)

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
            STATE.auto_archived_signatures.clear()
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
            STATE.loading_model = None


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


_SPLIT_GGUF_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def _parse_split_gguf_name(filename: str) -> Optional[Tuple[str, int, int]]:
    match = _SPLIT_GGUF_RE.match(Path(str(filename or "")).name)
    if not match:
        return None
    part = int(match.group("part"))
    total = int(match.group("total"))
    if part < 1 or total < 2 or part > total:
        return None
    return match.group("prefix"), part, total


def _resolve_model_parts(filename: str) -> Tuple[List[Path], int]:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    split_info = _parse_split_gguf_name(path.name)
    if not split_info:
        return [path], 1
    prefix, part, total = split_info
    if part != 1:
        first_name = f"{prefix}-00001-of-{total:05d}.gguf"
        if (MODEL_DIR / first_name).is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Select the first file '{first_name}' to load this split GGUF.",
            )
    parts: List[Path] = []
    missing: List[str] = []
    for idx in range(1, total + 1):
        part_name = f"{prefix}-{idx:05d}-of-{total:05d}.gguf"
        part_path = MODEL_DIR / part_name
        if part_path.is_file():
            parts.append(part_path)
        else:
            missing.append(part_name)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Incomplete split GGUF ({len(parts)}/{total} parts found). Missing: {', '.join(missing[:3])}",
        )
    return parts, total


def scan_models() -> List[Dict[str, Any]]:
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_files = sorted(MODEL_DIR.glob("*.gguf"))
    split_groups: Dict[Tuple[str, int], Dict[int, Path]] = {}
    for f in all_files:
        if _is_mmproj_filename(f.name):
            continue
        split_info = _parse_split_gguf_name(f.name)
        if split_info:
            prefix, part, total = split_info
            split_groups.setdefault((prefix.lower(), total), {})[part] = f

    models = []
    seen_splits: Set[Tuple[str, int]] = set()
    for f in all_files:
        if _is_mmproj_filename(f.name):
            continue
        split_info = _parse_split_gguf_name(f.name)
        part_count = 1
        missing_parts = 0
        if split_info:
            prefix, part, total = split_info
            key = (prefix.lower(), total)
            if key in seen_splits:
                continue
            parts_map = split_groups.get(key, {})
            # Only represent a split GGUF once (via part 00001, or lowest present part)
            first_part = min(parts_map.keys()) if parts_map else part
            if part != first_part:
                continue
            seen_splits.add(key)
            part_count = total
            missing_parts = max(0, total - len(parts_map))
            file_size_bytes = 0
            for part_path in parts_map.values():
                try:
                    file_size_bytes += part_path.stat().st_size
                except OSError:
                    pass
        else:
            try:
                file_size_bytes = f.stat().st_size
            except OSError:
                file_size_bytes = 0

        try:
            with open(f, "rb") as hdr:
                valid_gguf = hdr.read(4) == b"GGUF"
        except OSError:
            valid_gguf = False

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
            "part_count": part_count,
            "missing_parts": missing_parts,
            "valid_gguf": valid_gguf,
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


def _has_nvidia_smi() -> bool:
    if shutil.which("nvidia-smi") is not None:
        return True
    return Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe").is_file()


def _probe_system_ram() -> Tuple[int, int]:
    """Return (total_ram_bytes, avail_ram_bytes) across Windows, macOS, and Linux."""
    if os.name == "nt":
        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        try:
            ms = _MEMORYSTATUSEX(dwLength=ctypes.sizeof(_MEMORYSTATUSEX))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys), int(ms.ullAvailPhys)
        except Exception:
            return 0, 0
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            total = page_size * total_pages
            avail = total
            if "SC_AVPHYS_PAGES" in os.sysconf_names:
                avail = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
            return total, avail
        except Exception:
            pass
    return 0, 0


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


def _probe_ggml_gpu_devices() -> List[Dict[str, Any]]:
    """Probe Vulkan, CUDA, or Metal GPU devices directly via loaded ggml backend APIs."""
    try:
        ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
        base_candidates = [
            RUNTIME_LLAMA_LIB_DIR / f"ggml-base{ext}",
            RUNTIME_LLAMA_LIB_DIR / f"libggml-base{ext}",
        ]
        ggml_candidates = [
            RUNTIME_LLAMA_LIB_DIR / f"ggml{ext}",
            RUNTIME_LLAMA_LIB_DIR / f"libggml{ext}",
        ]
        base_path = next((p for p in base_candidates if p.is_file()), None)
        ggml_path = next((p for p in ggml_candidates if p.is_file()), None)
        if base_path is None or ggml_path is None:
            return []
        gb = ctypes.CDLL(str(base_path))
        gg = ctypes.CDLL(str(ggml_path))
        if hasattr(gg, "ggml_backend_load_all_from_path"):
            gg.ggml_backend_load_all_from_path(str(RUNTIME_LLAMA_LIB_DIR).encode("utf-8"))
        gg.ggml_backend_dev_count.restype = ctypes.c_size_t
        gg.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        gg.ggml_backend_dev_get.restype = ctypes.c_void_p
        gb.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
        gb.ggml_backend_dev_name.restype = ctypes.c_char_p
        gb.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
        gb.ggml_backend_dev_description.restype = ctypes.c_char_p
        gb.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        gb.ggml_backend_dev_type.restype = ctypes.c_int
        gb.ggml_backend_dev_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        devices: List[Dict[str, Any]] = []
        count = int(gg.ggml_backend_dev_count())
        for idx in range(count):
            dev = gg.ggml_backend_dev_get(idx)
            if not dev:
                continue
            dev_type = int(gb.ggml_backend_dev_type(dev))
            # ggml_backend_dev_type: 0=CPU, 1=GPU, 2=ACCEL, 3=IGPU
            if dev_type not in (1, 3):
                continue
            free_mem = ctypes.c_size_t(0)
            total_mem = ctypes.c_size_t(0)
            gb.ggml_backend_dev_memory(dev, ctypes.byref(free_mem), ctypes.byref(total_mem))
            if int(total_mem.value) <= 0:
                continue
            desc = gb.ggml_backend_dev_description(dev) or gb.ggml_backend_dev_name(dev) or b"GPU"
            devices.append({
                "name": desc.decode("utf-8", errors="replace").strip() or "GPU",
                "total_bytes": int(total_mem.value),
                "free_bytes": int(free_mem.value),
                "is_igpu": dev_type == 3,
            })
        if len(devices) > 1 and any(d["total_bytes"] >= 4 * GIB for d in devices):
            filtered = [d for d in devices if not d.get("is_igpu") and d["total_bytes"] >= 2 * GIB]
            if filtered:
                devices = filtered
        return devices
    except Exception:
        return []


def _probe_gpu_devices() -> List[Dict[str, Any]]:
    if BACKEND_NAME == "cpu" or not llama_cpp.llama_supports_gpu_offload():
        return []
    if BACKEND_NAME == "cuda":
        devices = _probe_nvidia_gpus()
        if devices or _has_nvidia_smi():
            return devices
        return _probe_ggml_gpu_devices()
    devices = _probe_ggml_gpu_devices()
    if devices:
        return devices
    if sys.platform == "darwin":
        total_ram, avail_ram = _probe_system_ram()
        if total_ram > 0:
            return [{"name": "Apple Metal Unified Memory", "total_bytes": int(total_ram * 0.75), "free_bytes": int(avail_ram * 0.85)}]
    return []


def _measure_free_vram_bytes() -> int:
    try:
        return sum(max(0, int(d.get("free_bytes") or 0)) for d in _probe_gpu_devices())
    except Exception:
        return 0


def _compute_ssm_state_reserve(reader: GGUFReader, architecture: str, ssm_layer_count: int) -> int:
    """Estimate conservative VRAM state reserve for Mamba / Gated-DeltaNet / hybrid SSM layers."""
    if ssm_layer_count <= 0:
        return 0
    inner_size = int(_gguf_value(reader, f"{architecture}.ssm.inner_size") or 4096)
    state_size = int(_gguf_value(reader, f"{architecture}.ssm.state_size") or 128)
    conv_kernel = int(_gguf_value(reader, f"{architecture}.ssm.conv_kernel") or 4)
    group_count = int(_gguf_value(reader, f"{architecture}.ssm.group_count") or 16)
    recurrent_bytes = inner_size * state_size * 4
    conv_bytes = max(0, conv_kernel - 1) * (inner_size + 2 * group_count * state_size) * 4
    # Include workspace / scratch headroom per recurrent layer without over-blocking GPU offload
    per_layer_bytes = max(20 * 1024 ** 2, min(64 * 1024 ** 2, (recurrent_bytes + conv_bytes) * 4 + 12 * 1024 ** 2))
    return ssm_layer_count * per_layer_bytes


def preflight_model(
    filename: str,
    context_size: int,
    mmproj_model: str = "auto",
    credit_vram_bytes: int = 0,
) -> dict:
    model_parts, part_count = _resolve_model_parts(filename)
    path = model_parts[0]
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
        part_readers = [reader] + [GGUFReader(s) for s in model_parts[1:]]
        tensor_parameter_count = sum(int(tensor.n_elements) for pr in part_readers for tensor in pr.tensors)
        parameter_count = int(metadata_parameter_count or tensor_parameter_count)
        weight_bytes = sum(int(tensor.n_bytes) for pr in part_readers for tensor in pr.tensors)
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
    token_embd_bytes = 0
    key_layers: Set[int] = set()
    value_layers: Set[int] = set()
    fused_qkv_layers: Set[int] = set()
    attn_other_layers: Set[int] = set()
    ssm_layers: Set[int] = set()
    has_ssm_state = any(key.startswith(f"{architecture}.ssm.") for key in reader.fields)
    for pr in part_readers:
        for tensor in pr.tensors:
            parts = tensor.name.split(".")
            n_bytes = int(tensor.n_bytes)
            if len(parts) >= 3 and parts[0] == "blk":
                try:
                    index = int(parts[1])
                except ValueError:
                    non_layer_bytes += n_bytes
                    continue
                if 0 <= index < layer_count:
                    layer_weight_bytes[index] += n_bytes
                    if len(parts) >= 4:
                        comp = parts[2]
                        if comp == "attn_k" and parts[3] == "weight":
                            key_layers.add(index)
                        elif comp == "attn_v" and parts[3] == "weight":
                            value_layers.add(index)
                        elif comp == "attn_qkv" and parts[3] == "weight":
                            fused_qkv_layers.add(index)
                        elif comp.startswith("attn_"):
                            attn_other_layers.add(index)
                        elif "ssm" in comp:
                            ssm_layers.add(index)
                            has_ssm_state = True
                else:
                    non_layer_bytes += n_bytes
            else:
                non_layer_bytes += n_bytes
                if "token_embd" in tensor.name:
                    token_embd_bytes += n_bytes

    # llama.cpp keeps token_embd mapped on CPU RAM rather than VRAM unless row-split forced
    gpu_non_layer_bytes = max(0, non_layer_bytes - token_embd_bytes)
    if gpu_non_layer_bytes == 0 and non_layer_bytes > 0:
        gpu_non_layer_bytes = int(non_layer_bytes * 0.5)

    attention_layers = key_layers & value_layers
    if not attention_layers and fused_qkv_layers:
        attention_layers = fused_qkv_layers
    if not attention_layers:
        inferred = attn_other_layers - ssm_layers
        if inferred:
            attention_layers = inferred
        elif not has_ssm_state:
            attention_layers = set(range(layer_count))

    ssm_layer_count = len(ssm_layers) if ssm_layers else (max(0, layer_count - len(attention_layers)) if has_ssm_state else 0)
    ssm_state_bytes = _compute_ssm_state_reserve(reader, architecture, ssm_layer_count) if has_ssm_state else 0

    warnings: List[str] = []
    if part_count > 1:
        warnings.append(f"Split GGUF model ({part_count} parts summed for total weights).")

    if kv_metadata_supported and attention_layers:
        eff_attn_layers = len(attention_layers)
        eff_kv_heads = kv_head_count
        eff_k_len = key_length
        eff_v_len = value_length
        kv_method = "hybrid_ssm_kv" if has_ssm_state else ("swa_full_kv" if sliding_window else "dense_gqa_fp16")
    else:
        eff_attn_layers = len(attention_layers) or (0 if has_ssm_state and not attention_layers else layer_count)
        eff_kv_heads = kv_head_count or head_count or 8
        eff_k_len = key_length or ((embedding_length // max(1, head_count)) if (embedding_length and head_count) else 128)
        eff_v_len = value_length or eff_k_len
        kv_method = "conservative_estimate_fp16"
        warnings.append("Attention KV metadata is partial; using conservative VRAM sizing.")

    kv_f16_bytes = (
        eff_attn_layers
        * context_size
        * eff_kv_heads
        * (eff_k_len + eff_v_len)
        * 2
    )
    kv_q8_bytes = (kv_f16_bytes * 34 + 63) // 64
    kv_q4_bytes = (kv_f16_bytes * 18 + 63) // 64

    mmproj_reserve_bytes = int(mmproj_bytes * 1.15) if paired_mmproj else 0
    compute_reserve_bytes = int(0.5 * GIB + 0.04 * weight_bytes) + mmproj_reserve_bytes + ssm_state_bytes
    devices = _probe_gpu_devices()
    if paired_mmproj:
        warnings.append(f"Paired vision projector '{paired_mmproj}' (+{round(mmproj_reserve_bytes / GIB, 2)} GiB VRAM reserve).")
    if has_ssm_state:
        warnings.append(
            f"Hybrid SSM/recurrent layout ({eff_attn_layers} attention + {ssm_layer_count} SSM layers; "
            f"+{round(ssm_state_bytes / GIB, 2)} GiB state reserve)."
        )
    if sliding_window is not None:
        warnings.append(f"Sliding-window attention ({sliding_window}); budgeting full KV cache for stability.")

    gpu_layers = 0
    gpu_weight_bytes = 0
    kv_cache_bytes = kv_f16_bytes
    kv_type = int(llama_cpp.GGML_TYPE_F16)
    kv_cache_type = "F16"
    offload_kqv = True
    usable_bytes = 0
    gpu_name = None
    total_bytes = 0
    free_bytes = 0
    ram_total_bytes, ram_avail_bytes = _probe_system_ram()

    if BACKEND_NAME == "cpu" or not llama_cpp.llama_supports_gpu_offload():
        offload_kqv = False
        warnings.append("CPU backend selected; model will run in system RAM.")
    elif not devices:
        offload_kqv = False
        warnings.append("No compatible GPU reported; selecting CPU + system RAM loading.")
    else:
        total_bytes = sum(int(d["total_bytes"]) for d in devices)
        raw_free_bytes = sum(int(d["free_bytes"]) for d in devices)
        free_bytes = min(total_bytes, raw_free_bytes + max(0, int(credit_vram_bytes or 0)))
        if len(devices) == 1:
            gpu_name = devices[0]["name"]
        else:
            gpu_name = f"{devices[0]['name']} (+{len(devices) - 1} more GPUs)"
        safety_reserve = GPU_SAFETY_RESERVE_BYTES * len(devices)
        usable_bytes = min(
            int(total_bytes * GPU_MAX_UTILIZATION),
            max(0, free_bytes - safety_reserve),
        )

        def fitting_layers(kv_bytes: int, include_kqv: bool = True) -> Tuple[int, int]:
            kv_vram = kv_bytes if include_kqv else 0
            available = usable_bytes - kv_vram - compute_reserve_bytes - gpu_non_layer_bytes
            for candidate in range(layer_count, 0, -1):
                candidate_bytes = sum(layer_weight_bytes[layer_count - candidate:])
                if candidate_bytes <= available:
                    return candidate, candidate_bytes
            return 0, 0

        f16_layers, f16_weight_bytes = fitting_layers(kv_f16_bytes, include_kqv=True)
        q8_layers, q8_weight_bytes = fitting_layers(kv_q8_bytes, include_kqv=True)
        q4_layers, q4_weight_bytes = fitting_layers(kv_q4_bytes, include_kqv=True)

        gpu_layers = f16_layers
        gpu_weight_bytes = f16_weight_bytes
        if q8_layers > gpu_layers:
            kv_cache_bytes = kv_q8_bytes
            kv_type = int(llama_cpp.GGML_TYPE_Q8_0)
            kv_cache_type = "Q8_0"
            kv_method = f"{kv_method}_q8"
            gpu_layers = q8_layers
            gpu_weight_bytes = q8_weight_bytes
            warnings.append("KV cache set to Q8_0 to fit more layers in VRAM.")
        if gpu_layers < layer_count and q4_layers > gpu_layers:
            kv_cache_bytes = kv_q4_bytes
            kv_type = int(llama_cpp.GGML_TYPE_Q4_0)
            kv_cache_type = "Q4_0"
            kv_method = f"{kv_method}_q4"
            gpu_layers = q4_layers
            gpu_weight_bytes = q4_weight_bytes
            warnings.append("KV cache stepped down to Q4_0 to fit more layers in VRAM.")
        if gpu_layers == 0:
            ram_kv_layers, ram_kv_weight_bytes = fitting_layers(0, include_kqv=False)
            if ram_kv_layers > 0:
                offload_kqv = False
                kv_cache_bytes = kv_f16_bytes
                kv_type = int(llama_cpp.GGML_TYPE_F16)
                kv_cache_type = "F16"
                kv_method = f"{kv_method}_ram_kv"
                gpu_layers = ram_kv_layers
                gpu_weight_bytes = ram_kv_weight_bytes
                warnings.append("VRAM is tight; keeping KV cache in system RAM to maximize GPU layer offload.")
            else:
                offload_kqv = False
                warnings.append("Weights and runtime overhead exceed free VRAM; falling back to system RAM (CPU offload).")

        other_app_vram = max(0, total_bytes - free_bytes)
        if gpu_layers < layer_count and other_app_vram > 2 * GIB:
            warnings.append(
                f"~{other_app_vram / GIB:.2f} GiB VRAM is used by other apps; remaining layers will use hybrid RAM offload."
            )

    if ram_total_bytes > 0 and gpu_layers < layer_count:
        ram_weights = max(0, weight_bytes - gpu_weight_bytes)
        ram_kv = 0 if offload_kqv else kv_cache_bytes
        if (ram_weights + ram_kv) > int(ram_total_bytes * 0.90):
            warnings.append(
                f"Model requires ~{(ram_weights + ram_kv) / GIB:.1f} GiB system RAM beyond VRAM (system total: {ram_total_bytes / GIB:.1f} GiB); loading may swap heavily."
            )

    predicted_gpu_bytes = 0
    if gpu_layers > 0:
        kv_vram_used = kv_cache_bytes if offload_kqv else 0
        predicted_gpu_bytes = gpu_non_layer_bytes + gpu_weight_bytes + kv_vram_used + compute_reserve_bytes

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
        "ssm_layers": ssm_layer_count,
        "part_count": part_count,
        "n_gpu_layers": gpu_layers,
        "gpu_count": len(devices),
        "offload_kqv": offload_kqv,
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
        "system_ram_total_bytes": ram_total_bytes,
        "system_ram_avail_bytes": ram_avail_bytes,
        "mmproj_model": paired_mmproj,
        "mmproj_bytes": mmproj_bytes,
        "vision_capable": bool(paired_mmproj),
        "warnings": warnings,
        "estimate_note": "Tensor bytes read across all GGUF files; KV & SSM reserves scaled to hardware VRAM + RAM.",
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
        STATE.loaded_vram_bytes = 0

        for _ in range(3):
            gc.collect()
        time.sleep(0.4)
        log.info("Resident model flushed. VRAM released.")


def _is_vram_oom_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return any(
        tok in msg
        for tok in ("out of memory", "oom", "cudamalloc", "vk_error_out_of", "failed to allocate", "not enough memory")
    )


def _build_load_attempts(preflight: Dict[str, Any], context_size: int) -> List[Dict[str, Any]]:
    """Build primary attempt plus automatic progressive fallbacks if allocation or arch compat fails."""
    init_ngl = max(0, int(preflight.get("n_gpu_layers") or 0))
    init_kv_type = int(preflight.get("kv_type") or llama_cpp.GGML_TYPE_F16)
    init_kv_name = str(preflight.get("kv_cache_type") or "F16")
    init_offload_kqv = bool(preflight.get("offload_kqv", init_ngl > 0))
    f16_type = int(llama_cpp.GGML_TYPE_F16)
    q8_type = int(llama_cpp.GGML_TYPE_Q8_0)

    attempts: List[Dict[str, Any]] = [
        {
            "n_gpu_layers": init_ngl,
            "kv_type": init_kv_type,
            "kv_cache_type": init_kv_name,
            "flash_attn": bool(FLASH_ATTN and init_ngl > 0),
            "offload_kqv": bool(init_offload_kqv and init_ngl > 0),
            "mmproj_gpu": init_ngl > 0,
            "n_ctx": context_size,
            "reason": None,
        }
    ]
    seen = {
        (init_ngl, init_kv_type, bool(FLASH_ATTN and init_ngl > 0), bool(init_offload_kqv and init_ngl > 0), init_ngl > 0, context_size)
    }

    def add(ngl: int, kv_t: int, kv_n: str, fa: bool, okqv: bool, mm_gpu: bool, ctx: int, reason: str) -> None:
        key = (ngl, kv_t, fa, okqv, mm_gpu, ctx)
        if key not in seen:
            seen.add(key)
            attempts.append({
                "n_gpu_layers": ngl,
                "kv_type": kv_t,
                "kv_cache_type": kv_n,
                "flash_attn": fa,
                "offload_kqv": okqv,
                "mmproj_gpu": mm_gpu,
                "n_ctx": ctx,
                "reason": reason,
            })

    if init_ngl > 0:
        if init_kv_type != f16_type or FLASH_ATTN:
            add(
                init_ngl,
                f16_type,
                "F16",
                False,
                init_offload_kqv,
                True,
                context_size,
                "Retried with F16 KV and flash-attn disabled for architecture compatibility.",
            )
        for frac, tag in ((0.75, "75%"), (0.50, "50%"), (0.25, "25%")):
            step_ngl = max(1, int(init_ngl * frac))
            if step_ngl < init_ngl:
                add(
                    step_ngl,
                    q8_type if init_kv_type == f16_type else init_kv_type,
                    "Q8_0" if init_kv_type == f16_type else init_kv_name,
                    bool(FLASH_ATTN),
                    True,
                    False if frac <= 0.50 else True,
                    context_size,
                    f"Stepped down GPU offload to {step_ngl} layers ({tag}) due to tight VRAM.",
                )
                add(
                    step_ngl,
                    f16_type,
                    "F16",
                    False,
                    False,
                    False,
                    context_size,
                    f"Stepped down to {step_ngl} GPU layers with KV cache in system RAM.",
                )

    add(0, f16_type, "F16", False, False, False, context_size, "Automatically fell back to CPU + system RAM loading.")
    if context_size > 4096:
        add(0, f16_type, "F16", False, False, False, 4096, "Automatically lowered context length to 4096 for system RAM fit.")
    return attempts


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
        STATE.loading_model = filename
    try:
        unload_current_model()
        preflight = preflight_model(filename, context_size, mmproj_model=mmproj_model)
        paired_mmproj = preflight.get("mmproj_model")
        vram_before = _measure_free_vram_bytes()
        log.info(
            f"Loading '{filename}' (preflight n_gpu_layers={preflight['n_gpu_layers']}, n_ctx={context_size}, mmproj={paired_mmproj or 'none'}) ..."
        )
        t0 = time.perf_counter()
        attempts = _build_load_attempts(preflight, context_size)
        last_exc: Optional[Exception] = None
        chosen: Optional[Dict[str, Any]] = None
        last_oom_failed_ngl: Optional[int] = None
        cached_handlers: Dict[bool, Any] = {}

        try:
            for attempt_idx, attempt in enumerate(attempts):
                if (
                    last_oom_failed_ngl is not None
                    and attempt["n_gpu_layers"] >= last_oom_failed_ngl
                    and attempt["offload_kqv"]
                ):
                    continue
                if attempt_idx > 0:
                    log.warning(
                        f"Retrying '{filename}' (attempt {attempt_idx + 1}): "
                        f"ngl={attempt['n_gpu_layers']}, kv={attempt['kv_cache_type']}, "
                        f"fa={attempt['flash_attn']}, kqv_gpu={attempt['offload_kqv']}, n_ctx={attempt['n_ctx']}"
                    )
                    gc.collect()
                    time.sleep(0.2)

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
                        mm_gpu = bool(attempt["mmproj_gpu"])
                        if mm_gpu not in cached_handlers:
                            clip_path = MODEL_DIR / paired_mmproj
                            try:
                                cached_handlers[mm_gpu] = MTMDChatHandler(
                                    clip_model_path=str(clip_path),
                                    use_gpu=mm_gpu,
                                    verbose=False,
                                )
                            except Exception as mm_exc:
                                if mm_gpu:
                                    log.warning(f"Vision mmproj GPU init failed ({mm_exc}); retrying on CPU.")
                                    continue
                                raise
                        chat_handler = cached_handlers[mm_gpu]
                        effective_chat_format = None

                llama_kwargs: Dict[str, Any] = {
                    "model_path": str(path),
                    "n_gpu_layers": int(attempt["n_gpu_layers"]),
                    "n_ctx": int(attempt["n_ctx"]),
                    "n_threads": N_THREADS,
                    "n_batch": N_BATCH,
                    "last_n_tokens_size": 1024,
                    "type_k": int(attempt["kv_type"]),
                    "type_v": int(attempt["kv_type"]),
                    "offload_kqv": bool(attempt["offload_kqv"]),
                    "flash_attn": bool(attempt["flash_attn"]),
                    "chat_format": effective_chat_format,
                    "verbose": False,
                }
                if chat_handler is not None:
                    llama_kwargs["chat_handler"] = chat_handler

                try:
                    STATE.llm = Llama(**llama_kwargs)
                    chosen = attempt
                    break
                except Exception as exc:
                    last_exc = exc
                    STATE.llm = None
                    if _is_vram_oom_error(exc) and attempt["n_gpu_layers"] > 0:
                        last_oom_failed_ngl = attempt["n_gpu_layers"]
                    log.warning(f"Load attempt {attempt_idx + 1} failed: {exc}")

            if chosen is None or STATE.llm is None:
                raise last_exc or RuntimeError("All model load attempts failed.")

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

        if chosen.get("reason"):
            preflight.setdefault("warnings", []).append(chosen["reason"])
        preflight["n_gpu_layers"] = int(chosen["n_gpu_layers"])
        preflight["kv_type"] = int(chosen["kv_type"])
        preflight["kv_cache_type"] = str(chosen["kv_cache_type"])
        preflight["offload_kqv"] = bool(chosen["offload_kqv"])
        preflight["context_size"] = int(chosen["n_ctx"])

        vram_after = _measure_free_vram_bytes()
        measured_vram = max(0, vram_before - vram_after) if vram_before > 0 else 0

        dt = time.perf_counter() - t0
        STATE.loaded_model = filename
        STATE.loaded_at = time.time()
        STATE.load_seconds = dt
        STATE.loaded_gpu_layers = int(chosen["n_gpu_layers"])
        STATE.loaded_context_size = int(chosen["n_ctx"])
        STATE.loaded_kv_cache_type = str(chosen["kv_cache_type"])
        STATE.loaded_chat_format = chat_format
        STATE.loaded_mmproj = paired_mmproj
        STATE.loaded_vision_capable = bool(paired_mmproj)
        STATE.loaded_vram_bytes = (
            max(int(preflight.get("predicted_gpu_bytes") or 0), measured_vram)
            if int(chosen["n_gpu_layers"]) > 0
            else 0
        )
        log.info(
            f"Model '{filename}' resident ({STATE.loaded_gpu_layers} GPU layers, kv={STATE.loaded_kv_cache_type}, vision={STATE.loaded_vision_capable}). Load time: {dt:.2f}s"
        )
        return preflight
    finally:
        with STATE._lease:
            STATE.locked = False
            STATE.loading_model = None


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

    raw_models: List[Dict[str, Any]] = []
    mmprojs: List[Dict[str, Any]] = []
    split_sizes: Dict[Tuple[str, int], int] = {}
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
            split_info = _parse_split_gguf_name(Path(rfilename).name)
            if split_info:
                prefix, _part, total = split_info
                split_key = (f"{Path(rfilename).parent.as_posix()}/{prefix.lower()}", total)
                split_sizes[split_key] = split_sizes.get(split_key, 0) + size_bytes
            raw_models.append(entry)

    models: List[Dict[str, Any]] = []
    for entry in raw_models:
        split_info = _parse_split_gguf_name(entry["local_name"])
        if split_info:
            prefix, part, total = split_info
            if part != 1:
                continue
            split_key = (f"{Path(entry['filename']).parent.as_posix()}/{prefix.lower()}", total)
            total_size = split_sizes.get(split_key, entry["size_bytes"])
            entry = {
                **entry,
                "size_bytes": total_size,
                "size_gb": round(total_size / (1024 ** 3), 2),
                "part_count": total,
            }
        models.append(entry)
    return {"repo_id": clean_repo, "models": models, "mmprojs": mmprojs}


def _check_download_disk_space(required_bytes: int, label: str = "model") -> None:
    if required_bytes <= 0:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(MODEL_DIR).free)
    safety_margin = 128 * 1024 * 1024
    if free_bytes < required_bytes + safety_margin:
        raise RuntimeError(
            f"Insufficient free disk space in {MODEL_DIR} for {label} "
            f"(need ~{required_bytes / GIB:.2f} GB, only {free_bytes / GIB:.2f} GB free)."
        )


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
    headers: Dict[str, str] = {"User-Agent": "GHostPCP/1.4.0"}
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
            _check_download_disk_space(int(resp2.headers.get("Content-Length") or 0), local_filename)
            return _write_hf_stream(resp2, part_path, target, stage_label, existing, t_start, cancel_event)
        resp.raise_for_status()
        if resp.status_code == 200 and existing > 0:
            # Server ignored Range header; restart cleanly from 0
            existing = 0
        _check_download_disk_space(int(resp.headers.get("Content-Length") or 0), local_filename)
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
        split_info = _parse_split_gguf_name(Path(filename).name)
        if split_info and split_info[1] == 1:
            prefix, _, total = split_info
            save_split = _parse_split_gguf_name(save_as)
            save_prefix = save_split[0] if save_split else prefix
            remote_dir = str(Path(filename).parent.as_posix())
            for part_idx in range(1, total + 1):
                part_name = f"{prefix}-{part_idx:05d}-of-{total:05d}.gguf"
                remote_part = f"{remote_dir}/{part_name}" if remote_dir and remote_dir != "." else part_name
                local_part = f"{save_prefix}-{part_idx:05d}-of-{total:05d}.gguf"
                _stream_hf_file(
                    repo_id=repo_id,
                    remote_filename=remote_part,
                    local_filename=local_part,
                    stage_label=f"Downloading {local_part} (part {part_idx}/{total})",
                    cancel_event=MODEL_DOWNLOAD_STATE.cancel_event,
                )
        else:
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
    auto_mount_model: bool = True
    system_prompt: StrictStr = Field(default=DEFAULT_SETTINGS["system_prompt"], max_length=12000)
    pcp_enabled: bool = True
    pcp_top_k: StrictInt = Field(default=4, ge=1, le=12)
    tts_engine: Literal["piper", "openai_api"] = "piper"
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=128)
    tts_api_url: StrictStr = Field(default="http://127.0.0.1:3900", max_length=500)
    tts_api_model: StrictStr = Field(default="tts-1", max_length=120)
    tts_api_key: StrictStr = Field(default="", max_length=500)
    mic_device: StrictStr = Field(default="auto", max_length=200)
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
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=128)
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
    tts_voice: Optional[StrictStr] = Field(default=None, min_length=1, max_length=128)
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
    text: StrictStr = Field(min_length=1, max_length=100000)
    voice: StrictStr = Field(default="silas", max_length=128)
    play_on_host: Optional[bool] = None
    engine: Optional[Literal["piper", "openai_api"]] = None
    api_url: Optional[StrictStr] = Field(default=None, max_length=500)
    api_model: Optional[StrictStr] = Field(default=None, max_length=120)


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
        try:
            _check_download_disk_space(int(float(preset.get("size_gb") or 0) * GIB), save_as)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    credit_vram = STATE.loaded_vram_bytes if STATE.llm is not None else 0
    plan = preflight_model(
        req.model,
        req.n_ctx,
        mmproj_model=req.mmproj_model,
        credit_vram_bytes=credit_vram,
    )
    if STATE.loaded_model:
        if credit_vram > 0:
            plan["warnings"].append(
                f"Credited ~{credit_vram / GIB:.2f} GiB currently held by '{STATE.loaded_model}' "
                "(automatically unloaded when you click MOUNT)."
            )
        else:
            plan["warnings"].append(
                "A model is currently resident; loading will unload it before allocation."
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
            STATE.loading_model = None
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
        STATE.auto_archived_signatures.clear()
        return {"status": "reset", "model": model}
    finally:
        with STATE._lease:
            STATE.locked = False
            STATE.loading_model = None


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
        "loading_model": STATE.loading_model,
        "mmproj_model": STATE.loaded_mmproj,
        "vision_capable": STATE.loaded_vision_capable,
        "n_gpu_layers": STATE.loaded_gpu_layers,
        "vram_gb": round(STATE.loaded_vram_bytes / GIB, 2) if STATE.loaded_vram_bytes else 0.0,
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
            text = _strip_reasoning_blocks(text)
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
        unsaved_msgs = [
            m for m in raw_msgs
            if _turn_signature(m) not in STATE.auto_archived_signatures
        ]
        settings = load_settings()
        capture_cont = (
            req.capture_continuity
            if req.capture_continuity is not None
            else bool(settings.get("auto_continuity_summary", True))
        )
        continuity_note = _capture_session_continuity_note(raw_msgs) if capture_cont else ""

        if raw_msgs and not unsaved_msgs:
            files = sorted(active_dir.glob("ghost_pcp_*.md"))
            result = {
                "saved_to": files[-1].name if files else "already_archived",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "turns_saved": 0,
                "passages_added": 0,
                "total_volumes": PCP_INDEX.file_count,
                "total_passages": PCP_INDEX.passage_count,
            }
        else:
            try:
                result = PCP_INDEX.append_session(unsaved_msgs, continuity_note=continuity_note or None)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                log.error(f"Save session failed: {exc}")
                raise HTTPException(status_code=500, detail=f"Failed to save session: {exc}") from exc

        result["continuity_note"] = continuity_note
        STATE.auto_archived_signatures.clear()
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
            STATE.loading_model = None


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
def tts_voices_endpoint(
    engine: Optional[str] = None,
    api_url: Optional[str] = None,
    x_tts_api_key: Optional[str] = Header(default=None),
):
    settings = load_settings()
    chosen_engine = (engine or settings.get("tts_engine") or "piper").strip().lower()
    if chosen_engine == "openai_api":
        chosen_url = (api_url if api_url is not None else settings.get("tts_api_url")) or "http://127.0.0.1:3900"
        chosen_key = x_tts_api_key if x_tts_api_key is not None else str(settings.get("tts_api_key") or "")
        voices, err = scan_openai_tts_voices(chosen_url, api_key=chosen_key)
        return {"voices": voices, "engine": "openai_api", "api_url": chosen_url, "error": err}
    return {"voices": scan_piper_voices(), "engine": "piper", "error": None}


@app.post("/v1/tts", dependencies=[Depends(verify_access)])
def tts_endpoint(req: TTSRequest, request: Request):
    is_local_loopback = _is_local_loopback_request(request)
    play_locally = req.play_on_host if req.play_on_host is not None else is_local_loopback
    settings = load_settings()
    chosen_engine = (req.engine or settings.get("tts_engine") or "piper").strip().lower()

    try:
        if chosen_engine == "openai_api":
            chosen_url = (req.api_url or settings.get("tts_api_url") or "http://127.0.0.1:3900").strip()
            chosen_model = (req.api_model or settings.get("tts_api_model") or "tts-1").strip()
            chosen_key = str(settings.get("tts_api_key") or "")
            wav_bytes = read_aloud_openai_api(
                req.text,
                req.voice,
                api_url=chosen_url,
                api_model=chosen_model,
                api_key=chosen_key,
            )
        else:
            wav_bytes = read_aloud(req.text, req.voice, play_locally=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        log.error(f"TTS error: {exc}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    played_on_host = _play_wav_on_host(wav_bytes) if play_locally else False
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-Played-On-Host": "true" if played_on_host else "false",
        },
    )


@app.post("/v1/tts/stop", dependencies=[Depends(verify_access)])
def tts_stop_endpoint():
    stop_read_aloud()
    return {"status": "stopped"}


@app.get("/v1/audio/devices", dependencies=[Depends(verify_access)])
def audio_devices_endpoint():
    settings = load_settings()
    return {
        "devices": scan_input_devices(),
        "selected": settings.get("mic_device") or "auto",
    }


@app.post("/v1/audio/test", dependencies=[Depends(verify_access)])
def audio_test_endpoint(request: Request, device: str = ""):
    if not _is_local_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Host microphone test is restricted to local desktop sessions.",
        )
    if _AUDIO_REC.recording:
        raise HTTPException(status_code=409, detail="Stop active microphone recording before testing.")
    settings = load_settings()
    pref = device.strip() or str(settings.get("mic_device") or "auto")
    probe = _HostAudioRecorder()
    try:
        dev_name = probe.start(device_pref=pref)
        time.sleep(0.75)
        samples, duration_sec = probe.stop_and_extract()
        import numpy as np

        if samples.size > 0:
            centered = samples - float(np.median(samples))
            peak = round(float(np.max(np.abs(centered))), 5)
            rms = round(float(np.sqrt(np.mean(centered * centered))), 5)
        else:
            peak, rms = 0.0, 0.0
        return {
            "device": dev_name,
            "sample_rate": probe.capture_sample_rate,
            "duration_sec": duration_sec,
            "peak": peak,
            "rms": rms,
            "heard_signal": peak >= 0.002,
        }
    except Exception as exc:
        probe.cancel()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/record/start", dependencies=[Depends(verify_access)])
def audio_record_start_endpoint(request: Request):
    if not _is_local_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Host microphone recording is restricted to local desktop sessions.",
        )
    settings = load_settings()
    mic_pref = str(settings.get("mic_device") or "auto")
    try:
        device_name = _AUDIO_REC.start(device_pref=mic_pref)
        return {
            "status": "recording",
            "device": device_name,
            "sample_rate": _AUDIO_REC.capture_sample_rate,
        }
    except Exception as exc:
        log.error(f"Mic start error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/record/stop", dependencies=[Depends(verify_access)])
def audio_record_stop_endpoint(request: Request):
    if not _is_local_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail="Host microphone recording is restricted to local desktop sessions.",
        )
    capture_sr = int(_AUDIO_REC.capture_sample_rate or _AUDIO_REC.SAMPLE_RATE)
    dev_name = _AUDIO_REC.device_name or "Default Microphone"
    samples, duration_sec = _AUDIO_REC.stop_and_extract()
    import numpy as np

    if samples.size > 0:
        centered = samples - float(np.median(samples))
        peak = round(float(np.max(np.abs(centered))), 5)
        rms = round(float(np.sqrt(np.mean(centered * centered))), 5)
    else:
        peak, rms = 0.0, 0.0

    if duration_sec <= 0.05:
        log.info(f"Mic stop: device={dev_name!r} too short ({duration_sec}s)")
        return {
            "text": "",
            "duration_sec": duration_sec,
            "device": dev_name,
            "sample_rate": capture_sr,
            "peak": peak,
            "rms": rms,
            "engine": "whisper-base.en-onnx",
        }
    try:
        text = _get_stt_engine().transcribe_float32(samples, sample_rate=capture_sr)
        log.info(
            f"Mic stop: device={dev_name!r} sr={capture_sr}Hz dur={duration_sec}s "
            f"peak={peak:.4f} rms={rms:.4f} chars={len(text)}"
        )
        return {
            "text": text,
            "duration_sec": duration_sec,
            "device": dev_name,
            "sample_rate": capture_sr,
            "peak": peak,
            "rms": rms,
            "engine": "whisper-base.en-onnx",
        }
    except Exception as exc:
        log.error(f"Speech transcription failed: {exc}")
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

    # Decode standard PCM/WAV/FLAC/OGG directly in memory using built-in soundfile + Whisper ONNX
    try:
        samples, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        text = _get_stt_engine().transcribe_float32(samples, sample_rate=int(sr))
        return {"text": text, "engine": "whisper-base.en-onnx"}
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
    if not cleaned:
        STATE.auto_archived_signatures.clear()
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


def _turn_signature(msg: Dict[str, Any]) -> str:
    role = str(msg.get("role") or "").strip().lower()
    plain = _strip_reasoning_blocks(_extract_message_plain_text(msg.get("content"))).strip()
    return hashlib.sha1(f"{role}:{plain[:500]}".encode("utf-8", errors="replace")).hexdigest()


def _fit_messages_to_context(
    messages: List[Dict[str, Any]],
    n_ctx: int,
    max_tokens: int,
    evicted_out: Optional[List[Dict[str, Any]]] = None,
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
        dropped = fitted.pop(drop_idx)
        if evicted_out is not None and dropped.get("role") in ("user", "assistant"):
            evicted_out.append(dropped)

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
        is_qwen3 = bool(STATE.loaded_model and "qwen3" in STATE.loaded_model.lower())
        sys_msg = (
            "Decide whether the user's message requires looking up external public facts, news, weather, prices, or web pages on the internet. "
            "Reply NONE for personal conversation, greetings, emotional venting, roleplay, device/app troubleshooting, or questions answerable without searching online. "
            "Otherwise reply ONLY with a concise 2-7 word search query."
        )
        usr_msg = str(user_prompt or "")[:600]
        if is_qwen3:
            sys_msg += "\n/no_think"
            usr_msg += "\n/no_think"
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ],
            temperature=0.0,
            top_p=0.9,
            max_tokens=48,
            stream=False,
        )
        raw = str(((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        raw = _strip_reasoning_blocks(raw)
        raw = re.sub(r"[\r\n\"'`]+", " ", raw).strip()
        norm = re.sub(r"[\s.!?]+$", "", raw).upper()
        if not raw or norm in ("NONE", "NO", "N/A", "NONE NEEDED", "NO SEARCH", "NO SEARCH NEEDED"):
            return "NONE"
        if len(raw) > 120:
            return "NONE"
        return raw
    except Exception:
        return "NONE"


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
                cleaned = _strip_reasoning_blocks(plain_prior).strip()
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

    # Inject date-only wall-clock anchor after static persona/rules to preserve intra-day KV prefix reuse
    if "Current local date:" not in raw_messages[0]["content"]:
        date_anchor = datetime.now().strftime("%Y-%m-%d (%A)")
        raw_messages[0]["content"] = (
            raw_messages[0]["content"].rstrip()
            + f"\n\nTEMPORAL ANCHOR:\nCurrent local date: {date_anchor}"
        )

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
                + "\n\n"
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
        if hasattr(PCP_INDEX, "search_formatted"):
            excerpts = PCP_INDEX.search_formatted(retrieval_query, limit=top_k)
        else:
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
    evicted_turns: List[Dict[str, Any]] = []
    raw_messages = _fit_messages_to_context(
        raw_messages,
        effective_n_ctx,
        effective_max_tokens,
        evicted_out=evicted_turns,
    )

    auto_archived_turns = 0
    if evicted_turns and use_pcp and PCP_INDEX is not None:
        fresh_evicted: List[Dict[str, Any]] = []
        for turn in evicted_turns:
            if not _extract_message_plain_text(turn.get("content")).strip():
                continue
            sig = _turn_signature(turn)
            if sig not in STATE.auto_archived_signatures:
                fresh_evicted.append(turn)
        if fresh_evicted:
            try:
                PCP_INDEX.append_session(fresh_evicted, timestamp_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                for turn in fresh_evicted:
                    STATE.auto_archived_signatures.add(_turn_signature(turn))
                auto_archived_turns = len(fresh_evicted)
                turn_warnings.append(
                    f"Auto-archived {auto_archived_turns} older message(s) to PCP memory before sliding context."
                )
            except Exception as exc:
                log.warning(f"Overflow auto-archive skipped cleanly: {exc}")

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
    ctx_pct = min(100, round((prompt_tokens_est / max(1, effective_n_ctx)) * 100))
    if ctx_pct >= 85 and auto_archived_turns == 0:
        turn_warnings.append(
            f"Context window is {ctx_pct}% full; overflowing turns auto-archive into PCP memory (click SAVE & CLEAR anytime)."
        )
    STATE.last_turn_meta = {
        "sources": turn_sources,
        "warnings": turn_warnings,
        "prompt_tokens_est": prompt_tokens_est,
        "n_ctx": effective_n_ctx,
        "ctx_pct": ctx_pct,
        "auto_archived_turns": auto_archived_turns,
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


class _ChannelNormalizer:
    """Convert Harmony (<|channel|>), <thought>, and [THINK] tokens into <think>...</think> across stream chunks."""

    _TOKEN_RE = re.compile(
        r"(?:<\|start\|>[a-z0-9_:-]*)?<\|channel\|>(analysis|commentary|final)(?:<\|constrain\|>[^\s<]+)?(?:<\|message\|>)?|"
        r"<\|start\|>[a-z0-9_:-]*|<\|end\|>|<\|return\|>|<\|message\|>|<thought>|</thought>|\[THINK\]|\[/THINK\]",
        re.IGNORECASE,
    )
    _TAIL_RE = re.compile(
        r"(?:"
        r"(?:<\|start\|>[a-z0-9_:-]{0,24})?<\|channel\|>(?:[a-z]{0,16})(?:<\|constrain\|>[^\s<]{0,24})?(?:<\|m(?:e(?:s(?:s(?:a(?:g(?:e\|?)?)?)?)?)?)?)?|"
        r"<\|start\|>[a-z0-9_:-]{0,24}|"
        r"<\|[a-z_|]{0,16}|"
        r"<|"
        r"<t(?:h(?:o(?:u(?:g(?:h(?:t)?)?)?)?)?)?|"
        r"</t(?:h(?:o(?:u(?:g(?:h(?:t)?)?)?)?)?)?|"
        r"\[(?:t(?:h(?:i(?:n(?:k)?)?)?)?|/(?:t(?:h(?:i(?:n(?:k)?)?)?)?)?)"
        r")$",
        re.IGNORECASE,
    )

    def __init__(self):
        self.pending = ""
        self.in_reasoning = False

    def _replace_match(self, match: re.Match) -> str:
        channel = (match.group(1) or "").lower()
        token = match.group(0).lower()
        if channel in ("analysis", "commentary") or token in ("<thought>", "[think]"):
            if not self.in_reasoning:
                self.in_reasoning = True
                return "<think>"
            return ""
        if channel == "final" or token in ("</thought>", "[/think]", "<|end|>", "<|return|>"):
            if self.in_reasoning:
                self.in_reasoning = False
                return "</think>"
            return ""
        return ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buf = self.pending + text
        self.pending = ""
        tail_match = self._TAIL_RE.search(buf)
        hold_from = tail_match.start() if tail_match else len(buf)
        ready = buf[:hold_from]
        self.pending = buf[hold_from:]
        if not ready:
            return ""
        if "<" not in ready and "[" not in ready:
            return ready
        return self._TOKEN_RE.sub(self._replace_match, ready)

    def flush(self) -> str:
        out = self._TOKEN_RE.sub(self._replace_match, self.pending) if self.pending else ""
        self.pending = ""
        if self.in_reasoning:
            self.in_reasoning = False
            out += "</think>"
        return out


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
    channel_norm = _ChannelNormalizer()
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
                    normalized = channel_norm.feed(content)
                    filtered = think_gate.feed(normalized) if normalized else ""
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

        norm_tail = channel_norm.flush()
        if norm_tail:
            filtered_tail = think_gate.feed(norm_tail)
            if filtered_tail:
                yield _sse({
                    "id": f"chatcmpl-{generation_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": filtered_tail}}],
                })
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
        prefill_sec = max(0.0, (first_token_at - start) if first_token_at else 0.0)
        decode_now = elapsed_now - prefill_sec
        rate_now = token_count / decode_now if decode_now > 0 else 0.0
        meta_payload = dict(turn_meta or {})
        meta_payload["tok_per_sec"] = round(rate_now, 1)
        meta_payload["prefill_sec"] = round(prefill_sec, 2)
        meta_payload["elapsed_sec"] = round(elapsed_now, 2)
        meta_payload["vram_gb"] = round(STATE.loaded_vram_bytes / GIB, 2) if STATE.loaded_vram_bytes else 0.0
        meta_payload["gpu_layers"] = STATE.loaded_gpu_layers
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
        try:
            content = out["choices"][0]["message"]["content"]
            if isinstance(content, str):
                if show_reasoning:
                    out["choices"][0]["message"]["content"] = _normalize_reasoning_tags(content)
                else:
                    cleaned = _strip_reasoning_blocks(content).strip()
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
            "vram_gb": round(STATE.loaded_vram_bytes / GIB, 2) if STATE.loaded_vram_bytes else 0.0,
            "gpu_layers": STATE.loaded_gpu_layers,
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
            print(_select_startup_banner())
        except UnicodeEncodeError:
            pass
    log.info("GHostPCP boot sequence initiated.")
    log.info(f"Model directory : {MODEL_DIR.resolve()}")
    log.info(f"PCP Data dir    : {PCP_DATA_DIR.resolve()}")
    log.info(f"Settings file   : {SETTINGS_PATH.resolve()}")
    log.info(f"Selected backend: {BACKEND_NAME} ({RUNTIME_LLAMA_LIB_DIR}) · {N_THREADS} physical threads")
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


def _pick_auto_mount_target() -> Optional[Tuple[str, dict]]:
    """Choose the preferred persona model or last-used valid local GGUF when auto-mount is enabled."""
    settings = load_settings()
    if not settings.get("auto_mount_model", True):
        return None
    valid_models = [m["id"] for m in scan_models() if m.get("valid_gguf", True)]
    if not valid_models:
        return None
    available = set(valid_models)
    active_persona = _get_active_persona() if _personas_registry_path().is_file() else None
    preferred = str((active_persona or {}).get("preferred_model") or "").strip()
    if preferred and preferred in available:
        return preferred, settings
    saved_model = str(settings.get("model") or "").strip()
    if saved_model and saved_model in available:
        return saved_model, settings
    if len(valid_models) == 1:
        return valid_models[0], settings
    return None


def _schedule_startup_auto_mount() -> None:
    """Trigger background model auto-mount when running the real server executable/script."""
    if os.environ.get("GHOST_NO_AUTOLOAD", "").strip() == "1":
        return
    picked = _pick_auto_mount_target()
    if not picked:
        return
    model_id, settings = picked

    def _worker():
        try:
            with STATE._lease:
                if STATE.llm is not None or STATE.locked or STATE.busy:
                    return
            log.info(f"Auto-mounting startup model '{model_id}' ...")
            load_model(
                model_id,
                int(settings.get("n_ctx") or DEFAULT_N_CTX),
                str(settings.get("chat_format") or "auto"),
                str(settings.get("mmproj_model") or "auto"),
            )
        except Exception as exc:
            log.warning(f"Startup auto-mount skipped cleanly: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


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
            _schedule_startup_auto_mount()
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
            _schedule_startup_auto_mount()

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
