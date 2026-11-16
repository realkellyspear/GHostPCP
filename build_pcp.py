# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the standalone GHostPCP desktop application (GHostPCP.exe).

Run from this folder with the virtual environment active:
    python build_pcp.py

Produces GHostPCP.exe beside ./ghost_models, ./pcp_data, and ./backends.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_PLUGINS = {
    "cpu": "ggml-cpu.dll",
    "cuda": "ggml-cuda.dll",
    "vulkan": "ggml-vulkan.dll",
}


def _copy_external_backends() -> list[str]:
    import llama_cpp
    import pefile

    source = Path(llama_cpp.__file__).resolve().parent / "lib"
    if not source.is_dir():
        raise FileNotFoundError(f"llama_cpp native library directory not found: {source}")

    common = [
        path for path in source.glob("*.dll")
        if path.name not in BACKEND_PLUGINS.values()
    ]
    if not any(path.name == "llama.dll" for path in common):
        raise FileNotFoundError("llama_cpp wheel is missing llama.dll")

    cuda_plugin = source / BACKEND_PLUGINS["cuda"]
    cuda_runtime = []
    if cuda_plugin.is_file():
        cuda_imports = {
            entry.dll.decode(errors="replace").lower()
            for entry in pefile.PE(str(cuda_plugin)).DIRECTORY_ENTRY_IMPORT
            if entry.dll.decode(errors="replace").lower().startswith("cublas")
        }
        runtime_roots = [
            HERE / ".venv" / "Lib" / "site-packages" / "nvidia",
            Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
            Path(os.environ.get("CUDA_PATH", "")) / "bin",
        ]
        available_runtime = {
            path.name.lower(): path
            for root in runtime_roots if root.is_dir()
            for path in root.rglob("*.dll")
        }
        missing_runtime = cuda_imports - set(available_runtime)
        if missing_runtime:
            raise FileNotFoundError(
                "CUDA backend requires external NVIDIA runtime DLLs not found by the builder: "
                + ", ".join(sorted(missing_runtime))
            )
        cuda_runtime = [
            path for name, path in sorted(available_runtime.items())
            if name.startswith("cublas")
        ]

    built = []
    for backend, plugin_name in BACKEND_PLUGINS.items():
        plugin = source / plugin_name
        if not plugin.is_file():
            continue

        target = HERE / "backends" / backend
        target.mkdir(parents=True, exist_ok=True)
        for stale_plugin in BACKEND_PLUGINS.values():
            if stale_plugin != plugin_name:
                stale_path = target / stale_plugin
                if stale_path.exists():
                    stale_path.unlink()

        files = list(common)
        for linked_plugin in BACKEND_PLUGINS.values():
            linked_path = source / linked_plugin
            if linked_path.is_file():
                files.append(linked_path)
        files.append(plugin)
        if backend in {"cpu", "cuda"}:
            files.extend(cuda_runtime)
        files = list({path.name.lower(): path for path in files}.values())
        for path in files:
            dest = target / path.name
            if dest.exists():
                dest.unlink()
            try:
                os.link(path, dest)
            except OSError:
                shutil.copy2(path, dest)
        built.append(backend)

    if "cpu" not in built:
        raise FileNotFoundError("llama_cpp wheel is missing the CPU backend DLL")
    return built


def build() -> None:
    models = HERE / "ghost_models"
    html = HERE / "ghost_pcp.html"
    icon = HERE / "assets" / "ghost_icon.ico"
    models.mkdir(parents=True, exist_ok=True)
    if not html.is_file():
        raise FileNotFoundError(f"UI file not found: {html}")
    if not icon.is_file():
        raise FileNotFoundError(f"App icon not found: {icon}")

    backends = _copy_external_backends()
    print(f"[build] external backends staged: {', '.join(backends)}")

    workpath = HERE / "build_pcp"
    output = HERE / "GHostPCP.exe"
    old_output = HERE / "GHostPCP.old.exe"
    if old_output.exists():
        try:
            old_output.unlink()
        except OSError:
            pass
    if output.exists():
        try:
            output.unlink()
        except OSError:
            try:
                output.rename(old_output)
                print(f"[build] Running GHostPCP.exe renamed to {old_output.name} so new build can write cleanly.")
            except OSError:
                pass

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "GHostPCP",
        "--distpath", str(HERE),
        "--workpath", str(workpath),
        "--specpath", str(HERE),
        "--icon", str(icon),
        "--add-data", f"{html};.",
        "--add-data", f"{HERE / 'assets'};assets",
        "--collect-submodules", "webview",
        "--collect-submodules", "llama_cpp",
        "--collect-submodules", "piper",
        "--collect-submodules", "pypdf",
        "--collect-data", "piper",
        "--collect-all", "onnxruntime",
        "--collect-data", "_sounddevice_data",
        "--collect-binaries", "_sounddevice_data",
        "--collect-data", "_soundfile_data",
        "--collect-binaries", "_soundfile_data",
        "--hidden-import", "webview.platforms.winforms",
        "--hidden-import", "safetensors.numpy",
        "--hidden-import", "tokenizers",
        "--hidden-import", "qrcode.image.svg",
        "--hidden-import", "multipart",
        "--hidden-import", "sounddevice",
        "--hidden-import", "soundfile",
        "--hidden-import", "_sounddevice",
        "--hidden-import", "_soundfile",
        "--exclude-module", "torch",
        "--exclude-module", "scipy",
        "--exclude-module", "sklearn",
        "--exclude-module", "sentence_transformers",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "pytorch_lightning",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "F5_TTS",
        str(HERE / "ghost_pcp.py"),
    ]
    subprocess.check_call(args, cwd=HERE)
    if workpath.is_dir():
        shutil.rmtree(workpath, ignore_errors=True)
    output = HERE / "GHostPCP.exe"
    print(f"Built standalone executable: {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
