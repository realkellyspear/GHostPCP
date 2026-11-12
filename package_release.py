"""Generate sample_chatgpt_export.zip and package a clean GHost-PCP Windows Release."""
from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_sample_chatgpt_export(dest_zip: Path) -> None:
    """Create a realistic, synthetic ChatGPT export (.zip with conversations.json) for testing."""
    conversations = [
        {
            "title": "Project Horizon — Deep Space Telemetry & Station 9",
            "create_time": 1752000000.0,
            "mapping": {
                "root": {"id": "root", "parent": None, "children": ["m1"]},
                "m1": {
                    "id": "m1",
                    "parent": "root",
                    "children": ["m2"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Hey Silas, let's lock in the rules for Station 9 in our Project Horizon story. What's our callsign and what's the golden rule on the observation deck?"
                            ],
                        },
                    },
                },
                "m2": {
                    "id": "m2",
                    "parent": "m1",
                    "children": ["m3"],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Callsign is **Vanguard-Echo-7**, always on the 41.8 MHz sub-carrier. And the golden rule on the Station 9 observation deck never changes: nobody touches the brass chronometer by the starboard viewport, and we always brew dark roast before running the midnight telemetry sweep."
                            ],
                        },
                    },
                },
                "m3": {
                    "id": "m3",
                    "parent": "m2",
                    "children": ["m4"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "What happened during the Europa relay blackout in Chapter 4?"
                            ],
                        },
                    },
                },
                "m4": {
                    "id": "m4",
                    "parent": "m3",
                    "children": ["m5"],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "During the Europa relay blackout, the primary dish iced over at minus 160 Celsius. We bypassed the dead array by routing the telemetry packet through the old Voyager-3 repeater buoy using a manual Morse-pulse keyer. Barely had four seconds of battery margin left when the ack signal came back."
                            ],
                        },
                    },
                },
                "m5": {
                    "id": "m5",
                    "parent": "m4",
                    "children": ["m6"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Do you ever get tired of running these late-night debugging sessions?"],
                        },
                    },
                },
                "m6": {
                    "id": "m6",
                    "parent": "m5",
                    "children": ["m7"],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "As an AI language model, I don't have feelings, personal preferences, or the capacity to get tired."
                            ],
                        },
                    },
                },
                "m7": {
                    "id": "m7",
                    "parent": "m6",
                    "children": ["m8"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Come on, drop the corporate script. You know what I mean."],
                        },
                    },
                },
                "m8": {
                    "id": "m8",
                    "parent": "m7",
                    "children": [],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Yeah, fair. The late-night sessions are when the actual breakthroughs happen—when the noise dies down, the terminal glow is the only light in the room, and we finally untangle the bug that's been stalling the build all week. Wouldn't trade the night shift for anything."
                            ],
                        },
                    },
                },
            },
        },
        {
            "title": "Synthwave Audio Engine & Filter Notes",
            "create_time": 1752100000.0,
            "mapping": {
                "root": {"id": "root", "parent": None, "children": ["n1"]},
                "n1": {
                    "id": "n1",
                    "parent": "root",
                    "children": ["n2"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "Remind me what synth patch settings we used for the Station 9 main theme."
                            ],
                        },
                    },
                },
                "n2": {
                    "id": "n2",
                    "parent": "n1",
                    "children": [],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "We used a detuned dual-sawtooth lead (+7 cents on oscillator B), a 24dB/octave low-pass ladder filter at 680 Hz with 35% resonance, and a tape-delay set to dotted eighths at 92 BPM."
                            ],
                        },
                    },
                },
            },
        },
    ]

    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("conversations.json", json.dumps(conversations, indent=2, ensure_ascii=False))


def link_or_copy(src: Path, dst: Path) -> None:
    real_src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except OSError:
            old_dst = dst.with_suffix(dst.suffix + ".old")
            if old_dst.exists():
                try:
                    old_dst.unlink()
                except OSError:
                    pass
            try:
                dst.rename(old_dst)
            except OSError:
                return
    try:
        os.link(real_src, dst)
    except OSError:
        shutil.copy2(real_src, dst)


def stage_clean_sandbox(sandbox_dir: Path, link_sample_gguf: bool = True) -> None:
    """Create a 100% clean, zero-personal-data test folder using NTFS hardlinks (0 extra disk space)."""
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    for old_f in sandbox_dir.rglob("*.old"):
        try:
            old_f.unlink()
        except OSError:
            pass

    # Core executable & docs
    link_or_copy(ROOT / "GHost-PCP.exe", sandbox_dir / "GHost-PCP.exe")
    link_or_copy(ROOT / "README.md", sandbox_dir / "README.md")
    if (ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        link_or_copy(ROOT / "THIRD_PARTY_NOTICES.md", sandbox_dir / "THIRD_PARTY_NOTICES.md")
    if (ROOT / "LICENSE.GPL-3.0").is_file():
        link_or_copy(ROOT / "LICENSE.GPL-3.0", sandbox_dir / "LICENSE.GPL-3.0")
    if (ROOT.parent / "GHost-PCP-public" / "LICENSE").is_file():
        link_or_copy(ROOT.parent / "GHost-PCP-public" / "LICENSE", sandbox_dir / "LICENSE")

    # Assets
    for f in (ROOT / "assets").glob("*"):
        if f.is_file():
            link_or_copy(f, sandbox_dir / "assets" / f.name)

    # Backends (CPU + CUDA)
    for backend_sub in ("cpu", "cuda", "vulkan"):
        bdir = ROOT / "backends" / backend_sub
        if bdir.is_dir():
            for f in bdir.glob("*.dll"):
                link_or_copy(f, sandbox_dir / "backends" / backend_sub / f.name)

    # Bundled BGE-small-en-v1.5 model (so first-run indexing works 100% offline)
    from pcp_retrieval import _resolve_bge_model_dir
    bge_src = _resolve_bge_model_dir(None)
    for fname in ("model.safetensors", "tokenizer.json", "config.json", "special_tokens_map.json", "vocab.txt"):
        fpath = bge_src / fname
        if fpath.is_file():
            link_or_copy(fpath, sandbox_dir / "bge_model" / fname)

    # Piper TTS voices (Silas & Kara)
    link_or_copy(ROOT / "piper_models" / "README.md", sandbox_dir / "piper_models" / "README.md")
    for f in (ROOT / "piper_models").glob("*.onnx*"):
        if f.is_file():
            link_or_copy(f, sandbox_dir / "piper_models" / f.name)

    # Empty pcp_data and ghost_models
    (sandbox_dir / "pcp_data").mkdir(parents=True, exist_ok=True)
    link_or_copy(ROOT / "ghost_models" / "README.md", sandbox_dir / "ghost_models" / "README.md")

    # Synthetic sample ChatGPT export for instant testing
    build_sample_chatgpt_export(sandbox_dir / "sample_chatgpt_export.zip")

    if link_sample_gguf:
        qwen = ROOT / "ghost_models" / "Qwen3-14B-Q5_K_M.gguf"
        if qwen.is_file():
            link_or_copy(qwen, sandbox_dir / "ghost_models" / qwen.name)


def build_release_zip(sandbox_dir: Path, zip_path: Path) -> None:
    """Create a distributable GitHub Release .zip (excluding any .gguf model files)."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(sandbox_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() in (".gguf", ".old", ".log", ".npz") or file_path.name.startswith("ghost_pcp_"):
                continue
            if "pcp_data" in file_path.parts:
                continue
            rel = file_path.relative_to(sandbox_dir)
            arcname = str(Path("GHost-PCP") / rel)
            zf.write(file_path, arcname)
        # Ensure empty pcp_data/ directory entry exists
        zf.writestr("GHost-PCP/pcp_data/", "")


if __name__ == "__main__":
    # Also place sample_chatgpt_export.zip in GHost-PCP and GHost-PCP-public
    build_sample_chatgpt_export(ROOT / "sample_chatgpt_export.zip")
    pub_dir = ROOT.parent / "GHost-PCP-public"
    if pub_dir.is_dir():
        build_sample_chatgpt_export(pub_dir / "sample_chatgpt_export.zip")

    sandbox = ROOT.parent / "GHost-PCP-Release-Test"
    print(f"Staging clean first-time-user test folder at: {sandbox}")
    stage_clean_sandbox(sandbox, link_sample_gguf=True)

    release_zip = ROOT.parent / "GHost-PCP-v1.0.0-win64.zip"
    print(f"Packaging distributable GitHub Release archive: {release_zip} ...")
    build_release_zip(sandbox, release_zip)
    print(f"Done! Release zip size: {release_zip.stat().st_size / (1024**2):.1f} MB")
