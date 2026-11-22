![GHostPCP — GPU Hosted Pure Concentrated Personality](GHostPCP%20Promo/1github_banner.jpg)

# GHostPCP

**GPU Hosted Pure Concentrated Personality**  
*Break Your AI Out of the Cloud and Onto Your Own Hardware*

GHostPCP is a local-first desktop and mobile-paired AI workbench for Windows. Import your conversation archives from **OpenAI ChatGPT**, **Google Gemini**, **Anthropic Claude**, **xAI Grok**, or **Character.AI**, along with `.pdf`, `.txt`, or `.md` documents; organize isolated AI personas with temporal memory grounding, structured core facts, evolving continuity logs, saved threads, and editable Markdown memory volumes; download or mount local `.gguf` text or **Vision (`mmproj`)** models across almost any architecture; and optionally toggle live **Web Search & URL Reading** or external **OpenAI-compatible TTS** whenever you want them.

![GHostPCP desktop interface](GHostPCP%20Promo/screenshots_v1.4.0/ghost_pcp_first_screen.png)

---

## Why GHostPCP Exists

![Why GHostPCP](GHostPCP%20Promo/why_gpcp.png)

Standard "chat with your docs" tools act like research librarians (*"According to source document 4..."*). GHostPCP gives every AI persona its own second-brain memory folder and grounds responses naturally in character—with hybrid exact-name + semantic recall, real-time temporal awareness, durable relationships, and portable `.zip` backups you own outright.

1. **Universal Archive Ingestion & Multi-Persona Isolation**: Drop your **OpenAI ChatGPT** (`.zip` / `conversations.json`), **Google Gemini Takeout** (`MyActivity.json` / `MyActivity.html`), **Anthropic Claude**, **xAI Grok**, **Character.AI**, or `.pdf` / `.txt` / `.md` files into **[ ARCHIVE ]**. GHostPCP auto-detects the format, deduplicates repeated conversations, removes canned corporate refusals (*"As an AI language model..."*), preserves source message timestamps (`USER [YYYY-MM-DD HH:MM:SS]:`), creates an isolated folder per persona (`pcp_data/personas/<id>/`), writes chronological `ghost_pcp_*.md` volumes, and builds a local **BGE + BM25 hybrid index** (`BAAI/bge-small-en-v1.5`).
2. **Temporal Grounding, Zero-Loss Overflow & Second-Brain Memory Tools (`[ MEMORIES ]`)**:
   - **Temporal Awareness**: Injects the local date (`TEMPORAL ANCHOR`) and exact relative ages (`Yesterday · 2026-11-21`, `1 year, 5 months ago · 2025-06-05`) on retrieved memory chunks, applying gentle recency tie-breaking when memories tie semantically while still letting older memories win cleanly when directly relevant.
   - **Zero-Loss Context Overflow Protection**: When long sessions exceed the active context window, older overflowing turns automatically save into the active persona's `ghost_pcp_*.md` archive (deduplicated against later **SAVE & CLEAR** calls), with a warning highlight at `>= 85%` context usage.
   - **Structured Second-Brain Tools**: Save **Core Facts & People** with relationship aliases (e.g. linking *"Jaedon"* to *"my brother"*) so indirect questions recall accurately, capture evolving session **Continuity Notes** automatically on **SAVE & CLEAR**, bookmark and reload named conversation **Threads**, search or edit/delete individual `ghost_pcp_*.md` volumes (with automatic backups), and export or restore any persona as a single **`.zip` backup bundle**.
3. **Universal Local GGUF Engine, Vision (`mmproj`) & Smart Preflight**:
   - **Broad Architecture & Hardware Support**: Runs standard dense models, MoE models, sliding-window attention (`gpt-oss`, Gemma 2/3, Mistral), hybrid SSM/attention architectures (`qwen3.5` / `qwen35moe`, Mamba), fused QKV models, and multi-part split GGUFs (`-00001-of-0000N.gguf`) across **NVIDIA CUDA**, **Vulkan** (AMD/Intel), **Apple Metal**, multi-GPU rigs, and **CPU + system RAM** fallback—plus automatic physical CPU core detection (`12` physical vs `24` SMT threads on Zen 5, etc.).
   - **VRAM Preflight & Automatic OOM Recovery**: Reads GGUF tensor indices across all files, budgets attention KV + hybrid SSM recurrent state, steps KV cache **`F16 → Q8_0 → Q4_0`** (and keeps KV in RAM via `offload_kqv=False` on ultra-tight GPUs) before dropping GPU layers, credits currently resident VRAM when previewing model swaps (**CHECK FIT**), and automatically steps down GPU layers / flash-attn / RAM fallback on OOM or architecture incompatibilities instead of failing load.
   - **Per-Persona Brains, Auto-Mount, Vision & Reasoning Normalization**: Assign a preferred GGUF model per persona (`preferred_model`), auto-mount on startup or persona switch, attach/paste (`Ctrl+V`) images via `llama.cpp` MTMD (`mmproj`), normalize `<think>`, `<thought>`, `[THINK]`, and OpenAI Harmony (`<|channel|>analysis` / `final`) tags consistently across chat, TTS, and logs, inspect live `prefill`, `tok/s`, `ctx %`, and `VRAM` telemetry, or pull single/split GGUFs from HuggingFace with HTTP `Range` `.part` resume, `GGUF` magic-header validation, and free disk-space checks.
4. **Opt-In Live Web Search & URL Reader (`WEB: OFF / ON`)**: Strictly **OFF by default** for 100% offline privacy. Flip **`WEB: ON`** in the chat bar to let your AI read pasted `http(s)://` links or perform fast DuckDuckGo/Wikipedia (no API key needed), self-hosted SearXNG, Brave, or Tavily lookups with clickable source chips below replies, Qwen3-aware query rewriting, and SSRF protection against local network URLs.
5. **Two-Way Local Voice (`[ MIC ]`, Piper TTS & External OpenAI TTS API) + Phone Tunnel**:
   - Dictate offline with **`[ MIC ]`** powered by bundled **Whisper-Base.en int8 ONNX** (`0 GB` GPU VRAM — handles desktop mics via `sounddevice` and HTTPS phone browsers).
   - Listen to replies using built-in local **Piper ONNX** voices (`Silas`, `Kara`, or any `.onnx` + `.onnx.json` pair in `piper_models/`), or connect an optional external **OpenAI-compatible TTS server** (such as **VoiceStudio** on `http://127.0.0.1:3900`, Kokoro, Qwen3-TTS, or Chatterbox — separate process over HTTP REST, no external TTS binaries or weights bundled).
   - Open **[ PHONE / TUNNEL ]** to scan a QR code for synced remote mobile access via local Wi-Fi, Tailscale, or a 1-click Cloudflare HTTPS Quick Tunnel with 192-bit pairing-token authentication.

## See It in Action

**Desktop Workspace & Live Memory Chat**

<img src="GHostPCP%20Promo/screenshots_v1.4.0/pcp_welcome.png" alt="GHostPCP welcome workspace" width="700">

**Universal GGUF Engine, Hybrid SSM Preflight & External TTS Config**

<img src="GHostPCP%20Promo/screenshots_v1.4.0/config_preflight.png" alt="GHostPCP engine preflight and VoiceStudio config" width="700">

**Second-Brain Memory & Continuity Inspector**

<img src="GHostPCP%20Promo/screenshots_v1.4.0/memories_screen.png" alt="GHostPCP second-brain memory search inspector" width="700">

**Universal Archive Import & Completion**

<img src="GHostPCP%20Promo/screenshots_v1.4.0/ingest_screen.png" alt="GHostPCP universal archive import and persona setup" width="380"> <img src="GHostPCP%20Promo/screenshots_v1.4.0/ingest_complete.png" alt="GHostPCP archive indexing complete" width="380">

**Cloudflare Quick Tunnel & Live Synced Mobile UI**

<img src="GHostPCP%20Promo/screenshots_v1.4.0/safe_phone_tunnel.png" alt="GHostPCP Cloudflare Quick Tunnel pairing QR" width="600">

<p>
<img src="GHostPCP%20Promo/screenshots_v1.4.0/pcp_phone.png" alt="GHostPCP mobile synced chat" width="230">
<img src="GHostPCP%20Promo/screenshots_v1.4.0/pcp_phone_1.png" alt="GHostPCP mobile engine config" width="230">
<img src="GHostPCP%20Promo/screenshots_v1.4.0/pcp_phone_2.png" alt="GHostPCP mobile archive view" width="230">
</p>

---

## Folder Layout

```text
GHostPCP/
  GHostPCP.exe                 Standalone desktop application (built locally)
  ghost_pcp.py                 FastAPI + llama.cpp + vision, preflight, TTS/STT & PCP server
  pcp_ingest.py                Universal importer (ChatGPT, Gemini, Claude, Grok, Character.AI, docs)
  pcp_retrieval.py             Hybrid BGE-small-en-v1.5 (Safetensors+NumPy) + BM25 + temporal decay indexer
  pcp_web.py                   Opt-in web search (DDG/Wikipedia/SearXNG/Brave/Tavily) & SSRF-safe URL reader
  persona.py                   Grounding prefix, core facts/aliases formatter & filter defaults
  ghost_pcp.html               Desktop & mobile-responsive interface (Vision, Web toggle, Memories, Mic)
  build_pcp.py                 PyInstaller executable + per-backend DLL stager
  package_release.py           Zero-data release zip packager & sample export generator
  tests/test_pcp.py            Automated unit & FastAPI TestClient regression suite
  ghost_models/                Place your local .gguf (single or split) & mmproj files here
    README.md
  piper_models/                Place local Piper TTS voice models here
  stt_model/                   Bundled offline Whisper-Base.en int8 ONNX STT model (0 VRAM)
  pcp_data/                    Generated persona folders, ghost_pcp_*.md volumes & embeddings.npz (git-ignored)
  assets/                      GHost circle logo and icons
```

---

## Architecture Highlights for Contributors

- **Zero-PyTorch Runtime Footprint**: Semantic embeddings (`BAAI/bge-small-en-v1.5`) run directly via `safetensors.numpy` + `tokenizers` + multi-threaded NumPy SGEMM (matching PyTorch SentenceTransformers to `0.999999+` cosine similarity) alongside `onnxruntime` CPU Whisper STT, keeping `GHostPCP.exe` lean and preserving 100% of GPU VRAM for `llama.cpp`.
- **Single-File Native Frontend & OpenAI-Compatible Backend**: Exposes `/v1/chat/completions`, `/v1/models`, `/v1/preflight`, `/v1/load`, `/v1/personas/*`, `/v1/pcp/*`, `/v1/tts`, and `/v1/stt` from a self-contained FastAPI + PyWebView desktop server synced live with mobile clients.
- **Explicit Hardware & Concurrency Guards**: `EngineState` atomic generation/load leases prevent mid-stream unload/reset/persona races; `gguf` tensor-index inspection runs before native model allocation and falls back cleanly on OOM.

---

## Quick Start

> **Important:** Imports append volumes by default. If you select **Replace
> existing archive volumes**, the app backs up existing volumes to
> `pcp_data/backups/` before replacing them. Keep your own backup of the entire
> `pcp_data` folder, too. To keep fictional and personal memories separate,
> try the sample in a separate, clean installation.

### Option A: Download the Standalone Windows Release (Recommended)
1. Download and extract **[`GHostPCP-v1.4.0-win64.zip` (latest release)](https://github.com/realkellyspear/GHostPCP/releases/latest)** from Releases.
2. Double-click **`GHostPCP.exe`** — drop your own `.gguf` (single or split) and optional vision `mmproj*.gguf` files into `ghost_models/`, or download models directly inside **[ CONFIG ] → Get More GGUF Models**.
3. Drop your AI export (ChatGPT, Google Gemini Takeout, Claude, Grok, Character.AI, `.pdf`, `.txt`, or `.md`, or the included `sample_chatgpt_export.zip`) into **[ ARCHIVE ]**, and fill in your AI's name, what it calls you, and core memories it should never forget.
4. In **[ CONFIG ]**, select your model and click **MOUNT** (on future launches, GHostPCP auto-mounts your active persona's preferred model or last-used model automatically).
5. Open **[ MEMORIES ]** anytime to inspect or edit volumes, add Core Facts with relationship aliases, bookmark/reload threads, or export a `.zip` backup bundle. Toggle **`WEB: ON`** when you want live internet search or link reading, or paste/attach images with **`[ + IMG ]`** on Vision models.
6. *(Optional)* Click **[ PHONE / TUNNEL ]** and scan the QR code to pair your phone over Wi-Fi, Tailscale, or a 1-click Cloudflare HTTPS Quick Tunnel.
7. Responses stream as they are generated. Click **Stop** on the active response to interrupt generation.

### Option B: Run from Python Source
Install Python and dependencies, and supply your own GGUF model (set `GHOST_BACKEND=cuda|vulkan|cpu` to override hardware backend auto-detection, or `GHOST_NO_AUTOLOAD=1` to skip startup auto-mount). The first run may need an internet connection to download the embedding model; model weights and voice files (besides Silas voice and Kara voice) are not included.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe ghost_pcp.py
```

To run headless (API + Web UI without opening the desktop window):
```powershell
& .\.venv\Scripts\python.exe ghost_pcp.py --api-only
```

To run the automated test suite:
```powershell
& .\.venv\Scripts\python.exe -m unittest tests/test_pcp.py -v
```

### Try the Fictional Sample

Download [`sample_chatgpt_export.zip`](examples/sample_chatgpt_export.zip) and
import it into a clean installation using **[ ARCHIVE ]**. This small, fictional
ChatGPT-format export lets you try the workflow without using personal data.
After mounting a model, try asking:

- "What is our callsign and the golden rule on the Station 9 observation deck?"
- "What happened during the Europa relay blackout?"

The sample is not a benchmark: responses can vary and may contain errors.
Later imports append to the sample unless you select **Replace existing archive
volumes**.

---

## Building `GHostPCP.exe` from Source

```powershell
& .\.venv\Scripts\python.exe build_pcp.py
```

---

## Privacy & Security

- Your archive volumes (`pcp_data/ghost_pcp_*.md`, `pcp_data/personas/`), vector index (`embeddings.npz`), models (`*.gguf`), and settings (`ghost_pcp_settings.json`) are excluded by `.gitignore`. Check what you stage before committing; ignore rules are not a guarantee against uploading private data.
- Desktop loopback (`127.0.0.1`) works automatically. Any remote connection (LAN Wi-Fi, Tailscale, or Cloudflare Tunnel) is gated by a 192-bit cryptographic pairing token embedded in the QR code link (`/#<token>`). You can rotate the pairing code at any time in **[ PHONE / TUNNEL ]**.
- Same-Wi-Fi access uses unencrypted HTTP. Use a trusted network, keep the pairing link private, and rotate it if exposed. Cloudflare Tunnel is optional and uses an external service; model and dependency downloads also require internet access.

## License

GHostPCP's application code is licensed under the **GNU General Public License
v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE). The standalone
Windows release includes its first-party corresponding source in the `source/`
directory, including the application, UI, and build scripts.

The release also bundles third-party components and models under their own
licenses; those retain their original terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
