![GHostPCP — GPU Hosted Pure Concentrated Personality](GHostPCP%20Promo/1github_banner.jpg)

# GHostPCP

**GPU Hosted Pure Concentrated Personality**  
*Break Your AI Out of the Cloud and Onto Your Own Hardware*

GHostPCP is a local-first desktop and mobile-paired AI workbench for Windows. Import your conversation archives from **OpenAI ChatGPT**, **Google Gemini**, **Anthropic Claude**, **xAI Grok**, or **Character.AI**, along with `.pdf`, `.txt`, or `.md` documents; organize isolated AI personas with structured core facts, evolving continuity logs, saved threads, and editable Markdown memory volumes; download or mount local `.gguf` text or **Vision (`mmproj`)** models; and optionally toggle live **Web Search & URL Reading** whenever you want your AI connected to the world.

![GHostPCP desktop interface](GHostPCP%20Promo/ghost_pcp_first_screen.png)

---

## Why GHostPCP Exists

![Why GHostPCP](GHostPCP%20Promo/why_gpcp.png)

Standard "chat with your docs" tools act like research librarians (*"According to source document 4..."*). GHostPCP gives every AI persona its own second-brain memory folder and grounds responses naturally in character—with hybrid exact-name + semantic recall, durable relationships, and portable `.zip` backups you own outright.

1. **Universal Archive Ingestion & Multi-Persona Isolation**: Drop your **OpenAI ChatGPT** (`.zip` / `conversations.json`), **Google Gemini Takeout** (`MyActivity.json` / `MyActivity.html`), **Anthropic Claude**, **xAI Grok**, **Character.AI**, or `.pdf` / `.txt` / `.md` files into **[ ARCHIVE ]**. GHostPCP auto-detects the format, deduplicates repeated conversations, removes canned corporate refusals (*"As an AI language model..."*), creates an isolated folder per persona (`pcp_data/personas/<id>/`), writes chronological `ghost_pcp_*.md` volumes, and builds a local **BGE + BM25 hybrid index** (`BAAI/bge-small-en-v1.5`).
2. **True Continuity & Second-Brain Memory Tools (`[ MEMORIES ]`)**: Save structured **Core Facts & People** with relationship aliases (e.g. linking *"Jordan"* to *"my brother"*) so indirect questions still recall accurately, capture evolving session **Continuity Notes** automatically on **SAVE & CLEAR**, bookmark and reload named conversation **Threads**, search or view/edit/delete individual `ghost_pcp_*.md` volumes (with automatic backups), and export or restore any persona as a single **`.zip` backup bundle**.
3. **Local Vision (`[ + IMG ]` + `mmproj`) & In-App Model Downloader**: Attach or paste (`Ctrl+V`) images for vision-capable GGUF models paired with an `mmproj` projector via `llama.cpp` MTMD. Don't have a model yet? Open **[ CONFIG ] → Get More GGUF Models** to download curated starter models or inspect and pull `.gguf` files from any public HuggingFace repository straight into `./ghost_models`, complete with NVIDIA VRAM preflight sizing, FP16/Q8_0 KV cache selection, and live `tok/s` telemetry.
4. **Opt-In Live Web Search & URL Reader (`WEB: OFF / ON`)**: Strictly **OFF by default** for 100% offline privacy. Flip **`WEB: ON`** in the chat bar to let your AI read pasted `http(s)://` links or perform fast DuckDuckGo/Wikipedia (no API key needed), self-hosted SearXNG, Brave, or Tavily lookups with clickable source chips below replies and SSRF protection against local network URLs.
5. **Two-Way Local Voice (`[ MIC ]` & Piper TTS) + Phone Tunnel Anywhere**: Dictate offline with **`[ MIC ]`** powered by bundled **Moonshine Base ONNX** (`0 GB` GPU VRAM — works natively inside the desktop app via `sounddevice` and from paired HTTPS phone browsers) and click the speaker button on replies for instant local **Piper ONNX** speech synthesis (`Silas`, `Kara`, or any `.onnx` + `.onnx.json` voice in `piper_models/`). Open **[ PHONE / TUNNEL ]** to scan a QR code for encrypted mobile access via local Wi-Fi, Tailscale, or a one-click Cloudflare Quick Tunnel.

## See It in Action

**Desktop Workspace**

<img src="GHostPCP%20Promo/pcp_welcome.png" alt="GHostPCP desktop workspace" width="700">

**Archive Import & Indexing**

<img src="GHostPCP%20Promo/ingest_screen.png" alt="GHostPCP archive import and indexing" width="600">

**Phone & Cloudflare Quick Tunnel**

<img src="GHostPCP%20Promo/cloudflare_page.png" alt="GHostPCP Cloudflare remote mobile connection" width="400"> <img src="GHostPCP%20Promo/pcp_phone.png" alt="GHostPCP mobile chat interface" width="260">

---

## Folder Layout

```text
GHostPCP/
GHostPCP.exe                 Standalone desktop application (built locally)
  ghost_pcp.py                 FastAPI + llama.cpp + vision & PCP RAG server
  pcp_ingest.py                Universal importer (ChatGPT, Gemini, Claude, Grok, Character.AI, docs)
  pcp_retrieval.py             Hybrid BGE-small-en-v1.5 + BM25 indexer & volume manager
  pcp_web.py                   Opt-in web search (DDG/Wikipedia/SearXNG/Brave/Tavily) & SSRF-safe URL reader
  persona.py                   Grounding prefix, core facts/aliases formatter & filter defaults
  ghost_pcp.html               Desktop & mobile-responsive interface (Vision, Web toggle, Memories, Mic)
  build_pcp.py                 PyInstaller executable builder
  ghost_models/                Place your local .gguf models here
    README.md
  piper_models/                Place local Piper TTS voice models here
  stt_model/                   Bundled offline Moonshine-Base int8 ONNX STT model (0 VRAM)
  pcp_data/                    Generated ghost_pcp_01.md..N volumes & embeddings.npz (git-ignored)
  assets/                      GHost circle logo and icons
```

---

## Quick Start

> **Important:** Imports append volumes by default. If you select **Replace
> existing archive volumes**, the app backs up existing volumes to
> `pcp_data/backups/` before replacing them. Keep your own backup of the entire
> `pcp_data` folder, too. To keep fictional and personal memories separate,
> try the sample in a separate, clean installation.

### Option A: Download the Standalone Windows Release (Recommended)
1. Download and extract **[`GHostPCP-v1.3.0-win64.zip`](https://github.com/realkellyspear/GHostPCP/releases/latest)** from Releases.
2. Double-click **`GHostPCP.exe`** — drop your own `.gguf` (and optional vision `mmproj*.gguf`) files into `ghost_models/`, or download starter models directly inside **[ CONFIG ] → Get More GGUF Models**.
3. Drop your AI export (ChatGPT, Google Gemini Takeout, Claude, Grok, Character.AI, `.pdf`, `.txt`, or `.md`, or the included `sample_chatgpt_export.zip`) into **[ ARCHIVE ]**, and fill in your AI's name, what it calls you, and core memories it should never forget.
4. In **[ CONFIG ]**, select your model and click **MOUNT**.
5. Open **[ MEMORIES ]** anytime to inspect or edit volumes, add Core Facts with relationship aliases, bookmark/reload threads, or export a `.zip` backup bundle. Toggle **`WEB: ON`** when you want live internet search or link reading, or paste/attach images with **`[ + IMG ]`** on Vision models.
6. *(Optional)* Click **[ PHONE / TUNNEL ]** and scan the QR code to pair your phone over Wi-Fi, Tailscale, or a 1-click Cloudflare HTTPS Quick Tunnel.
7. Responses stream as they are generated. Click **Stop** on the active response to interrupt generation.

### Option B: Run from Python Source
Install Python and dependencies, and supply your own GGUF model. The first run may need an internet connection to download the embedding model; model weights and voice files (besides Silas voice and Kara voice) are not included.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe ghost_pcp.py
```

To run headless (API + Web UI without opening the desktop window):
```powershell
& .\.venv\Scripts\python.exe ghost_pcp.py --api-only
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
