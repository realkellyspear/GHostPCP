![GHostPCP — GPU Hosted Pure Concentrated Personality](GHostPCP%20Promo/1github_banner.jpg)

# GHostPCP

**GPU Hosted Pure Concentrated Personality**  
*Break Your AI Out of the Cloud and Onto Your Own Hardware*

GHostPCP is a local-first desktop and mobile-paired application for Windows. Import your OpenAI ChatGPT export (`.zip` or `conversations.json`) or Google Gemini Takeout export (`.zip`, `MyActivity.json`, `MyActivity.html`, or source folder), along with `.pdf`, `.txt`, or `.md` documents, save multiple isolated AI personas with their own folders, mount a local `.gguf` language model, and chat with a persona grounded in your conversation history.

![GHostPCP desktop interface](GHostPCP%20Promo/ghost_pcp_first_screen.png)

---

## Why GHostPCP Exists

![Why GHostPCP](GHostPCP%20Promo/why_gpcp.png)

Standard "chat with your docs" tools act like research librarians (*"According to source document 4..."*). GHostPCP prompts the model to treat your archive and attached files as the authoritative source for the persona's history and respond without citations. Like any language model, it can still make mistakes or invent details.

1. **Drop & Forget Onboarding (ChatGPT, Google Gemini, PDFs & Multiple Personas)**: Drop your ChatGPT export, Google Gemini Takeout (`MyActivity.json`, `MyActivity.html`, `.zip`, or source folder), `.pdf`, `.txt`, or `.md` files into **[ ARCHIVE ]** — GHostPCP auto-detects Google vs OpenAI, asks you who the AI is (name, what it calls you, core anchors it should never forget), creates an isolated folder per persona (`pcp_data/personas/<id>/`), strips out canned corporate refusals (*"As an AI language model..."*), organizes everything into clean Markdown volumes (`ghost_pcp_01.md`, `ghost_pcp_02.md`, ...), and builds a cached semantic vector index (`embeddings.npz`) using `BAAI/bge-small-en-v1.5`. Choose which AI you want to talk to on startup or switch anytime from the header.
2. **Hardware-Aware GGUF Workbench**: Powered by `llama.cpp` with automatic NVIDIA VRAM preflight sizing (`Check Fit` / `Mount` / `Unload`), FP16/Q8_0 KV cache selection, live token streaming, a per-turn **Stop** control, and sampling controls.
3. **In-Chat File Uploads, Persona Grounding & Save & Clear Continuity**: Every turn retrieves the most relevant passages from your `ghost_pcp_*.md` volumes in ~24ms and grounds the model silently in character. Use **`[ + FILE ]`** right in the chat bar (desktop or phone) to attach `.pdf`, `.txt`, or `.md` files as collapsible attachment cards—with one-click **SAVE TO DB** (and automatic indexing for large documents). Click **SAVE & CLEAR** at the top of the window to save the current chat thread with a timestamp directly into your PCP archive (`ghost_pcp_NN.md`), update the vector index in ~0.13s, and clear the window so the conversation continues without missing a beat.
4. **Built-In Fast Local Read-Aloud Voices**: Choose your preferred voice (such as **Silas** or **Kara**, or any `.onnx` voice dropped into `piper_models/`) in **[ CONFIG ]**, and click the speaker icon on any of the model's responses for instant local Piper ONNX synthesis (~0.17s latency) and seamless background playback via `sounddevice` on PC (or Web Audio when connected from your phone).
5. **Use Anywhere (Wi-Fi, Tailscale, or One-Click HTTPS Tunnel)**: Open **[ PHONE / TUNNEL ]** to scan a pairing QR code on your phone over local Wi-Fi, Tailscale, or a one-click Cloudflare HTTPS Quick Tunnel. Your phone gets the full interface—chat, attach files, listen to voice responses, start a new session, or switch models remotely.

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
  ghost_pcp.py                 FastAPI + llama.cpp + PCP RAG server & desktop host
  pcp_ingest.py                Auto-detecting Google Gemini & OpenAI ChatGPT parser, filter, & volume writer
  pcp_retrieval.py             Local BGE-small-en-v1.5 semantic indexer & search engine
  persona.py                   Universal citation-free grounding prompt & filter defaults
  ghost_pcp.html               Desktop & mobile-responsive interface
  build_pcp.py                 PyInstaller executable builder
  ghost_models/                Place your local .gguf models here
    README.md
  piper_models/                Place local Piper TTS voice models here
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
1. Download and extract **[`GHostPCP-v1.2.0-win64.zip`](https://github.com/realkellyspear/GHostPCP/releases/latest)** from Releases.
2. Copy one or more `.gguf` models (for example, `Qwen3-14B-Q5_K_M.gguf`) into `ghost_models/`.
3. Double-click **`GHostPCP.exe`**.
4. Drop your ChatGPT export (`.zip` or `conversations.json`), Google Gemini Takeout export (`.zip`, `MyActivity.json`, `MyActivity.html`, or source folder), `.pdf`, `.txt`, or `.md` files (or the included `sample_chatgpt_export.zip`) into the **Archive Import** window, and fill in your AI's name, what it calls you, and core memories it should never forget.
5. In **[ CONFIG ]**, select your model and click **MOUNT**.
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
