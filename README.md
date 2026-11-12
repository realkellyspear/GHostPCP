![GHost PCP — GPU Hosted Pure Concentrated Personality](GHostPCP%20Promo/pcp_banner_fixed.png)

# GHost PCP

**GPU Hosted Pure Concentrated Personality**  
*Break Your AI Out of the Cloud and Onto Your Own Hardware*

GHost PCP is a local-first desktop and mobile-paired application for Windows. Import your ChatGPT export archive (`.zip` or `conversations.json`), mount a local `.gguf` language model, and chat with a persona grounded in your conversation history.

![GHost PCP desktop interface](GHostPCP%20Promo/save_clear.png)

---

## Why GHost PCP Exists

Standard "chat with your docs" tools act like research librarians (*"According to source document 4..."*). GHost PCP prompts the model to treat your archive as the authoritative source for the persona's history and respond without citations. Like any language model, it can still make mistakes or invent details.

1. **Drop & Forget Onboarding**: Drop your ChatGPT export `.zip` (or `conversations.json`) into the app. In the background, GHost PCP unpacks the conversation tree, strips out canned corporate refusals (*"As an AI language model..."*), organizes your history into clean Markdown volumes (`ghost_pcp_01.md`, `ghost_pcp_02.md`, ...), and builds a cached semantic vector index (`embeddings.npz`) using `BAAI/bge-small-en-v1.5`.
2. **Hardware-Aware GGUF Workbench**: Powered by `llama.cpp` with automatic NVIDIA VRAM preflight sizing (`Check Fit` / `Mount` / `Unload`), FP16/Q8_0 KV cache selection, and full sampling control.
3. **Citation-Free Persona Grounding & Save & Clear Continuity**: Every turn retrieves the most relevant passages from your `ghost_pcp_*.md` volumes in ~24ms and grounds the model silently in character. Click **SAVE & CLEAR** at the top of the window to save the current chat thread with a timestamp directly into your PCP archive (`ghost_pcp_NN.md`), update the vector index in ~0.13s, and clear the window so the conversation continues without missing a beat.
4. **Built-In Fast Local Read-Aloud Voices**: Choose your preferred voice (such as **Silas** or **Kara**, or any `.onnx` voice dropped into `piper_models/`) in **[ CONFIG ]**, and click the speaker icon on any of the model's responses for instant local Piper ONNX synthesis (~0.17s latency) and seamless background playback via `sounddevice` on PC (or Web Audio when connected from your phone).
5. **Use Anywhere (Wi-Fi, Tailscale, or One-Click HTTPS Tunnel)**: Open **[ PHONE / TUNNEL ]** to scan a pairing QR code on your phone over local Wi-Fi, Tailscale, or a one-click Cloudflare HTTPS Quick Tunnel. Your phone gets the full interface—chat, listen to voice responses, start a new session, or switch models remotely.

## See It in Action

**Model and persona configuration**

<img src="GHostPCP%20Promo/persona_config.png" alt="GHost PCP model and persona configuration" width="600">

**Phone interface**

<img src="GHostPCP%20Promo/pcp_phone_cropped.png" alt="GHost PCP phone interface" width="300">

---

## Folder Layout

```text
GHost-PCP/
  GHost-PCP.exe                Standalone desktop application (built locally)
  ghost_pcp.py                 FastAPI + llama.cpp + PCP RAG server & desktop host
  pcp_ingest.py                ChatGPT export parser, corporate filter, & volume writer
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

> **Important:** Importing an archive replaces existing `ghost_pcp_*.md`
> volumes in this installation, including saved sessions. Back up your entire
> `pcp_data` folder first. Try the sample in a separate, clean installation,
> not one containing memories you want to keep.

### Option A: Run a Locally Built Executable
An executable is not included in this repository. Build it using the instructions below first.
1. Copy one or more `.gguf` models (for example, `Qwen3-14B-Q5_K_M.gguf`) into `ghost_models/`.
2. Double-click **`GHost-PCP.exe`**.
3. If you haven't imported an archive yet, drop your ChatGPT export `.zip` or `conversations.json` into the **Archive Import** window.
4. In **[ CONFIG ]**, select your model and click **MOUNT**.

### Option B: Run from Python Source
Install Python and dependencies, and supply your own GGUF model. The first run may need an internet connection to download the embedding model; model weights and voice files are not included.

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
Importing a different archive later replaces the sample volumes.

---

## Building `GHost-PCP.exe` from Source

```powershell
& .\.venv\Scripts\python.exe build_pcp.py
```

---

## Privacy & Security

- Your archive volumes (`pcp_data/ghost_pcp_*.md`), vector index (`embeddings.npz`), models (`*.gguf`), and settings (`ghost_pcp_settings.json`) are excluded by `.gitignore`. Check what you stage before committing; ignore rules are not a guarantee against uploading private data.
- Desktop loopback (`127.0.0.1`) works automatically. Any remote connection (LAN Wi-Fi, Tailscale, or Cloudflare Tunnel) is gated by a 192-bit cryptographic pairing token embedded in the QR code link (`/#<token>`). You can rotate the pairing code at any time in **[ PHONE / TUNNEL ]**.
- Same-Wi-Fi access uses unencrypted HTTP. Use a trusted network, keep the pairing link private, and rotate it if exposed. Cloudflare Tunnel is optional and uses an external service; model and dependency downloads also require internet access.

## License

MIT. See [LICENSE](LICENSE).
