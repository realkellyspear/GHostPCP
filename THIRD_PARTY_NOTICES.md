# Third-Party Notices & Licenses

**GHostPCP** first-party application code is licensed under the GNU General
Public License v3.0 or later (see `LICENSE`). Bundled third-party components
and models retain their own licenses.

The standalone Windows release bundle (`GHostPCP-v1.1.0-win64.zip`) includes the following third-party open-source components and pre-trained models, governed by their respective licenses:

---

## 1. Piper TTS & eSpeak NG (`piper-tts` 1.8.0 / `espeak-ng`)

- **Project**: `piper1-gpl` (`piper-tts`) & `espeak-ng`
- **Authors**: The Home Assistant Authors / Open Home Foundation & eSpeak NG contributors
- **Source Code**:
  - https://github.com/OHF-Voice/piper1-gpl
  - https://github.com/espeak-ng/espeak-ng
  - GHostPCP complete source code and build scripts (`build_pcp.py`, `package_release.py`, `requirements.txt`): https://github.com/realkellyspear/GHost-PCP
- **License**: **GPL-3.0-or-later** (GNU General Public License v3.0 or later — full text included in `LICENSE`).

---

## 2. Piper Voice Models (`piper_models/`)

The standalone Windows release bundles (or downloads on demand) pre-trained voice models from the official [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) repository (exact upstream snapshot commit [`e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/tree/e7229f7d33)):

- **`silas.onnx`, `silas.onnx.json`, & `silas.MODEL_CARD`**
  - **Upstream Model**: [`en/en_US/john/medium/en_US-john-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/john/medium/en_US-john-medium.onnx) (`SHA256: 789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988`, `63,531,379` bytes)
  - **Model Card**: https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/john/medium/MODEL_CARD (bundled at `piper_models/silas.MODEL_CARD`)
  - **Dataset / Training**: [LibriVox](https://librivox.org) (`https://brycebeattie.com/files/tts/` — finetuned from Public Domain `en_US-kristin-medium` on Public Domain LibriVox recordings; listed on Piper's official [Commercial & Permissive Use list](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/COMMERCIAL_USE.md)).
  - **License**: **Public Domain**

- **`kara.onnx`, `kara.onnx.json`, & `kara.MODEL_CARD`**
  - **Upstream Model**: [`en/en_US/kristin/medium/en_US-kristin-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/kristin/medium/en_US-kristin-medium.onnx) (`SHA256: 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`, `63,531,379` bytes)
  - **Model Card**: https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/kristin/medium/MODEL_CARD (bundled at `piper_models/kara.MODEL_CARD`)
  - **Dataset / Training**: [LibriVox](https://librivox.org) (`https://brycebeattie.com/files/tts/` — trained from scratch on Public Domain LibriVox recordings; listed on Piper's official [Commercial & Permissive Use list](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/COMMERCIAL_USE.md)).
  - **License**: **Public Domain**

*(Both bundled voice models have an entire training chain consisting exclusively of Public Domain LibriVox recordings, with their upstream `MODEL_CARD` files included alongside the `.onnx` files in `piper_models/`. Users may delete or replace the `.onnx` files in `piper_models/` at any time.)*

---

## 3. Semantic Embedding Model (`bge_model/`)

- **`BAAI/bge-small-en-v1.5`** (`model.safetensors`, `tokenizer.json`, `vocab.txt`, `config.json`, `special_tokens_map.json`):
  - **Author**: Beijing Academy of Artificial Intelligence (BAAI)
  - **Source**: https://huggingface.co/BAAI/bge-small-en-v1.5
  - **License**: **MIT License**

---

## 4. Additional Runtime Libraries

- **`llama.cpp` / `llama-cpp-python`**: Georgi Gerganov & contributors / Andrei Betlen (**MIT License**)
- **ONNX Runtime (`onnxruntime` 1.30.0)**: Microsoft Corporation (**MIT License**)
- **NVIDIA CUDA Runtime (`cublas64_13.dll`, `cublasLt64_13.dll`)**: NVIDIA Corporation (redistributable runtime components under NVIDIA CUDA Toolkit EULA)
- **`pypdf`**: Matthieu Thillay, Martin Thoma & pypdf contributors (**BSD-3-Clause License**)
- **FastAPI, Uvicorn, Pydantic, NumPy, Safetensors, Tokenizers, SoundDevice, SoundFile (`libsndfile`, LGPL-2.1), PyWebView, QRCode**: Distributed under their respective open-source licenses (MIT / BSD / Apache-2.0 / LGPL-2.1).
