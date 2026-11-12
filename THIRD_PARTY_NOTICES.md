# Third-Party Notices & Licenses

**GHost PCP** source code is licensed under the MIT License (see `LICENSE`).

The standalone Windows release bundle (`GHost-PCP-v1.0.0-win64.zip`) includes the following third-party open-source components and pre-trained models, governed by their respective licenses:

---

## 1. Piper TTS & eSpeak NG (`piper-tts` 1.8.0 / `espeak-ng`)

- **Project**: `piper1-gpl` (`piper-tts`) & `espeak-ng`
- **Authors**: The Home Assistant Authors / Open Home Foundation & eSpeak NG contributors
- **Source Code**:
  - https://github.com/OHF-Voice/piper1-gpl
  - https://github.com/espeak-ng/espeak-ng
  - GHost PCP complete source code and build scripts (`build_pcp.py`, `package_release.py`, `requirements.txt`): https://github.com/realkellyspear/GHost-PCP
- **License**: **GPL-3.0-or-later** (GNU General Public License v3.0 or later — full text included in `LICENSE.GPL-3.0`).

---

## 2. Piper Voice Models (`piper_models/`)

The standalone Windows release bundles (or downloads on demand) pre-trained voice models from the official [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) repository (exact upstream snapshot commit [`e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/tree/e7229f7d33)):

- **`silas.onnx` & `silas.onnx.json`**
  - **Upstream Model**: [`en/en_US/ryan/high/en_US-ryan-high.onnx`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/ryan/high/en_US-ryan-high.onnx) (`SHA256: b3990d7606e183ec8dbfba70a4607074f162de1a0c412e0180d1ff60bb154eca`, `120,786,792` bytes)
  - **Model Card**: https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/ryan/high/MODEL_CARD
  - **Dataset**: [RyanSpeech](https://www.kaggle.com/datasets/roholazandie/ryanspeech)
  - **License**: **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International — https://creativecommons.org/licenses/by-nc-sa/4.0/)

- **`kara.onnx` & `kara.onnx.json`**
  - **Upstream Model**: [`en/en_US/amy/medium/en_US-amy-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/amy/medium/en_US-amy-medium.onnx) (`SHA256: b3a6e47b57b8c7fbe6a0ce2518161a50f59a9cdd8a50835c02cb02bdd6206c18`, `63,201,294` bytes)
  - **Model Card**: https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/amy/medium/MODEL_CARD
  - **Dataset / Upstream**: https://github.com/MycroftAI/mimic3-voices (licensed under **CC BY-SA 4.0** — https://github.com/MycroftAI/mimic3-voices/blob/master/LICENSE; finetuned from U.S. English `lessac` medium voice, [Blizzard Challenge 2013 Lessac dataset](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html)).

*(Note: Because `silas.onnx` (`en_US-ryan-high`) and `kara.onnx` (`en_US-amy-medium` finetuned from `lessac`) carry non-commercial dataset terms, the bundled `piper_models/` files are provided strictly for **non-commercial use**. Users may delete or replace the `.onnx` files in `piper_models/` at any time.)*

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
- **FastAPI, Uvicorn, Pydantic, NumPy, Safetensors, Tokenizers, SoundDevice, SoundFile (`libsndfile`, LGPL-2.1), PyWebView, QRCode**: Distributed under their respective open-source licenses (MIT / BSD / Apache-2.0 / LGPL-2.1).
