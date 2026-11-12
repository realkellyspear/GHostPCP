# GHost PCP — Local Read-Aloud Voices (`piper_models`)

Drop your local Piper TTS `.onnx` and `.onnx.json` voice model pairs into this directory.

Default voices supported out of the box:
- `silas.onnx` + `silas.onnx.json`
- `kara.onnx` + `kara.onnx.json`

Any `.onnx` model placed in this folder will automatically appear in the **Read-Aloud Voice** dropdown inside **[ CONFIG ]**, and will be used when clicking the speaker icon on the model's responses.

## Voice Model Attribution & Licenses

- `silas.onnx` (`rhasspy/piper-voices` [`en_US-ryan-high` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/ryan/high/MODEL_CARD), `SHA256: b3990d7606e183ec8dbfba70a4607074f162de1a0c412e0180d1ff60bb154eca`): Trained on the [RyanSpeech dataset](https://www.kaggle.com/datasets/roholazandie/ryanspeech) under **CC BY-NC-SA 4.0**.
- `kara.onnx` (`rhasspy/piper-voices` [`en_US-kristin-medium` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/kristin/medium/MODEL_CARD), `SHA256: 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`): Trained from scratch on [LibriVox](https://librivox.org) recordings under **Public Domain**.

If `silas.onnx` or `kara.onnx` are not already present in `piper_models/`, GHost PCP will automatically download them on demand from `rhasspy/piper-voices` on Hugging Face the first time you click the speaker button. See `THIRD_PARTY_NOTICES.md` for full details.
