# GHost PCP — Local Read-Aloud Voices (`piper_models`)

Drop your local Piper TTS `.onnx` and `.onnx.json` voice model pairs into this directory.

Default voices supported out of the box:
- `silas.onnx` + `silas.onnx.json`
- `kara.onnx` + `kara.onnx.json`

Any `.onnx` model placed in this folder will automatically appear in the **Read-Aloud Voice** dropdown inside **[ CONFIG ]**, and will be used when clicking the speaker icon on the model's responses.

## Voice Model Attribution & Licenses

- `silas.onnx` (`rhasspy/piper-voices` [`en_US-john-medium` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/john/medium/MODEL_CARD), `SHA256: 789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988`): Trained on [LibriVox](https://librivox.org) recordings (finetuned from `en_US-kristin-medium`) under **Public Domain**.
- `kara.onnx` (`rhasspy/piper-voices` [`en_US-kristin-medium` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/kristin/medium/MODEL_CARD), `SHA256: 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`): Trained from scratch on [LibriVox](https://librivox.org) recordings under **Public Domain**.

If `silas.onnx` or `kara.onnx` are not already present in `piper_models/`, GHost PCP will automatically download them on demand from `rhasspy/piper-voices` on Hugging Face the first time you click the speaker button. See `THIRD_PARTY_NOTICES.md` for full details.
