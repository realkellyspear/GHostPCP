# Local Speech-to-Text (`stt_model/`)

This folder holds the quantized `onnx-community/moonshine-base-ONNX` speech recognition weights (`encoder_model_int8.onnx`, `decoder_model_int8.onnx`, `tokenizer.json`, `config.json`, MIT License) used by GHostPCP for 100% offline microphone dictation (`[ MIC ]`) on CPU with zero GPU VRAM usage.

If running directly from source without the release archive, GHostPCP downloads these ~63 MB files automatically from Hugging Face on first use of `[ MIC ]`.
