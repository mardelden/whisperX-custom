# WhisperX

## Build & Dev Commands

```bash
# Install (dev)
uv sync --all-extras --dev

# Run CLI
uvx whisperx --model large-v3 --language en audio.wav

# Install from source
pip install -e .

# Run directly
python -m whisperx audio.wav --model large-v3
```

## Architecture

3-stage pipeline: **VAD + ASR → Align → Diarize**

1. **VAD + ASR** (`asr.py`, `vads/`): Voice activity detection segments audio, then batched Whisper transcription via `faster-whisper` / CTranslate2. VAD methods: pyannote (default) or silero.
2. **Alignment** (`alignment.py`): Forced phoneme-level alignment using wav2vec2 models. Per-language model selection from torchaudio (en/fr/de/es/it) or HuggingFace (27+ languages). Produces word-level timestamps.
3. **Diarization** (`diarize.py`): Speaker diarization via `pyannote.audio`. Assigns speaker labels to word segments. Requires HuggingFace token.

Entry points: CLI via `__main__.py` → `transcribe.py:transcribe_task()`. Python API via lazy imports in `__init__.py`.

## Module Structure

```
whisperx/
├── __init__.py          # Lazy import public API
├── __main__.py          # CLI entry point (argparse)
├── transcribe.py        # Pipeline orchestration (transcribe_task)
├── asr.py               # FasterWhisperPipeline, WhisperModel (batched)
├── alignment.py         # load_align_model(), align()
├── diarize.py           # DiarizationPipeline, assign_word_speakers()
├── audio.py             # load_audio() via ffmpeg, mel spectrogram
├── schema.py            # TypedDicts: TranscriptionResult, AlignedTranscriptionResult
├── utils.py             # Language defs, LANGUAGES dict, writer utilities
├── log_utils.py         # Logging config
├── conjunctions.py      # Language-specific conjunctions
├── SubtitlesProcessor.py # SRT/VTT output formatting
└── vads/
    ├── __init__.py      # Exports: Vad, Silero, Pyannote
    ├── vad.py           # Base Vad class, merge_chunks()
    ├── pyannote.py      # Pyannote VAD (default, uses local assets/pytorch_model.bin)
    └── silero.py        # Silero VAD (lighter, via torch.hub)
```

## Key Patterns

- **Lazy loading**: `__init__.py` defers all imports until accessed
- **Pipeline architecture**: Each stage independent, optional (--no_align, --diarize flag)
- **Language-aware processing**: Alignment model selected per detected language; sentence tokenization uses language-specific NLTK tokenizers
- **Memory optimization**: GPU memory cleared between stages
- **Batched inference**: ASR processes VAD chunks in configurable batch sizes

## Critical Rules

- **Pyannote auth**: `DiarizationPipeline` no longer takes a token parameter — relies on `HF_TOKEN` env var. pyannote 4.x uses `token=` internally.
- **Pyannote 4.x**: Uses `pyannote-audio>=4.0.0`. Returns `DiarizeOutput` — extract `.speaker_diarization` attribute.
- **Python support**: 3.10–3.13 (`>=3.10, <3.14`). Dev default: 3.10 (`.python-version`).
- **PyTorch**: Pinned to `~=2.8.0`. macOS uses CPU-only builds.
- **Audio codec**: `av<16.0.0` constraint.
- **Version**: 3.7.4 (in `pyproject.toml`).
