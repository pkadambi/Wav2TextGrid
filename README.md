# Wav2TextGrid

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Wav2TextGrid-blue)](https://huggingface.co/pkadambi/wav2textgrid)
![Security](https://github.com/pkadambi/Wav2TextGrid/actions/workflows/security.yml/badge.svg)
![Code Quality](https://github.com/pkadambi/Wav2TextGrid/actions/workflows/code-quality.yml/badge.svg)
![macOS/Windows/Linux Inference](https://github.com/pkadambi/Wav2TextGrid/actions/workflows/inference.yml/badge.svg)

*Version 0.1.2 (Beta)* — A phonetic forced aligner 

Wav2TextGrid has been tested on Ubuntu 18.04, Ubuntu 22.04, Windows 11, and macOS 15.6.1

⚠️ This aligner is currently in development. If you encounter issues, please open an issue or email me directly: pkadambi@asu.edu

## 🚀 Quick Start (User Installation)

### For Users - Install from PyPI

```bash
pip install Wav2TextGrid==0.1.2
```

⚠️ **Note**: The package installs the CPU version of PyTorch by default. Install your preferred PyTorch GPU support (v2.5+) BEFORE installing if you would like to use a GPU.

**Requirements**: Python 3.10+

### Basic Usage

1. **Align a single file**:
   ```bash
   w2tg /path/to/audio.wav /path/to/transcript.lab ./output.TextGrid
   ```

2. **Align an entire directory**:
   ```bash
   w2tg /path/to/audio_dir/ /path/to/transcript_dir/ ./outputs/
   ```

---

## 🛠️ Development Setup

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Clone and Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/pkadambi/Wav2TextGrid.git
cd Wav2TextGrid

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run --only-group dev pre-commit install
```

### Development Commands (Makefile)

The project includes a Makefile with common development tasks:

```bash
# Code formatting
make format          # Format code with Ruff
make format-check    # Check formatting without changes

# Linting
make lint            # Fix linting issues with Ruff  
make lint-check      # Check linting without fixes

# Type checking
make mypy-check      # Run mypy type checking

# Cleanup
make fresh-slate     # Cleans python environment by deleting .venv and uv.lock
```

### Dependency Groups

The project uses `uv` dependency groups defined in `pyproject.toml`:

- **`dev`**: Development tools (pre-commit, ruff, mypy)
- **`security`**: Security scanning tools (safety) ()

```bash
# Install only development dependencies
uv sync --only-group dev

# Install specific groups
uv sync --group dev --group security

# Run commands with specific groups
uv run --only-group dev ruff check .
```

### Pre-commit Hooks

Set up pre-commit to ensure code quality:

```bash
# Install pre-commit hooks (run once)
uv run --only-group dev pre-commit install

# Run pre-commit on all files
uv run --only-group dev pre-commit run --all-files

# Pre-commit will now run automatically on git commits
```


---

## 📁 Data Format and Best Practices

### Recommended Directory Structure

Follow the Kaldi-style organization for best results:

```
/Dataset/folder/
├── Speaker1/
│   ├── file1.lab
│   ├── file1.wav
│   ├── file2.lab
│   ├── file2.wav
├── Speaker2/
│   ├── file3.lab
│   ├── file3.wav
│   ├── file4.lab
│   ├── file4.wav
```

### File Formats

- **Audio files**: `.wav` format (16kHz recommended)
- **Transcript files**: `.lab` format containing plain text transcriptions
- **Output**: `.TextGrid` format compatible with Praat

---

## ⚙️ Training Custom Models

### 2. Train Your Model

```bash
w2tg_train /path/to/training_data/ /path/to/output_model/
```

### 3. Use Trained Model for Alignment

```bash
w2tg /path/to/audio.wav /path/to/transcript.lab ./output.TextGrid --aligner_model /path/to/your_model/
```

---

## 🏗️ Project Structure

```
Wav2TextGrid/
├── src/Wav2TextGrid/           # Main package source
│   ├── aligner_core/           # Core alignment algorithms
│   ├── utils/                  # Utility functions
│   ├── wav2textgrid.py         # Inference interface
│   └── wav2textgrid_train.py   # Training interface
├── scripts/                    # Development scripts
│   ├── run_inference_workflow.py  # CI/CD testing
│   └── test_local.py           # Local testing
├── examples/                   # Example audio/transcript pairs
├── .github/workflows/          # CI/CD pipelines
├── Makefile                    # Development commands
├── pyproject.toml              # Project configuration
└── uv.lock                     # Dependency lock file
```

---

## 🔧 Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Setup firewall using safety**
    =>https://docs.safetycli.com/safety-docs/firewall/introduction-to-safety-firewall


3. **Make your changes** and ensure code quality:
   ```bash
   make format      # Format code
   make lint        # Fix linting issues
   make mypy-check  # Type checking
   ```

4. **Commit with pre-commit checks**:
   ```bash
   git add .
   git commit -m "Your commit message"
   # Pre-commit hooks will run automatically
   ```

### Code Quality Standards

- **Formatting**: Ruff formatter
- **Linting**: Ruff linter with security checks
- **Type checking**: mypy
- **Pre-commit hooks**: Automatic checks on commit
- **CI/CD**: GitHub Actions for multi-platform testing

---

### Local Testing


## 🎵 Supported Formats

### Audio Formats
- **Primary**: `.wav` files (recommended: 16kHz sampling rate)
- **Alternative**: `.mp3` files (specify `--filetype=mp3` for `w2tg`)
- **Training**: Only `.wav` files supported for `w2tg_train`

### Transcript Format
- **File extension**: `.lab`
- **Content**: Single line of text transcript
- **Example**: `SHE HAD YOUR DARK SUIT IN GREASY WASH WATER ALL YEAR`

### TextGrid Format
- **Output**: Praat-compatible `.TextGrid` files
- **Structure**: Contains `IntervalTier` named "phones"
- **Training**: Must include phone-level alignments for training data

---

## ⚠️ Model Scope and Limitations

### Training Data
Wav2TextGrid was trained on:
- **Demographics**: Children ages 3–7 years
- **Dataset**: 3,700 utterances from Test of Childhood Stuttering (TOCS)
- **Duration**: Short utterances (~2–5 seconds)
- **Quality**: Clean, non-conversational audio

### Best Performance
- ✅ Child speech (ages 3–7)
- ✅ Short, clean utterances
- ✅ Read speech (non-conversational)
- ✅ High-quality audio recordings

### Use With Caution
- ⚠️ Adult speech (not validated after fine-tuning)
- ⚠️ Long utterances (>5-10 seconds)
- ⚠️ Conversational or spontaneous speech
- ⚠️ Noisy audio recordings
- ⚠️ Mixed adult/child conversations

**Recommendation**: Always validate alignments when using outside the intended scope.

---

## 🔗 Resources

- **Hugging Face Model**: [pkadambi/wav2textgrid](https://huggingface.co/pkadambi/wav2textgrid)
- **PyPI Package**: [Wav2TextGrid](https://pypi.org/project/Wav2TextGrid/)
- **Issues**: [GitHub Issues](https://github.com/pkadambi/Wav2TextGrid/issues)
- **Contact**: pkadambi@asu.edu

---

## 📄 Citation

If you use Wav2TextGrid in your research, please cite:

```bibtex
@software{kadambi2024wav2textgrid,
  title={Wav2TextGrid: A Phonetic Forced Aligner},
  author={Kadambi, Prad},
  year={2024},
  url={https://github.com/pkadambi/Wav2TextGrid},
  version={0.1.2}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Based on techniques from [Charsiu](https://github.com/lingjzhu/charsiu) by lingjzhu and henrynomeland
- Initialized with models trained on CommonVoice and LibriSpeech datasets
- Fine-tuned on Test of Childhood Stuttering (TOCS) corpus
- [ ] GUI Application for Windows (ongoing, will be complete by 05/31)
- [ ] Add training functionality to GUI (longer term, likely by 06/31) 

## Attribution and Citation

---
- Our paper can be found here: https://doi.org/10.1044/2024_JSLHR-24-00347

BibTeX
```
@article{kadambi2025tunable,
  title={A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech},
  author={Kadambi, Prad and Mahr, Tristan J and Hustad, Katherine C and Berisha, Visar},
  journal={Journal of Speech, Language, and Hearing Research},
  pages={1--19},
  year={2025},
  publisher={American Speech-Language-Hearing Association}
}
```
MLA

 ``` 
 Kadambi, Prad, et al. "A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech." Journal of Speech, Language, and Hearing Research (2025): 1-19.
 ``` 

APA
```
Kadambi, P., Mahr, T. J., Hustad, K. C., & Berisha, V. (2025). A Tunable Forced Alignment System Based on Deep Learning: Applications to Child Speech. Journal of Speech, Language, and Hearing Research, 1-19.
```


------
## Algorithm Details

------
### Pronunciation Dictionary
Uses `g2p_en` (CMUdict) for text-to-phoneme conversion.


### x-vector extraction

1. Voice Activity Detection (VAD): `speechbrain/vad-crdnn-libriparty` (Hugging Face)
2. xVector speaker embeddings: `speechbrain/spkrec-ecapa-voxceleb` (Hugging Face)

### Forced Alignment
- Frame-level phoneme prediction via Wav2Vec2, phoneme posterior distribution

- Alignment via Viterbi decoding (10ms granularity)

## Acknowledgements
---

This work was supported by the National Institute on Deafness and Other Communication Disorders of the National Institutes of Health under award number R01 DC019645.

---
Portions of this project adapt code from the excellent Charsiu aligner https://github.com/lingjzhu/charsiu.



