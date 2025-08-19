# ALBANS
---

# AI-Powered Video Transcription and Analysis Toolkit

This repository provides a powerful, local-first toolkit for transcribing video or audio files, identifying speakers, anonymizing sensitive data, and extracting key video clips. The entire workflow is designed to be run on your own machine, ensuring data privacy and control.

It leverages state-of-the-art open-source models like **Faster Whisper** for highly accurate transcription and **WhisperX with Pyannote** for robust speaker diarization.

## Workflow Overview

The toolkit enables a complete analysis pipeline:

1.  **Transcribe & Diarize**: Convert a video file into a structured `.docx` transcript, complete with timestamps and speaker labels (e.g., `SPEAKER_00`, `SPEAKER_01`).
2.  **Anonymize for Privacy**: Process the generated transcript to redact sensitive information (names, companies, etc.), making it safe to use with third-party AI tools like Google's NotebookLM or ChatGPT for analysis and summarization.
3.  **Identify Key Moments**: Use the timestamped transcript to find important highlights, quotes, or findings.
4.  **Extract Video Clips**: Use the video clipping utility to extract these key moments from the original video file, perfect for presentations, reports, or evidence gathering.

## Features

-   **High-Accuracy Transcription**: Powered by `faster-whisper`, an optimized implementation of OpenAI's Whisper model.
-   **Speaker Diarization**: Automatically identifies and labels different speakers throughout the audio using `whisperx` and `pyannote/speaker-diarization-3.1`.
-   **Flexible Export**: Save transcripts as `.txt` or formatted `.docx` files with timestamps and speaker labels.
-   **Data Anonymization**: A utility to find and replace sensitive text in your transcript, protecting privacy before cloud-based analysis.
-   **Precision Video Clipping**: A simple but powerful tool to cut video segments based on the exact start and end times identified in the transcript.
-   **Local & GPU-Accelerated**: Runs entirely on your local machine. It automatically uses a CUDA-enabled GPU if available for massive speed improvements.
-   **Configurable**: Easily change the Whisper model size (`tiny`, `small`, `medium`, `large`) to balance speed and accuracy.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.9+**
2.  **FFmpeg**: This is a critical dependency for audio/video processing.
    -   **Windows**: Download from the [official site](https://ffmpeg.org/download.html) and add the `bin` folder to your system's PATH.
    -   **macOS**: `brew install ffmpeg`
    -   **Linux**: `sudo apt update && sudo apt install ffmpeg`
3.  **PyTorch**: Install it according to your system's configuration (CPU or GPU). Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. For example, for a system with an NVIDIA GPU:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```4.  **(Optional) Git**: For cloning the repository.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -U faster-whisper "whisperx @ git+https://github.com/m-bain/whisperx.git" moviepy python-docx
    ```
    *Note: We install `whisperx` directly from its GitHub repository to ensure the latest version.*

3.  **Hugging Face Token (Required for Speaker Diarization):**
    The `pyannote/speaker-diarization-3.1` model requires you to accept its user agreement on Hugging Face.
    -   Visit the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) page and accept the terms.
    -   Visit the [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) page and accept the terms.
    -   Go to your Hugging Face [Access Tokens page](https://huggingface.co/settings/tokens) and generate a new token with `read` permissions.
    -   Set this token as an environment variable or place it directly in the script.

## Usage

The project is structured into three main parts within the `albans_model.ipynb` notebook.

### Part 1: Video Transcription and Diarization

This is the main script that converts your video into a timestamped document.

**Configuration:**

Open the notebook and edit the parameters in the first code cell:

```python
# ─── PARAMETERS (change these) ─────────────────────────────
VIDEO       = r"C:\path\to\your\video.mp4"
WORK_DIR    = Path(r"C:\path\to\your\outputs")
MODEL_SIZE  = "small"  # Options: "tiny", "small", "medium", "large"
HF_TOKEN    = "YOUR_HUGGING_FACE_TOKEN" # Your token from huggingface.co
```

**Execution:**

Run the cells in sequence. The main execution block performs the full pipeline:

```python
# 1. Initialize the transcriber
tr = LocalWhisperTranscriber(VIDEO, MODEL_SIZE, WORK_DIR)

# 2. Extract the full audio from the video
tr.extract_audio_full()

# 3. Transcribe the audio to text
tr.transcribe()

# 4. Identify and label speakers (e.g., speakers=2 if you know there are two)
tr.diarize(speakers=2, hf_token=HF_TOKEN)

# 5. Export the final transcript to a .docx file
docx_path = tr.export("docx")
print("✅ DOCX generated:", docx_path)
```

### Part 2: Anonymizing the Transcript

This script takes the generated `.docx` file and redacts sensitive information.

**Configuration:**

Set the input/output paths and define your replacement rules:

```python
# 1. Path to the transcript generated in Part 1
archivo_original = r"C:\path\to\your\outputs\your_video_20231027_123456.docx"

# 2. Path for the anonymized output file
archivo_anonimizado = r"C:\path\to\your\outputs\your_video_blindada.docx"

# 3. Define the words/phrases to be replaced
mapa_de_reemplazos = {
    "Project Phoenix": "[PROJECT_NAME]",
    "Jane Doe": "[PERSON_A]",
    "Specific Company Inc.": "[ORGANIZATION]",
    # You can also use regular expressions for patterns like emails or IDs
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[REDACTED_EMAIL]'
}
```

**Execution:**

Run the cell containing the `anonimizar_word` function call. A new, "blinded" document will be created.

### Part 3: Cutting Video Clips

After reviewing your transcript and identifying key moments, use this script to extract the corresponding video clips.

**Configuration:**

Set the video path and define the clips you want to extract. Timestamps are in seconds.

```python
# 1. Path to the original video file
VIDEO_ORIGINAL = r"C:\path\to\your\video.mp4"

# 2. Directory where the clips will be saved
DIRECTORIO_SALIDA = r"C:\path\to\your\outputs\clips"

# 3. Define a list of clips to extract
# You can calculate seconds as: (minutes * 60) + seconds
clips_a_extraer = [
    {
        "nombre": "key_finding_01",
        "inicio": (5 * 60) + 20,  # Clip starts at 5m 20s
        "fin": (6 * 60) + 15       # Clip ends at 6m 15s
    },
    {
        "nombre": "customer_testimonial_01",
        "inicio": (12 * 60) + 3,   # Clip starts at 12m 03s
        "fin": (12 * 60) + 45      # Clip ends at 12m 45s
    }
]
```

**Execution:**

Run the final cell. The script will iterate through your list and save each clip as a separate `.mp4` file in the output directory.

## Acknowledgments

This toolkit is built on the incredible work of the open-source community. Special thanks to:

-   [OpenAI](https://openai.com/) for the original Whisper model.
-   The [faster-whisper](https://github.com/guillaumekln/faster-whisper) team for their optimized implementation.
-   Max Bain for [WhisperX](https://github.com/m-bain/whisperx).
-   The [pyannote.audio](https://github.com/pyannote/pyannote-audio) team for their speaker diarization models.
