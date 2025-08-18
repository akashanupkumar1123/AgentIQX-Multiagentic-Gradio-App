# agents/video_agent.py
"""
VideoAgent
----------
Handles transcription of audio and video files using the OpenAI Whisper ASR model.

Responsibilities:
- Lazy-load Whisper model for performance (via shared loader in whisper_utils).
- Allow optional forced language or automatic detection.
- Clean and normalize output transcripts for better readability.
- Save transcripts to disk and return both the saved file path and transcript text.
"""

import os
import logging
import time
from typing import Tuple, Optional
from utils.whisper_utils import get_whisper_model as get_model  # ✅ Shared global loader for Whisper

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[VideoAgent] %(levelname)s: %(message)s'
)
logger = logging.getLogger("VideoAgent")

# -----------------------------------------------------------------------------
# Core function: Transcribe media to text
# -----------------------------------------------------------------------------
def transcribe_audio(file_path: str, language: Optional[str] = None) -> Tuple[str, float]:
    """
    Transcribes audio or video files into text using OpenAI Whisper.

    Args:
        file_path (str):
            Path to the input audio/video file (mp3, wav, mp4, etc.).
        language (Optional[str]):
            ISO language code to force transcription (e.g., "en").
            If None, Whisper auto-detects the language.

    Returns:
        tuple (transcript_text, processing_time_seconds):
            transcript_text (str): Cleaned transcript text.
            processing_time_seconds (float): Duration it took to process.
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return "", 0.0

    # Get cached Whisper model instance
    model = get_model()

    try:
        logger.info(f"Starting transcription for: {file_path}")
        start_time = time.time()

        # Build options for Whisper; include language if provided
        options = {}
        if language:
            options["language"] = language

        # Perform transcription
        result = model.transcribe(file_path, **options)
        elapsed = time.time() - start_time

        # Extract transcription text from result
        full_text = result["text"]

    except Exception as e:
        logger.error(f"Transcription failed for {file_path}: {e}")
        return "", 0.0

    # Normalize whitespace (remove excessive spaces/newlines)
    full_text = " ".join(full_text.split())
    logger.info(f"Transcription completed in {elapsed:.2f} seconds.")
    return full_text, elapsed

# -----------------------------------------------------------------------------
# Helper function: Save transcript to disk
# -----------------------------------------------------------------------------
def save_transcript(text: str, save_path: str):
    """
    Saves a transcript to a UTF-8 encoded text file.

    Args:
        text (str):
            The transcript text to save.
        save_path (str):
            Target path (including file name) for saving transcript.
    """
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Transcript saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save transcript at {save_path}: {e}")

# -----------------------------------------------------------------------------
# High-level function: Orchestrate full transcription flow
# -----------------------------------------------------------------------------
def handle_video_transcription(
    uploaded_file_path: str,
    save_folder: str = "data/transcripts",
    language: Optional[str] = None
) -> Tuple[str, str]:
    """
    Main handler to transcribe and save transcript for a video/audio file.

    Args:
        uploaded_file_path (str):
            Path to the uploaded media file.
        save_folder (str):
            Directory in which to store the transcript file.
        language (Optional[str]):
            Force transcription language (e.g., "en"), or None to auto-detect.

    Returns:
        tuple (saved_file_path, transcript_text):
            saved_file_path (str): Path to the saved transcript file.
            transcript_text (str): Transcript content.
    """
    os.makedirs(save_folder, exist_ok=True)

    # Prepare transcript save path
    base_name = os.path.splitext(os.path.basename(uploaded_file_path))[0]
    save_path = os.path.join(save_folder, f"{base_name}_transcript.txt")

    # Perform transcription
    transcript, proc_time = transcribe_audio(uploaded_file_path, language=language)

    # If transcription failed
    if not transcript:
        logger.warning(f"No transcript generated for: {uploaded_file_path}")
        return "", ""

    # Save transcript to disk
    save_transcript(transcript, save_path)
    logger.info(f"Transcript saved: {save_path} (processing took {proc_time:.2f} sec)")

    return save_path, transcript

# -----------------------------------------------------------------------------
# Debug / Local usage example (run this file directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    example_file = "example.mp4"  # Change to your own media file path
    path, text = handle_video_transcription(example_file)
    if text:
        print("\nTranscription Output:\n")
        print(text)
