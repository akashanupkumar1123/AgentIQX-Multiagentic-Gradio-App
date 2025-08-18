import os
import logging
import time
import fitz  # PyMuPDF
from typing import Optional, Tuple
from utils.whisper_utils import get_whisper_model  # Use shared cached Whisper model

# -----------------------------------------------------------------------------
# Logger setup for this utility module
# -----------------------------------------------------------------------------
logger = logging.getLogger("TranscriptUtils")

# -----------------------------------------------------------------------------
# Default Whisper model size - can be overridden globally if needed
# -----------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "tiny"

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted raw text from the PDF, concatenated across all pages.
    
    Raises:
        FileNotFoundError: if the PDF file does not exist.
        RuntimeError: on any error reading the PDF.
    """
    if not os.path.isfile(pdf_path):
        msg = f"PDF file not found: {pdf_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    try:
        # Open the PDF document safely using context manager
        with fitz.open(pdf_path) as doc:
            # Extract text from each page and join with newlines
            text = "\n".join(page.get_text() for page in doc)
        logger.info(f"Extracted text from PDF: {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise RuntimeError(f"Error reading PDF: {e}")

def transcribe_audio_or_video(file_path: str, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Transcribe an audio or video file using the Whisper model.
    
    Args:
        file_path (str): Path to the media file to transcribe.
        language (Optional[str]): Force transcription language code (e.g., 'en').
                                  If None, language will be auto-detected.
    
    Returns:
        Tuple[str, str]: Tuple of (Transcript text, detected language code).
                         Returns empty strings if transcription fails.
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return "", ""
    
    # Retrieve the shared Whisper model instance
    model = get_whisper_model()
    
    try:
        options = {}
        if language:
            options["language"] = language
        
        # Perform transcription using Whisper API
        result = model.transcribe(file_path, **options)
        
        # Normalize whitespace in output transcript
        full_text = " ".join(result["text"].split())
        
        detected_language = result.get("language", "unknown")
        logger.info(f"Transcription completed for: {file_path}")
        return full_text, detected_language
    
    except Exception as e:
        logger.error(f"Error transcribing {file_path}: {e}")
        return "", ""

def save_text_to_file(text: str, output_path: str) -> None:
    """
    Save text content to a file. Creates parent directories if they don't exist.
    
    Args:
        text (str): Text content to save.
        output_path (str): Full path (including filename) where text should be saved.
    
    Raises:
        IOError: If writing file fails.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved text to file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save file {output_path}: {e}")
        raise IOError(f"Failed to save file: {e}")
