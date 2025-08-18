import logging
import torch
import whisper

# -----------------------------------------------------------------------------
# Logger setup for Whisper utilities
# -----------------------------------------------------------------------------
logger = logging.getLogger("WhisperUtils")

# -----------------------------------------------------------------------------
# Global configuration for Whisper model size.
# Changing this variable affects all Whisper usage across the application.
# Typical allowed sizes: "tiny", "base", "small", "medium", "large"
# Larger models → higher accuracy but slower and more RAM/VRAM usage.
# -----------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "tiny"

# -----------------------------------------------------------------------------
# Global cache for the loaded Whisper model instance
# This ensures the model is loaded once per process and reused on every call
# -----------------------------------------------------------------------------
_model_cache = None

def get_whisper_model():
    """
    Lazily load and return a shared Whisper model instance.

    Automatically selects GPU if available, otherwise falls back to CPU.

    Returns:
        whisper.Whisper: Loaded Whisper model ready for transcription.
    """
    global _model_cache
    if _model_cache is None:
        # Choose device: CUDA GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {device}...")

        # Load Whisper model for the chosen device
        # Note:
        # - The `fp16` flag is omitted here for compatibility with some installations.
        # - You may conditionally enable it if your Whisper + hardware supports mixed precision.
        _model_cache = whisper.load_model(
            WHISPER_MODEL_SIZE,
            device=device
        )

        logger.info("Whisper model loaded successfully.")

    return _model_cache
