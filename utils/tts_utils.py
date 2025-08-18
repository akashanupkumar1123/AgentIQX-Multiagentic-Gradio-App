# utils/tts_utils.py
import os
import logging
from typing import Optional
from gtts import gTTS  # Google Text-to-Speech

# -----------------------------------------------------------------------------
# Logger setup for Text-to-Speech utility
# -----------------------------------------------------------------------------
logger = logging.getLogger("TTSUtils")

# -----------------------------------------------------------------------------
# Function: Convert text to speech
# -----------------------------------------------------------------------------
def text_to_speech(
    text: str,
    output_path: str = "output_audio.mp3",
    lang: str = "en",
    slow: bool = False
) -> str:
    """
    Convert text to speech using gTTS and save as an MP3 file.

    Args:
        text (str): Input text to convert into speech.
        output_path (str): Destination file path to save the audio.
                           Should end with `.mp3`. Defaults to "output_audio.mp3".
        lang (str): Language code for TTS (default "en").
        slow (bool): Whether to speak slowly (default False).

    Returns:
        str: Path to the saved MP3 file.

    Raises:
        ValueError: If `text` is empty or only whitespace.
        Exception: Any errors that occur during conversion or file saving.
    """
    # Ensure non-empty input text
    if not text.strip():
        raise ValueError("Input text for TTS is empty.")

    try:
        # Ensure the output directory exists (or current dir if only filename given)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Create gTTS object with given language and speed
        tts = gTTS(text=text, lang=lang, slow=slow)

        # Save the generated speech to the specified MP3 file
        tts.save(output_path)
        logger.info(f"TTS audio saved to: {output_path}")

        return output_path

    except Exception as e:
        # Log any errors during the TTS pipeline
        logger.error(f"Error during TTS conversion or saving: {e}")
        raise

# -----------------------------------------------------------------------------
# Example usage (commented out)
# -----------------------------------------------------------------------------
# try:
#     audio_path = text_to_speech("Hello world!", "output/hello.mp3")
#     print(f"Saved audio to {audio_path}")
# except Exception as e:
#     print(f"TTS failed: {e}")
