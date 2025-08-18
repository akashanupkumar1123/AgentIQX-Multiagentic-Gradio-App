import os
import sys

# Adjust sys.path so imports work regardless of Colab/local folder structure
sys.path.append("")
sys.path.append("utils")

from utils.transcript_utils import extract_text_from_pdf, transcribe_audio_or_video
from utils.summary_agent import generate_summary
from utils.tts_utils import text_to_speech
from utils.emails_utils import send_email

def test_full_pipeline(pdf_path, media_path, test_email):
    # 1. PDF Extraction
    print(f"Testing PDF text extraction from: {pdf_path}")
    pdf_text = ""
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        print(f"Extracted PDF text (first 300 chars):\n{pdf_text[:300]}\n")
    except Exception as e:
        print(f"PDF extraction failed: {e}")

    # 2. Audio/Video Transcription
    print(f"Testing media transcription from: {media_path}")
    transcript = ""
    try:
        transcript = transcribe_audio_or_video(media_path)
        print(f"Transcription result (first 300 chars):\n{transcript[:300]}\n")
    except Exception as e:
        print(f"Media transcription failed: {e}")

    # 3. Summarization
    print("Testing summarization generation...")
    summary = ""
    try:
        text_to_summarize = transcript if transcript.strip() else pdf_text
        if not text_to_summarize or not text_to_summarize.strip():
            print("No text available to summarize. Skipping summarization.")
        else:
            summary = generate_summary(text_to_summarize)
            print(f"Generated summary (first 300 chars):\n{summary[:300]}\n")
    except Exception as e:
        print(f"Summarization failed: {e}")

    # 4. TTS Generation
    print("Testing TTS generation...")
    tts_path = None
    try:
        if summary and summary.strip():
            tts_path = text_to_speech(summary, output_path="test_summary.mp3")
            print(f"TTS audio generated at: {tts_path}\n")
        else:
            print("No summary to synthesize. Skipping TTS.")
    except Exception as e:
        print(f"TTS generation failed: {e}")

    # 5. Email Sending
    print("Testing email sending...")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    if not sender_email or not sender_password:
        print("Sender email credentials not configured. Skipping email sending test.")
    elif not summary or not summary.strip():
        print("No summary available to send. Skipping email sending.")
    else:
        try:
            email_status = send_email(
                recipient=test_email,
                subject="Test Summary Email",
                body=summary,
                sender_email=sender_email,
                sender_password=sender_password
            )
            print(f"Email send status: {email_status}\n")
        except Exception as e:
            print(f"Email send failed: {e}")

if __name__ == "__main__":
    # Customize these paths and email as needed
    TEST_PDF = "sample.pdf"
    TEST_MEDIA = "sample_video.mp4"  # can be audio or video file
    TEST_EMAIL = "aamakkeda@gmail.com"   # Use a real email you own for testing

    # Verify sample files exist
    if not os.path.exists(TEST_PDF):
        print(f"ERROR: PDF file not found: {TEST_PDF}")
    if not os.path.exists(TEST_MEDIA):
        print(f"ERROR: Media file not found: {TEST_MEDIA}")

    print("Starting full pipeline test...")
    test_full_pipeline(TEST_PDF, TEST_MEDIA, TEST_EMAIL)
