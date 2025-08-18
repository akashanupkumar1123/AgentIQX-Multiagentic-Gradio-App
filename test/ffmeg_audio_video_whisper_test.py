import sys
import os
import shutil

# ----------------------------
# Ensure FFmpeg is discoverable
# ----------------------------
os.environ["PATH"] += os.pathsep + r"C:\Users\asus\AppData\Local\Microsoft\WinGet\Links"
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"[OK] Python can see FFmpeg at: {ffmpeg_path}")
else:
    print("[ERROR] FFmpeg not found in Python PATH!")
    sys.exit(1)

# ----------------------------
# Ensure project root is in sys.path
# ----------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transcript_utils import transcribe_audio_or_video

# ----------------------------
# Define test file absolute paths
# ----------------------------
mp3_path = r"C:\Users\asus\Desktop\AgentIQX\test\sample_audio.mp3"
mp4_path = r"C:\Users\asus\Desktop\AgentIQX\test\sample_video.mp4"

# Debug: check file existence
print(f"[DEBUG] MP3 exists: {os.path.exists(mp3_path)} at {mp3_path}")
print(f"[DEBUG] MP4 exists: {os.path.exists(mp4_path)} at {mp4_path}")

def run_transcription(file_path):
    """Run Whisper transcription on a single file and print results."""
    print(f"\n[INFO] Testing transcription on: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")

    try:
        transcript, language = transcribe_audio_or_video(file_path)
        print("[OK] Transcription successful")
        print(f"Detected language: {language}")
        print(f"Transcript length: {len(transcript)} characters")
        print("---- TRANSCRIPT START ----")
        print(transcript)
        print("---- TRANSCRIPT END ----\n")
        return transcript.strip()
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None

# ----------------------------
# Run on MP3 and MP4
# ----------------------------
mp3_transcript = run_transcription(mp3_path)
mp4_transcript = run_transcription(mp4_path)

# ----------------------------
# Compare results if both present
# ----------------------------
if mp3_transcript and mp4_transcript:
    print("[INFO] Comparing transcripts...")
    if mp3_transcript == mp4_transcript:
        print("[OK] MP3 and MP4 transcripts match exactly!")
    else:
        print("[WARN] MP3 and MP4 transcripts differ.")
        # Show first difference
        mp3_words = mp3_transcript.split()
        mp4_words = mp4_transcript.split()
        mismatch_index = next(
            (i for i in range(min(len(mp3_words), len(mp4_words)))
             if mp3_words[i] != mp4_words[i]),
            None
        )
        if mismatch_index is not None:
            print(f"First difference at word {mismatch_index}:")
            print(f"MP3: {' '.join(mp3_words[mismatch_index:mismatch_index+8])}")
            print(f"MP4: {' '.join(mp4_words[mismatch_index:mismatch_index+8])}")
