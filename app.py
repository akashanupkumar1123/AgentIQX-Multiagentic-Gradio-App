import os
import re
import time
import gradio as gr
from dotenv import load_dotenv

# == Local utils ==
from utils.transcript_utils import transcribe_audio_or_video, extract_text_from_pdf
from utils.summary_agent import generate_summary
from utils.tts_utils import text_to_speech
from utils.emails_utils import send_email

# -----------------------------------------------------------------------------
# Load environment variables from .env file (for email credentials)
# -----------------------------------------------------------------------------
# On Render, env vars are already injected automatically.
if os.getenv("RENDER") is None:  # detect local run
    from dotenv import load_dotenv
    load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# -----------------------------------------------------------------------------
# Email validation helper
# -----------------------------------------------------------------------------
def is_valid_email(email: str) -> bool:
    """Return True if provided string looks like a valid email address."""
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email or ""))

# -----------------------------------------------------------------------------
# Build custom summarization prompt based on user Q&A selections
# -----------------------------------------------------------------------------
def build_custom_prompt(style, detail, focus):
    """
    Construct a natural language prompt for summary generation.
    Args:
        style (str): Desired summary style (e.g., paragraph, bulleted list).
        detail (str): Level of detail (e.g., short, medium, detailed).
        focus (str): Specific focus areas or keywords.
    Returns:
        str: Combined instruction string for LLM.
    """
    parts = []
    if style:
        parts.append(f"Generate the summary in {style.lower()}.")
    if detail:
        parts.append(f"Provide {detail.lower()} detail.")
    if focus:
        parts.append(f"Focus especially on: {focus}.")
    return " ".join(parts)

# -----------------------------------------------------------------------------
# Pipeline: Transcription (PDF or audio/video) + Summarization
# -----------------------------------------------------------------------------
def transcribe_and_summarize(uploaded_file, style, detail, focus, progress=gr.Progress()):
    progress(0.1)
    if uploaded_file is None:
        return "❌ Please upload a file.", "", gr.update(visible=False)

    # Get actual path from uploaded Gradio file object or string
    file_path = uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    file_ext = os.path.splitext(file_path)[1].lower()

    # ✅ Ensure transcript is always a string
    if file_ext == ".pdf":
        transcript = extract_text_from_pdf(file_path)
    else:
        transcript, _ = transcribe_audio_or_video(file_path)  # returns (text, detected_lang)

    progress(0.5)

    if not transcript or transcript.startswith(("Error", "❌")):
        return transcript, "", gr.update(visible=False)

    # Build summarization prompt
    custom_prompt = build_custom_prompt(style, detail, focus)
    summary = generate_summary(transcript, custom_prompt)

    progress(1.0)
    return transcript, summary, gr.update(visible=True)

# -----------------------------------------------------------------------------
# Real-time email input validation
# -----------------------------------------------------------------------------
def validate_email_live(email, email_err, send_btn):
    """Update error message and button enable state based on email validity."""
    if not email:
        return (
            email_err.update(visible=False),
            send_btn.update(disabled=True)
        )
    if is_valid_email(email):
        return (
            email_err.update(visible=False),
            send_btn.update(disabled=False)
        )
    return (
        email_err.update(visible=True, value="❌ Invalid email format"),
        send_btn.update(disabled=True)
    )

# -----------------------------------------------------------------------------
# Send summary by email and/or generate TTS audio
# -----------------------------------------------------------------------------
def send_email_and_tts(email_to, text, speak_flag, progress=gr.Progress()):
    """
    Send summary via email & optionally generate TTS audio.

    Args:
        email_to (str): Recipient email address.
        text (str): Summary text to send/speak.
        speak_flag (bool): Whether to generate audio from the summary.
    Returns:
        tuple: (TTS audio path or None, status message string)
    """
    import time as _time
    tts_path, email_status = None, ""
    time.sleep(0.2)
    progress(0.3)

    # Validate email before doing anything else
    if email_to and not is_valid_email(email_to):
        return {"value": None}, "❌ Invalid email format — please check and try again."

    # Generate TTS if requested
    if speak_flag and text:
        try:
            tts_path = text_to_speech(text)
        except Exception as e:
            email_status = f"⚠️ TTS generation failed: {str(e)}"

    progress(0.7)

    # Send email if requested and content is present
    if email_to and text:
        try:
            send_status = send_email(
                email_to,
                "Your Generated Summary",
                text,
                sender_email=SENDER_EMAIL,
                sender_password=SENDER_PASSWORD
            )
            # Treat truthy return without "error" in text as success
            if send_status and "error" not in str(send_status).lower():
                timestamp = _time.strftime("%H:%M:%S on %b %d, %Y")
                email_status = f"✅ Email sent successfully to {email_to} at {timestamp}"
            else:
                email_status = f"❌ Failed to send email to {email_to} — please try again."
        except Exception as e:
            email_status = f"❌ Could not send email — {str(e)}"
    elif not email_to:
        email_status = "ℹ️ Email sending skipped — no recipient provided."
    else:
        email_status = "⚠️ No content to email."

    progress(1.0)
    return tts_path or None, email_status

# -----------------------------------------------------------------------------
# Load custom CSS
# -----------------------------------------------------------------------------
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    custom_css = f.read()

# -----------------------------------------------------------------------------
# Google Fonts
# -----------------------------------------------------------------------------
extra_head_html = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
"""

# -----------------------------------------------------------------------------
# UI Layout using Gradio Blocks
# -----------------------------------------------------------------------------
with gr.Blocks(css=custom_css, head=extra_head_html,
               title="AgentIQX Transcript Summarizer", theme=None) as app:

    # Track active tab
    active_tab = gr.State("tab1")

    # Header
    gr.Markdown("""
    <h1 style='text-align:center;color:#23e3ae'>AGENTIQX<br>
    <small style='color:#00ffe7'>MultiAgentic Transcript Summarizer</small></h1>
    <p style='text-align:center;color:#7ee2f8'>
    Extract, Summarize, Listen, and Email — all in one place.
    </p>
    """)

    with gr.Tabs(selected=active_tab) as tabs:
        # ---------------------------------------------------------------------
        # Tab 1: Transcription and Summarization
        # ---------------------------------------------------------------------
        with gr.Tab("📝 Transcribe & Summarize", id="tab1"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="📎 Upload PDF or Audio/Video",
                        file_types=[".pdf", ".mp3", ".wav", ".mp4", ".m4a", ".mov"]
                    )
                    # Q&A style summarization option inputs
                    style_choice = gr.Dropdown(
                        label="📄 Summary Style",
                        choices=["Bulleted list", "Numbered list", "Paragraph"],
                        value="Paragraph"
                    )
                    detail_choice = gr.Radio(
                        label="📊 Level of Detail",
                        choices=["Short", "Medium", "Detailed"],
                        value="Medium"
                    )
                    focus_text = gr.Textbox(
                        label="🎯 Focus on",
                        placeholder="e.g., keywords, people, events, action items"
                    )
                    process_btn = gr.Button("🚀 Process", variant="primary")

                with gr.Column():
                    transcript_output = gr.Textbox(
                        label="📝 Transcript",
                        lines=11,
                        interactive=False,
                        show_copy_button=True
                    )
                    summary_output = gr.Textbox(
                        label="✨ Summary",
                        lines=8,
                        interactive=False,
                        show_copy_button=True
                    )

            # Link process button to transcription + summary pipeline
            process_btn.click(
                transcribe_and_summarize,
                inputs=[file_input, style_choice, detail_choice, focus_text],
                outputs=[transcript_output, summary_output]
            )

        # ---------------------------------------------------------------------
        # Tab 2: Text-to-Speech and Email
        # ---------------------------------------------------------------------
        with gr.Tab("🔈 TTS & Email", id="tab2"):
            with gr.Column():
                email_input = gr.Textbox(label="📧 Email to send summary (optional)")
                email_err = gr.Textbox(value="", visible=False, elem_classes=["email-error-message"])
                speak_checkbox = gr.Checkbox(label="🗣️ Speak summary after generation", value=False)

            with gr.Row():
                send_btn = gr.Button("✉️ Send & Speak", variant="primary")
                tts_audio = gr.Audio(label="📣 Summary Audio", interactive=False)
                email_status_output = gr.Textbox(label="📨 Email Status", lines=2, interactive=False)

            # Live email validation
            email_input.change(
                validate_email_live,
                inputs=[email_input, email_err, send_btn],
                outputs=[email_err, send_btn]
            )

            # Link send button to email+TTS handler
            send_btn.click(
                send_email_and_tts,
                inputs=[email_input, summary_output, speak_checkbox],
                outputs=[tts_audio, email_status_output],
                queue=True
            )

# -----------------------------------------------------------------------------
# Launch the app
# -----------------------------------------------------------------------------

app.launch()
