# utils/summary_agent.py
import logging
from typing import Optional
from agents.llm_agent import call_llm
from agents.pdf_agent import process_pdf_for_qa

# -----------------------------------------------------------------------------
# Logger setup for SummaryAgent
# -----------------------------------------------------------------------------
logger = logging.getLogger("SummaryAgent")

# -----------------------------------------------------------------------------
# Summarization from plain text
# -----------------------------------------------------------------------------
def generate_summary(
    text: str,
    prompt: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 800
) -> str:
    """
    Summarize a given text string into bullet points (default) or using a custom style.

    Args:
        text (str): The transcript or text to summarize.
        prompt (Optional[str]): An optional custom instruction prompt for summarization.
        temperature (float): LLM temperature controlling creativity (default: 0.5).
        max_tokens (int): Max tokens for LLM output (default: 800).

    Returns:
        str: The summary text or an error/fallback message.
    """
    # Guard clause for empty input
    if not text.strip():
        logger.warning("Empty transcript text provided for summarization.")
        return "Transcript is empty."
    
    # Use provided prompt or default to bullet-point summary
    base_prompt = prompt or "Summarize the following transcript into clear bullet points."
    full_prompt = f"{base_prompt}\n\n{text}"
    
    logger.info("Generating summary using LLM.")
    try:
        summary = call_llm(full_prompt, temperature=temperature, max_tokens=max_tokens)
        return summary.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Summary generation failed: {str(e)}"

# -----------------------------------------------------------------------------
# Summarization directly from PDF
# -----------------------------------------------------------------------------
def generate_summary_from_pdf(
    pdf_path: str,
    prompt: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 800
) -> str:
    """
    Extract text from a PDF, summarize each chunk, 
    and return a combined summary of the entire document.

    Args:
        pdf_path (str): Path to the PDF to summarize.
        prompt (Optional[str]): Custom summarization instruction prompt.
        temperature (float): LLM temperature controlling creativity (default: 0.5).
        max_tokens (int): Max tokens for LLM output per chunk/final summary (default: 800).

    Returns:
        str: The combined summary of the PDF or an error message.
    """
    logger.info(f"Starting full-PDF summarization for: {pdf_path}")
    try:
        # Pass parameters through to process_pdf_for_qa in summarization mode
        summary = process_pdf_for_qa(
            pdf_path,
            summarize=True,
            fallback_threshold=100,
            password=None,
            index_save_path="data/faiss_indexes/",
            return_chunks=False,
            # These params need to be supported by process_pdf_for_qa in pdf_agent.py
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return summary.strip() if summary else "❌ No summary generated."
    except Exception as e:
        logger.error(f"PDF summarization failed: {e}")
        return f"PDF summarization failed: {str(e)}"
