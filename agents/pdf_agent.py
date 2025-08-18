import os
import logging
from typing import List, Optional, Union
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from utils.chunking_utils import ChunkingUtils
from utils.faiss_utils import save_faiss_index
from agents.llm_agent import call_llm  # For summarization

# -----------------------------------------------------------------------------
# Logging configuration (only if not already configured elsewhere)
# -----------------------------------------------------------------------------
logger = logging.getLogger("PDFAgent")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[PDFAgent] %(levelname)s: %(message)s'
    )

def extract_text_from_pdf(
    pdf_path: str,
    fallback_threshold: int = 100,
    password: Optional[str] = None
) -> str:
    """
    Extract text from a PDF using multiple methods:
    - PyPDF2 (standard extraction)
    - PyMuPDF blocks mode (structured blocks of text)
    - PyMuPDF raw mode (raw text)

    Combines all extracted texts to maximize coverage and avoids duplicates.
    Cleans up extra whitespace and logs warnings if extracted text is very short.

    Args:
        pdf_path (str): Path to the PDF file.
        fallback_threshold (int): Minimum acceptable text length; warns if less.
        password (Optional[str]): Password for encrypted PDFs.

    Returns:
        str: The merged and cleaned extracted text.
    """
    text_pypdf2 = ""
    text_fitz_blocks = ""
    text_fitz_raw = ""

    # --- PyPDF2 extraction ---
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            if password:
                reader.decrypt(password)
            else:
                logger.error("PDF is password-protected and no password was given.")
                return ""
        # Extract text from each page and join with newlines
        text_pypdf2 = "\n".join(page.extract_text() or "" for page in reader.pages)
        logger.info(f"PyPDF2 extracted {len(text_pypdf2)} characters")
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")

    # --- PyMuPDF extraction ---
    try:
        doc = fitz.open(pdf_path)

        # Blocks mode: collect text blocks from all pages
        block_parts = []
        for page in doc:
            for b in page.get_text("blocks"):
                if len(b) >= 5 and b[4]:
                    block_parts.append(b[4])
        text_fitz_blocks = "\n".join(block_parts)
        logger.info(f"PyMuPDF blocks extracted {len(text_fitz_blocks)} characters")

        # Raw mode: get raw text from each page
        raw_parts = [page.get_text("raw") for page in doc]
        text_fitz_raw = "\n".join(raw_parts)
        logger.info(f"PyMuPDF raw extracted {len(text_fitz_raw)} characters")

        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")

    # --- Merge results ---
    combined_texts = [t.strip() for t in (text_pypdf2, text_fitz_blocks, text_fitz_raw) if t and t.strip()]

    # Remove duplicates while preserving order
    seen = set()
    merged_lines = []
    for t in combined_texts:
        if t not in seen:
            merged_lines.append(t)
            seen.add(t)

    final_text = "\n".join(merged_lines)
    
    # --- Cleanup: normalize whitespace and reduce multiple blank lines ---
    final_text = final_text.strip()
    if final_text:
        lines = final_text.split("\n")
        cleaned_lines = [' '.join(line.split()) for line in lines]  # normalize spaces within each line
        final_text = "\n".join(cleaned_lines)
        # Replace 3+ consecutive newlines with 2 newlines for nicer formatting
        while "\n\n\n" in final_text:
            final_text = final_text.replace("\n\n\n", "\n\n")

    logger.info(f"Final merged text length: {len(final_text)} characters")

    # Warn if very short text which might indicate image-based or corrupted PDF
    if len(final_text) < fallback_threshold:
        logger.warning(
            f"Extracted text is very short ({len(final_text)} chars). "
            "PDF might be image-based or corrupted"
        )

    return final_text


def process_pdf_for_qa(
    pdf_path: str,
    index_save_path: str = "data/faiss_indexes/",
    return_chunks: bool = False,
    fallback_threshold: int = 100,
    password: Optional[str] = None,
    summarize: bool = False,
    prompt: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 800
) -> Union[List[str], str]:
    """
    Full processing pipeline for a PDF:
    1. Extract text from PDF
    2. Chunk the text for downstream processing
    3. Create and save embeddings and FAISS index for similarity search
    4. Optionally summarize the document by summarizing each chunk and combining

    Args:
        pdf_path (str): Path to PDF file to process.
        index_save_path (str): Directory to save FAISS index files.
        return_chunks (bool): If True, return text chunks instead of embeddings.
        fallback_threshold (int): Minimum text length warning threshold.
        password (Optional[str]): Password for password-protected PDFs.
        summarize (bool): If True, return a summary string instead of chunks.
        prompt (Optional[str]): Custom prompt text to use for summarization.
        temperature (float): Sampling temperature for LLM calls during summarization.
        max_tokens (int): Max tokens for LLM summarization calls.

    Returns:
        Union[List[str], str]: List of text chunks or a summary string depending on arguments.
    """
    logger.info(f"Processing: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # Extract raw text from PDF
    raw_text = extract_text_from_pdf(pdf_path, fallback_threshold, password)
    if not raw_text:
        logger.warning("No text extracted from PDF.")
        return [] if not summarize else "❌ No text extracted from PDF."

    # Chunk the extracted text using utility class
    chunker = ChunkingUtils()
    chunks = chunker.chunk_text(raw_text)
    logger.info(f"Chunked into {len(chunks)} segments.")

    # Prepare document identifier based on file and folder names
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    parent_dir = os.path.basename(os.path.dirname(pdf_path))
    doc_id = f"{parent_dir}_{basename}" if parent_dir else basename

    # Save the FAISS index for retrieval
    save_faiss_index(chunks, doc_id, index_dir=index_save_path)
    logger.info(f"FAISS index saved for: {doc_id}")

    # Return chunks immediately if requested without summary
    if return_chunks and not summarize:
        return chunks

    # If summarization requested, summarize each chunk and combine
    if summarize:
        logger.info("Summarizing all PDF chunks...")
        chunk_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_prompt = prompt or "Summarize the following text clearly and concisely:"
            full_prompt = f"{chunk_prompt}\n\n{chunk}"
            try:
                summary = call_llm(full_prompt, temperature=temperature, max_tokens=max_tokens)
                chunk_summaries.append(summary.strip())
                logger.info(f"Chunk {i}/{len(chunks)} summarized.")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")

        # Combine all chunk summaries into a single unified summary
        final_prompt = (
            "Combine the following chunk summaries into a single, coherent summary of the entire document:\n\n"
            + "\n\n".join(chunk_summaries)
        )
        try:
            final_summary = call_llm(final_prompt, temperature=temperature, max_tokens=max_tokens)
            logger.info("Final PDF summary generated.")
            return final_summary.strip()
        except Exception as e:
            logger.error(f"Final summary generation failed: {e}")
            # Return concatenated chunk summaries as fallback
            return "\n\n".join(chunk_summaries)

    # Default: return chunks or empty list
    return chunks if return_chunks else []
