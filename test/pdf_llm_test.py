import sys
import os

# 🔹 Add project root to sys.path so imports work no matter where the script is
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transcript_utils import extract_text_from_pdf
from agents.llm_agent import call_llm

# Locate the PDF in same folder as this script
pdf_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Step 1: Extract full text from PDF
text = extract_text_from_pdf(pdf_path)
print(f"Extracted chars: {len(text)}")
print("---- Extracted Text ----\n" + text + "\n------------------------")

# Step 2: Send to LLM for summarization
print("\n Calling LLM to summarize the PDF...")
summary = call_llm(f"Summarize the following document:\n\n{text}")
print("\n---- Summary ----\n" + summary + "\n-----------------")
