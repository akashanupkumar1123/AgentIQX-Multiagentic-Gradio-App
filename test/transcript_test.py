import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transcript_utils import extract_text_from_pdf

text = extract_text_from_pdf("sample.pdf")
print(f"Chars: {len(text)}")
print("----\n" + text + "\n----")
