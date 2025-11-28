# app/utils.py
from pypdf import PdfReader
import docx2txt
import os

def extract_text_from_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            text += p.extract_text() or ""
    except Exception as e:
        print("PDF parse error:", e)
    return text

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path) or ""
    except Exception as e:
        print("DOCX parse error:", e)
        return ""

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
