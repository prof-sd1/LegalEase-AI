# app.py

import streamlit as st
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import pytesseract
import numpy as np
import re
import json
import time
from datetime import datetime
from typing import List, Dict
import logging

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Clause Type Keywords
# -----------------------------
CLAUSE_TYPES = {
    "Termination": ["terminate", "termination", "end"],
    "Payment": ["payment", "fee", "charge"],
    "Confidentiality": ["confidential", "disclose", "nda"],
    "Jurisdiction": ["jurisdiction", "governing law", "dispute"],
    "Liability": ["liability", "liable", "damages"],
    "General": []
}

# -----------------------------
# Utility Functions
# -----------------------------

def extract_text_from_pdf(file_bytes):
    try:
        text = ""
        pdf_stream = BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        logger.exception("PDF extraction failed.")
        return ""

def extract_text_from_docx(file_bytes):
    try:
        docx_stream = BytesIO(file_bytes)
        doc = Document(docx_stream)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.exception("DOCX extraction failed.")
        return ""

def extract_text_from_file(file_bytes, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(file_bytes)
        if not text:
            images = extract_images_from_pdf(file_bytes)
            return extract_text_with_ocr(images)
        return text
    elif file_type == "docx":
        return extract_text_from_docx(file_bytes)
    return ""

def extract_images_from_pdf(file_bytes):
    images = []
    try:
        doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
    except Exception as e:
        logger.exception("Image extraction from PDF failed.")
    return images

def extract_text_with_ocr(images):
    text = ""
    for img in images:
        try:
            img = img.convert("L")
            text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            logger.exception("OCR failed.")
    return text.strip()

def analyze_contract_text(text: str) -> Dict[str, List[str]]:
    risks = {
        "Unclear Terms": [],
        "Missing Dates": [],
        "No Party Info": [],
        "Incomplete Clauses": []
    }
    if not re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", text):
        risks["Missing Dates"].append("No clear date mentioned in contract.")
    if not re.search(r"\b(?:between|among|by)\b", text.lower()):
        risks["No Party Info"].append("Missing party definitions.")
    if "..." in text:
        risks["Incomplete Clauses"].append("Ellipsis found indicating missing content.")
    unclear_terms = re.findall(r"\b(?:etc\.|and so on|miscellaneous)\b", text.lower())
    if unclear_terms:
        risks["Unclear Terms"].extend(unclear_terms)
    return {k: v for k, v in risks.items() if v}

def summarize_contract(text: str, sentence_count: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = sorted(sentences, key=len, reverse=True)
    summary = " ".join(sentences[:sentence_count])
    return summary.strip()

def split_into_clauses(text: str) -> List[str]:
    return re.split(r"\n+|(?<=\.)\s(?=[A-Z])", text)

def detect_clause_type(text: str) -> str:
    for label, keywords in CLAUSE_TYPES.items():
        if any(kw in text.lower() for kw in keywords):
            return label
    return "General"

# -----------------------------
# Streamlit App UI
# -----------------------------

st.set_page_config(layout="wide", page_title="📜 LegalEase AI")
st.title("📜 LegalEase AI")
st.caption("Analyze, summarize, and understand Ethiopian legal contracts.")

uploaded_file = st.file_uploader("Upload Contract (PDF or DOCX)", type=["pdf", "docx"])

def process_uploaded_file(uploaded_file):
    file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"
    file_bytes = uploaded_file.read()
    text = extract_text_from_file(file_bytes, file_type)
    return file_bytes, text, file_type

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Risk Analysis", "📝 Summarization", "📄 Clause View", "📤 Export"])

if uploaded_file:
    if "file_bytes" not in st.session_state:
        st.session_state["file_bytes"], st.session_state["file_text"], st.session_state["file_type"] = process_uploaded_file(uploaded_file)

    file_bytes = st.session_state["file_bytes"]
    text = st.session_state["file_text"]
    file_type = st.session_state["file_type"]

    # ---------------- Risk Analysis ----------------
    with tab1:
        st.subheader("🧠 Contract Risk Analysis")
        findings = analyze_contract_text(text)
        if findings:
            for category, issues in findings.items():
                st.markdown(f"### {category}")
                for item in issues:
                    st.warning(item)
        else:
            st.success("No major risks or missing parts found.")

        st.download_button(
            label="📤 Download Risk Report (JSON)",
            data=json.dumps(findings, indent=2),
            file_name="contract_risks.json",
            mime="application/json"
        )

    # ---------------- Summarization ----------------
    with tab2:
        st.subheader("📝 Contract Summary")
        summary = summarize_contract(text)
        st.info(summary)

    # ---------------- Clause View ----------------
    with tab3:
        st.subheader("📄 Clause Breakdown")
        clauses = split_into_clauses(text)
        for i, clause in enumerate(clauses):
            if len(clause.strip()) > 20:
                clause_type = detect_clause_type(clause)
                with st.expander(f"📌 Clause {i+1} [{clause_type}]"):
                    st.write(clause.strip())

    # ---------------- Export Text ----------------
    with tab4:
        st.subheader("📤 Export Full Text")
        st.download_button(
            label="Download Full Contract Text (.txt)",
            data=text,
            file_name="contract_text.txt",
            mime="text/plain"
        )
