# app.py - LegalEase AI: All Modules in One
import streamlit as st
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import pytesseract
import numpy as np
import re

# -----------------------------
# Services (Embedded for HF)
# -----------------------------

def extract_text_from_pdf(file_bytes):
    try:
        pdf_stream = BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""

def extract_text_with_ocr(file_bytes):
    try:
        pdf_stream = BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        return f"OCR failed: {str(e)}"

def extract_text_from_file(file_bytes, file_type):
    try:
        if file_type == "pdf":
            text = extract_text_from_pdf(file_bytes)
            if len(text.strip()) < 50:
                text = extract_text_with_ocr(file_bytes)
        elif file_type == "docx":
            doc_stream = BytesIO(file_bytes)
            doc = Document(doc_stream)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])
        else:
            text = ""
        return text
    except Exception as e:
        return f"Text extraction failed: {e}"

def summarize_text(text: str, max_length=300, min_length=100):
    if len(text.split()) < 50:
        return "Text too short to summarize."
    try:
        # Simulate BART (replace with real summarizer when deploying)
        sentences = text.split('. ')
        summary = '. '.join(sentences[:4]) + '...'
        return summary if len(summary) > 20 else "Summary could not be generated."
    except:
        return "Summary generation failed."

def analyze_contract_text(text: str):
    findings = []
    lines = text.split('\n')
    patterns = [
        (r"liable for all damages|no limit on liability", "Unlimited Liability", "high",
         "Cap liability to the amount paid under the contract."),
        (r"auto-renews|automatically renews", "Automatic Renewal", "medium",
         "Add 30-day notice to opt-out before renewal."),
        (r"may terminate at any time|without cause", "One-Sided Termination", "medium",
         "Ensure both parties have equal termination rights."),
        (r"governed by New York law|jurisdiction in London", "Foreign Jurisdiction", "high",
         "Use Ethiopian law and Addis Ababa courts for enforceability."),
    ]
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for pattern, issue, risk, suggestion in patterns:
            if st.session_state.get("debug"): print(pattern, line_lower)
            if re.search(pattern, line_lower):
                findings.append({
                    "line": i + 1,
                    "text": line.strip(),
                    "issue": issue,
                    "risk": risk,
                    "suggestion": suggestion
                })
    return findings

# Clean text for PDF output: remove or replace unsupported characters
def clean_text_for_pdf(text: str) -> str:
    # Replace em-dash and en-dash with hyphen
    text = text.replace('—', '-').replace('–', '-')
    # Remove emojis and non-ASCII chars (keep printable ASCII)
    return ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)

def generate_report_pdf(summary: str, findings: list):
    from fpdf import FPDF
    from datetime import datetime

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, clean_text_for_pdf('LegalEase AI — Contract Analysis Report'), ln=True, align='C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, clean_text_for_pdf(f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | Not Legal Advice'), align='C')

        def add_section(self, title, content):
            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, clean_text_for_pdf(title), ln=True)
            self.set_font('Helvetica', '', 12)
            self.multi_cell(0, 6, clean_text_for_pdf(content))
            self.ln(8)

        def add_risk_table(self, findings):
            if not findings:
                self.set_font('Helvetica', 'I', 12)
                self.cell(0, 10, 'No risks detected.', ln=True)
                self.ln(8)
                return

            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, 'Detected Risks', ln=True)
            self.set_font('Helvetica', 'B', 10)
            self.set_fill_color(220, 220, 220)
            self.cell(80, 8, 'Issue', 1, 0, 'C', 1)
            self.cell(30, 8, 'Risk', 1, 0, 'C', 1)
            self.cell(80, 8, 'Suggestion', 1, 1, 'C', 1)

            self.set_font('Helvetica', '', 9)
            for f in findings:
                c = (255, 180, 180) if f["risk"] == "high" else (255, 240, 180)
                self.set_fill_color(*c)
                self.cell(80, 6, clean_text_for_pdf(f["issue"]), 1, 0, 'L', 1)
                self.cell(30, 6, f["risk"].upper(), 1, 0, 'C', 1)
                self.cell(80, 6, clean_text_for_pdf(f["suggestion"][:60] + "..."), 1, 1, 'L', 1)
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.add_section("Document Summary", summary)
    pdf.add_risk_table(findings)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 6, clean_text_for_pdf("⚠️ DISCLAIMER: This report is AI-generated and not legal advice. Consult a licensed attorney."))

    # ✅ Return bytes instead of bytearray
    return bytes(pdf.output(dest='S'))
# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="LegalEase AI", layout="centered")
st.title("💼 LegalEase AI — Your Legal Assistant")
st.markdown("Powered by AI • For Ethiopian Law • Not Legal Advice")
st.caption("⚠️ This tool does not provide legal advice. Always consult a licensed attorney.")

# Sidebar - Debug
with st.sidebar:
    st.checkbox("🔧 Debug Mode", key="debug")

# Tabs
tab1, tab2, tab3 = st.tabs(["🗨️ Ask a Question", "📄 Summarize Document", "🔍 Analyze Risks"])

# -----------------------------
# TAB 1: Chat (Simulated)
# -----------------------------
with tab1:
    st.header("Chat with LegalEase AI")
    st.markdown("Ask about Ethiopian civil law, contracts, rights, and more.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("🧠 Thinking..."):
            response = (
                f"Based on Ethiopian Civil Code, {prompt} typically falls under general contract principles. "
                "For example, Article 1718 requires consent, capacity, object, and cause for a valid contract. "
                "For precise advice, consult a licensed lawyer."
            )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

# -----------------------------
# TAB 2: Summarize
# -----------------------------
with tab2:
    st.header("Summarize Legal Document")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:
        with st.spinner("🔍 Extracting text..."):
            file_bytes = uploaded_file.read()
            file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"
            text = extract_text_from_file(file_bytes, file_type)

        if len(text.strip()) < 50:
            st.warning("⚠️ Could not extract enough text. Is the file scanned or encrypted?")
        else:
            st.success("✅ Text extracted!")
            with st.expander("📄 Extracted Text (first 500 chars)"):
                st.write(text[:500] + "...")

            with st.spinner("🧠 Generating summary..."):
                summary = summarize_text(text)

            st.markdown("### 📝 Summary")
            st.write(summary)

# -----------------------------
# TAB 3: Risk Analyzer
# -----------------------------
with tab3:
    st.header("🔍 Smart Contract Risk Analysis")
    uploaded_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"], key="risk")

    if uploaded_file:
        with st.spinner("🔍 Scanning for risks..."):
            file_bytes = uploaded_file.read()
            file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"
            text = extract_text_from_file(file_bytes, file_type)

        if len(text.strip()) < 50:
            st.warning("⚠️ Not enough text extracted.")
        else:
            findings = analyze_contract_text(text)
            summary = summarize_text(text)

            if findings:
                st.markdown("### 🚩 Detected Risks")
                high = len([f for f in findings if f["risk"] == "high"])
                med = len([f for f in findings if f["risk"] == "medium"])
                st.metric("Total Issues", len(findings))
                st.markdown(f"- 🔴 High Risk: {high}")
                st.markdown(f"- 🟡 Medium Risk: {med}")

                for i, f in enumerate(findings):
                    with st.expander(f"📌 {f['issue']} (Line {f['line']})"):
                        st.write(f"**Clause**: {f['text']}")
                        st.write(f"**Risk Level**: {'🔴 High' if f['risk']=='high' else '🟡 Medium'}")
                        st.info(f"💡 **Suggestion**: {f['suggestion']}")
            else:
                st.success("✅ No major risks detected!")

            # PDF Report
            st.markdown("### 📄 Download Report")
            try:
                pdf_bytes = generate_report_pdf(summary, findings)
                st.download_button(
                    label="📥 Download Full Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"legalease_analysis_{uploaded_file.name.split('.')[0]}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ for Ethiopian legal empowerment • Capstone 2025")
