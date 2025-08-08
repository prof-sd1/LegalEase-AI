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

# -----------------------------
# Services
# -----------------------------

def extract_text_from_pdf(file_bytes):
    try:
        with BytesIO(file_bytes) as pdf_stream:
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
                return text
    except fitz.FileDataError as e:
        return f"Corrupt PDF or invalid data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_text_with_ocr(file_bytes, lang='eng'):
    try:
        pdf_stream = BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img, lang=lang)
            text.append(page_text.strip())
        return "\n".join(text)
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract OCR not installed. Check system PATH."
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
        elif file_type == "txt":
            text = file_bytes.decode("utf-8", errors="replace")  # Handles encoding issues
        else:
            text = ""
        return text
    except Exception as e:
        return f"Text extraction failed: {e}"
        
def summarize_text(
    text: str,
    max_sentences: int = 4,
    min_length: int = 100,
    max_length: int = 300,
) -> str:
    """
    Summarizes text by extracting key sentences.
    Args:
        text: Input text to summarize.
        max_sentences: Maximum number of sentences in the summary.
        min_length: Minimum summary length (chars). Returns fallback if shorter.
        max_length: Truncates summary if longer than this.
    Returns:
        Summary or error message.
    """
    # Check if text is too short
    if len(text.split()) < 50:
        return "Text too short to summarize."

    try:
        # Improved sentence splitting (handles abbreviations, multiple punctuations)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Select key sentences (here: first 'max_sentences')
        summary = " ".join(sentences[:max_sentences])

        # Truncate if exceeding max_length
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

        # Validate minimum length
        return summary if len(summary) >= min_length else "Summary too short."
    except Exception as e:
        return f"Summary failed: {str(e)}"

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ContractFinding:
    line: int
    matched_text: str
    full_context: str
    issue: str
    risk: str
    suggestion: str
    pattern_used: str  # Track which pattern was matched

def analyze_contract_text(
    text: str,
    context_lines: int = 1,
    custom_patterns: Optional[List[Dict]] = None
) -> List[ContractFinding]:
    """
    Enhanced contract analysis with:
    - Better duplicate detection
    - Configurable patterns
    - More accurate context capture
    - Structured return type
    - Pattern tracking

    Args:
        text: Contract text to analyze
        context_lines: Number of lines before/after to include as context
        custom_patterns: Optional list of additional patterns to check
    
    Returns:
        List of ContractFinding objects with detailed findings
    """
    # Default patterns (can be extended with custom_patterns)
    base_patterns = [
        {
            "regex": r"\bliable for all damages\b|\bno limit on liability\b",
            "issue": "Unlimited Liability",
            "risk": "high",
            "suggestion": "Cap liability to the amount paid under the contract."
        },
        {
            "regex": r"\bauto[- ]?renews\b|\bautomatically renews\b",
            "issue": "Automatic Renewal",
            "risk": "medium",
            "suggestion": "Add 30-day notice to opt-out before renewal."
        },
        {
            "regex": r"\bterminate at any time\b|\bwithout cause\b",
            "issue": "One-Sided Termination",
            "risk": "medium",
            "suggestion": "Ensure both parties have equal termination rights."
        },
        {
            "regex": r"\bgoverned by [^\n]+ law\b|\bjurisdiction in [^\n]+\b",
            "issue": "Foreign Jurisdiction",
            "risk": "high",
            "suggestion": "Use local law and courts for enforceability."
        },
    ]

    # Combine patterns and pre-compile regex
    all_patterns = base_patterns + (custom_patterns or [])
    compiled_patterns = []
    for p in all_patterns:
        try:
            compiled_patterns.append({
                **p,
                "compiled": re.compile(p["regex"], re.IGNORECASE)
            })
        except re.error:
            continue  # Skip invalid patterns

    findings = []
    seen_matches = set()  # Track (line_number, matched_text) to avoid dupes

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total_lines = len(lines)

    for line_num, line in enumerate(lines, start=1):
        for pattern in compiled_patterns:
            for match in pattern["compiled"].finditer(line):
                matched_text = match.group()
                match_key = (line_num, matched_text.lower())

                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)

                # Get context with accurate line numbers
                start_line = max(0, line_num - 1 - context_lines)
                end_line = min(total_lines, line_num + context_lines)
                context = "\n".join(
                    f"{i+1}: {line}" 
                    for i, line in enumerate(lines[start_line:end_line], start=start_line)
                )
                # Highlight all matches in line
                highlighted = pattern["compiled"].sub(
                    lambda m: f"**{m.group()}**", 
                    line
                )

                findings.append(ContractFinding(
                    line=line_num,
                    matched_text=highlighted,
                    full_context=context,
                    issue=pattern["issue"],
                    risk=pattern["risk"],
                    suggestion=pattern["suggestion"],
                    pattern_used=pattern["regex"]
                ))

    # Sort findings by line number
    findings.sort(key=lambda x: x.line)
    
    return findings
def clean_text_for_pdf(text: str, replace_newlines: bool = True) -> str:
    """
    Clean text for PDF generation by:
    - Replacing unsupported/unwanted Unicode characters
    - Handling whitespace and line breaks
    - Removing control characters
    - Optionally preserving or replacing newlines

    Args:
        text: Input text to clean
        replace_newlines: If True, replaces newlines with spaces (good for paragraphs).
                         If False, preserves newlines (good for preformatted text).

    Returns:
        Cleaned text suitable for PDF generation
    """
    if not text:
        return ""

    # Common Unicode replacements
    replacements = {
        # Dashes and quotes
        "—": "--",   # em dash
        "–": "-",    # en dash
        "―": "--",   # horizontal bar
        "’": "'",    # curly apostrophe
        "‘": "'",    # left single quote
        "“": '"',    # left double quote
        "”": '"',    # right double quote
        "…": "...",  # ellipsis
        "•": "*",    # bullet point
        "·": "*",    # middle dot
        "®": "(R)",  # registered trademark
        "©": "(C)",  # copyright
        "™": "(TM)", # trademark
        "°": " deg", # degree symbol
        "±": "+/-",  # plus-minus
        "×": "x",    # multiplication sign
        "÷": "/",    # division sign
        "¼": "1/4",
        "½": "1/2",
        "¾": "3/4",
    }

    # Replace Unicode characters
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Handle whitespace
    text = " ".join(text.split())  # Remove extra whitespace between words

    # Handle newlines based on parameter
    if replace_newlines:
        text = text.replace("\n", " ").replace("\r", " ")
    else:
        text = text.replace("\r\n", "\n").replace("\r", "\n")  # Normalize line endings

    # Remove control characters (except basic whitespace)
    text = "".join(
        char for char in text 
        if char == "\n" or char.isprintable()
    )

    # Final cleanup of any remaining special cases
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)  # Remove non-printable chars
    text = text.strip()

    return text

def generate_report_pdf(summary: str, findings: List[Dict], company_name: str = None) -> bytes:
    """
    Generate a professional PDF report from contract analysis results.
    
    Args:
        summary: Executive summary of the analysis
        findings: List of risk findings (each containing issue, risk, suggestion)
        company_name: Optional company name for report header
    
    Returns:
        PDF file as bytes
    """
    from fpdf import FPDF
    from datetime import datetime
    from typing import List, Dict

    class ContractPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)
            self.set_margins(left=15, top=15, right=15)
            self.company_name = company_name
            self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)

        def header(self):
            self.set_font('DejaVu', 'B', 16)
            title = 'LegalEase AI — Contract Analysis Report'
            if self.company_name:
                title += f"\nfor {self.company_name}"
            self.cell(0, 10, title, ln=True, align='C')
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('DejaVu', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
            self.set_y(-12)
            self.set_font('DejaVu', 'I', 6)
            self.cell(0, 10, 'AI-Generated Analysis - Not Legal Advice', 0, 0, 'C')

        def add_section(self, title, content, style='B'):
            self.set_font('DejaVu', style, 12)
            self.set_text_color(0, 0, 139)  # Dark blue
            self.cell(0, 8, title, ln=True)
            self.set_text_color(0, 0, 0)  # Black
            self.set_font('DejaVu', '', 10)
            self.multi_cell(0, 5, content)
            self.ln(5)

        def add_risk_table(self, findings):
            if not findings:
                self.set_font('DejaVu', 'I', 10)
                self.cell(0, 8, 'No significant risks detected.', ln=True)
                self.ln(8)
                return

            self.set_font('DejaVu', 'B', 12)
            self.set_text_color(0, 0, 139)  # Dark blue
            self.cell(0, 8, 'Risk Analysis Findings', ln=True)
            self.ln(2)
            
            # Table header
            self.set_fill_color(220, 220, 250)
            self.set_font('DejaVu', 'B', 9)
            self.set_text_color(0, 0, 0)
            self.cell(15, 6, 'Line', 1, 0, 'C', 1)
            self.cell(60, 6, 'Issue', 1, 0, 'C', 1)
            self.cell(20, 6, 'Risk', 1, 0, 'C', 1)
            self.cell(90, 6, 'Recommendation', 1, 1, 'C', 1)

            # Table rows
            self.set_font('DejaVu', '', 8)
            for idx, f in enumerate(findings):
                # Alternate row colors
                fill = 240 if idx % 2 == 0 else 255
                self.set_fill_color(255, 255, 255)
                
                # Risk level colors
                if f["risk"].lower() == "high":
                    self.set_text_color(178, 34, 34)  # Firebrick red
                elif f["risk"].lower() == "medium":
                    self.set_text_color(218, 165, 32)  # Goldenrod
                else:
                    self.set_text_color(0, 100, 0)  # Dark green
                
                self.cell(15, 6, str(f.get("line", "")), 1, 0, 'C', fill)
                self.set_text_color(0, 0, 0)  # Reset to black for other cells
                self.cell(60, 6, f["issue"], 1, 0, 'L', fill)
                self.set_font('DejaVu', 'B', 8)
                self.cell(20, 6, f["risk"].upper(), 1, 0, 'C', fill)
                self.set_font('DejaVu', '', 8)
                self.multi_cell(90, 6, f["suggestion"], 1, 1, 'L', fill)
            
            self.ln(10)

    try:
        pdf = ContractPDF()
        pdf.add_page()
        
        # Cover page
        pdf.set_font('DejaVu', 'B', 20)
        pdf.cell(0, 40, '', ln=True)  # Spacer
        pdf.cell(0, 15, 'Contract Analysis Report', ln=True, align='C')
        if company_name:
            pdf.set_font('DejaVu', 'I', 14)
            pdf.cell(0, 10, f'Prepared for {company_name}', ln=True, align='C')
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, datetime.now().strftime('%B %d, %Y'), ln=True, align='C')
        
        # Content pages
        pdf.add_page()
        pdf.add_section("Executive Summary", summary)
        
        if findings:
            pdf.add_section("Detailed Findings", 
                          "The following table summarizes potential risks identified in the contract:")
            pdf.add_risk_table(findings)
        else:
            pdf.add_section("Analysis Results", 
                          "No significant risks were identified in the contract document.")
        
        # Disclaimer section
        pdf.set_font('DejaVu', 'I', 9)
        pdf.set_text_color(139, 0, 0)  # Dark red
        pdf.multi_cell(0, 5, 
                      "DISCLAIMER: This automated contract analysis report is generated by LegalEase AI and "
                      "is provided for informational purposes only. It does not constitute legal advice. "
                      "Please consult with qualified legal counsel before making any decisions based on this analysis.")
        
        return pdf.output(dest='S').encode('latin-1')
    
    except Exception as e:
        error_pdf = FPDF()
        error_pdf.add_page()
        error_pdf.set_font('Arial', 'B', 12)
        error_pdf.cell(0, 10, 'Report Generation Failed', ln=True, align='C')
        error_pdf.set_font('Arial', '', 10)
        error_pdf.multi_cell(0, 6, f"An error occurred while generating the PDF report:\n{str(e)}")
        return error_pdf.output(dest='S').encode('latin-1')# -----------------------------

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="LegalEase AI",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Theme Configuration
# -----------------------------
def apply_theme(theme):
    """Apply theme-specific CSS styles"""
    if theme == "🌙 Dark":
        st.markdown("""
            <style>
                :root {
                    --primary: #1f2229;
                    --secondary: #2a2a3a;
                    --text: #fafafa;
                    --accent: #4e8cff;
                    --user-bg: #1a3a4a;
                    --assistant-bg: #2a2a3a;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                :root {
                    --primary: #f5f5f5;
                    --secondary: #f0f0f0;
                    --text: #000000;
                    --accent: #0068c9;
                    --user-bg: #e6f2ff;
                    --assistant-bg: #f0f0f0;
                }
            </style>
        """, unsafe_allow_html=True)
    
    # Shared CSS styles
    st.markdown(f"""
        <style>
            body, .stApp {{
                background-color: var(--primary);
                color: var(--text);
            }}
            .stButton>button, .stDownloadButton>button {{
                background-color: var(--accent);
                color: white;
                border-radius: 8px;
                padding: 0.5em 1em;
                border: none;
                transition: all 0.3s;
            }}
            .stButton>button:hover {{
                opacity: 0.8;
                transform: scale(1.02);
            }}
            .stTextInput>div>div>input {{
                background-color: var(--primary);
                color: var(--text);
                border-radius: 8px;
                border: 1px solid var(--accent);
            }}
            .chat-message-user {{
                background-color: var(--user-bg) !important;
                padding: 1em;
                border-radius: 10px;
                margin: 0.5em 0;
                color: var(--text) !important;
            }}
            .chat-message-assistant {{
                background-color: var(--assistant-bg) !important;
                padding: 1em;
                border-radius: 10px;
                margin: 0.5em 0;
                color: var(--text) !important;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                padding: 0.5rem 1rem;
                border-radius: 8px !important;
                transition: all 0.3s;
            }}
            .stTabs [aria-selected="true"] {{
                background-color: var(--accent);
                color: white !important;
            }}
            .stMarkdown h1 {{
                margin-bottom: 0.5rem;
            }}
            .disclaimer {{
                font-size: 0.8rem;
                color: #ff4b4b !important;
                margin-top: 0.5rem;
            }}
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar Configuration
# -----------------------------
with st.sidebar:
    # Theme Selector
    st.markdown("## 🌓 Display Settings")
    theme = st.radio("Color Theme", ["🌞 Light", "🌙 Dark"], key="theme_mode", horizontal=True)
    apply_theme(theme)
    
    # App Settings
    st.markdown("## ⚙️ Configuration")
    debug_mode = st.checkbox("🛠️ Debug Mode", help="Enable detailed logging")
    language = st.selectbox("🌍 Language", ["English", "Amharic", "Oromo"])
    
    # AI Settings
    st.markdown("## 🤖 AI Preferences")
    chatbot_mode = st.selectbox(
        "Assistant Mode",
        ["General Legal", "Contract Analysis", "Document Review"],
        help="Specialize the AI's responses"
    )
    
    chatbot_tone = st.select_slider(
        "Response Style",
        options=["Technical", "Balanced", "Simple"],
        value="Balanced"
    )
    
    # Legal Resources
    st.markdown("---")
    st.markdown("## 📚 Legal Resources")
    with st.expander("Ethiopian Legal Documents"):
        st.markdown("- [Civil Code](https://chilot.me)")
        st.markdown("- [Commercial Code](https://chilot.me)")
        st.markdown("- [Constitution](https://chilot.me)")
    
    # Footer
    st.markdown("---")
    st.markdown("### About LegalEase")
    st.caption("v1.0.0 | Made with ❤️ in Ethiopia")
    
# -----------------------------
# Main Page Layout
# -----------------------------
st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <h1 style="margin: 0;">💼 LegalEase AI</h1>
        <span style="font-size: 0.8rem; background: #4e8cff; color: white; padding: 0.2rem 0.5rem; border-radius: 1rem;">BETA</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("#### Democratizing Legal Access Through AI Technology")
st.markdown('<p class="disclaimer">⚠️ Disclaimer: AI-generated content. Not a substitute for professional legal advice.</p>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Main Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗨️ Smart Chat", 
    "📄 Document Tools",
    "🔍 Risk Analyzer",
    "⚖️ Legal Resources"
])

with tab1:
    st.markdown("### 💬 Chat with LegalEase AI")
    st.caption("Ask general legal questions or about specific cases")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Placeholder for AI response
        with st.chat_message("assistant"):
            response = f"AI response to: {prompt}"
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Document Summarizer")
        uploaded_file = st.file_uploader("Upload legal document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            if st.button("Generate Summary"):
                with st.spinner("Analyzing document..."):
                    # Placeholder for summary logic
                    st.success("Summary generated!")
                    st.markdown("**Document Summary:**")
                    st.write("This would be the AI-generated summary of the uploaded document.")
    
    with col2:
        st.markdown("### ✍️ Clause Extractor")
        st.caption("Identify specific clauses in contracts")
        if uploaded_file and st.button("Extract Clauses"):
            with st.spinner("Processing clauses..."):
                # Placeholder for clause extraction
                st.success("Clauses extracted!")
                with st.expander("View Extracted Clauses"):
                    st.write("1. **Termination Clause** (Page 3)")
                    st.write("2. **Liability Clause** (Page 5)")

with tab3:
    st.markdown("### 🔍 Contract Risk Analyzer")
    risk_file = st.file_uploader("Upload contract for analysis", type=["pdf", "docx"])
    
    if risk_file:
        if st.button("Analyze Risks", type="primary"):
            with st.spinner("Scanning for legal risks..."):
                # Placeholder for risk analysis
                st.success("Analysis complete!")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("High Risks", 3)
                    st.metric("Medium Risks", 2)
                    st.metric("Low Risks", 1)
                
                with col2:
                    with st.expander("🔴 High Risk: Unlimited Liability", expanded=True):
                        st.markdown("**Found on:** Page 4, Section 2.3")
                        st.markdown("**Recommendation:** Cap liability to contract value")
                    
                    with st.expander("🟠 Medium Risk: Automatic Renewal"):
                        st.markdown("**Found on:** Page 2, Section 1.5")
                        st.markdown("**Recommendation:** Add 30-day notice period")

with tab4:
    st.markdown("### ⚖️ Ethiopian Legal Resources")
    
    with st.container():
        st.markdown("#### 📜 Primary Legislation")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Civil Code**")
            st.caption("Proclamation No. 123/2021")
            st.download_button("Download PDF", "#")
        with cols[1]:
            st.markdown("**Commercial Code**")
            st.caption("Proclamation No. 124/2021")
            st.download_button("Download PDF", "#")
        with cols[2]:
            st.markdown("**Labor Law**")
            st.caption("Proclamation No. 1156/2019")
            st.download_button("Download PDF", "#")
    
    st.markdown("---")
    st.markdown("#### 🏛️ Government Portals")
    st.markdown("- [Federal Supreme Court](https://supremecourt.gov.et)")
    st.markdown("- [Ministry of Justice](https://moj.gov.et)")
    st.markdown("- [Ethiopian Law Archive](https://lawethiopia.com)")

# -----------------------------
# Responsive Adjustments
# -----------------------------
st.markdown("""
    <style>
        @media screen and (max-width: 768px) {
            .stTabs [data-baseweb="tab-list"] {
                flex-direction: column;
            }
            .stTabs [data-baseweb="tab"] {
                width: 100%;
                margin-bottom: 0.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)# -----------------------------
# TAB 1: Enhanced Chatbot
# -----------------------------

with tab1:
    st.header("💬 Smart Legal Assistant")
    st.markdown("Ask about Ethiopian civil law, contracts, rights, and more. You can also upload documents for analysis.")
    
    # Initialize chat history
    if "messages" not in st.session_state:  
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Hello! I'm LegalEase AI. I can help with:\n- Ethiopian contract law\n- Document review\n- Legal concepts\n\nHow can I assist you today?",
            "timestamp": datetime.now().isoformat()
        }]
    
    # Display chat messages
    for msg in st.session_state.messages:  
        with st.chat_message(msg["role"]):  
            st.write(msg["content"])
            if "timestamp" in msg:
                st.caption(f"{datetime.fromisoformat(msg['timestamp']).strftime('%H:%M')}")
    
    # Document upload for chat context
    uploaded_file = st.file_uploader(
        "📎 Upload document for context (optional)",
        type=["pdf", "docx"],
        key="chat_file"
    )
    doc_context = ""
    findings = []
    if uploaded_file:
        with st.spinner("Processing document..."):
            file_bytes = uploaded_file.read()  
            file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"  
            doc_context = extract_text_from_file(file_bytes, file_type)
            if len(doc_context.strip()) > 50:
                st.success(f"✅ Document loaded: {uploaded_file.name}")
                findings = analyze_contract_text(doc_context)
            else:
                st.warning("⚠️ Could not extract enough text from the document.")
    
    # Response generation logic
    def generate_response(prompt, context=None, mode="General Legal", tone="Balanced"):
        # Ethiopian legal knowledge base
        legal_db = {
            "contract": {
                "validity": {
                    "source": "Ethiopian Civil Code Articles 1678-1690",
                    "content": "Four essential elements:\n1. Consent of the parties\n2. Capacity to contract\n3. Lawful object\n4. Lawful cause",
                    "examples": "Common issues: unclear terms, unequal bargaining power, illegal purposes"
                },
                "termination": {
                    "source": "Civil Code Articles 1806-1820 & Labor Proclamation No. 1156/2019",
                    "content": "Termination conditions vary by contract type:\n- Employment: Notice periods apply\n- Commercial: May include breach clauses",
                    "notice": "Reasonable notice typically required (30 days common)"
                }
            },
            "property": {
                "rights": {
                    "source": "Constitution Article 40 & Civil Code",
                    "content": "Right to private property guaranteed\nExpropriation requires fair compensation"
                }
            }
        }

        # Generate base response
        response = ""
        sources = []
        
        # Contract-related questions
        if "contract" in prompt.lower() or "agreement" in prompt.lower():
            if "valid" in prompt.lower() or "enforce" in prompt.lower():
                contract = legal_db["contract"]["validity"]
                response = f"Contract Validity Requirements:\n{contract['content']}\n\nExamples: {contract['examples']}"
                sources.append(contract["source"])
            elif "terminat" in prompt.lower() or "end" in prompt.lower():
                termination = legal_db["contract"]["termination"]
                response = f"Contract Termination Rules:\n{termination['content']}\n\nNotice: {termination['notice']}"
                sources.append(termination["source"])
        
        # Property-related questions
        elif "property" in prompt.lower() or "land" in prompt.lower():
            property = legal_db["property"]["rights"]
            response = f"Property Rights:\n{property['content']}"
            sources.append(property["source"])
        
        # Default response
        else:
            response = f"Regarding {prompt}, Ethiopian law generally requires... [provide general principles]"

        # Add document context if available
        if context:
            summary = summarize_text(context)
            response = f"Document Context:\n- Summary: {summary[:200]}...\n\n{response}"
            
            if findings:
                response += "\n\nKey Document Findings:"
                for f in findings[:3]:
                    response += f"\n- {f['issue']} (Risk: {f['risk'].upper()})"

        # Add legal sources
        if sources:
            response += "\n\nLegal Sources:\n- " + "\n- ".join(sources)

        # Apply tone
        if tone == "Simple":
            response = response.replace("typically", "usually")
            response = response.replace("proportionality", "fair share")
            response = response.replace("shall be considered", "should be thought of as")
        elif tone == "Formal":
            response = response.replace("can", "may")
            response = response.replace("should", "it is advisable to")
            response = "In accordance with legal principles, " + response

        # Final disclaimer
        response += "\n\nNote: This is general information, not legal advice. Consult a qualified attorney."

        return response

    # Chat input
    if prompt := st.chat_input("Ask a legal question..."):  
        # Add user message
        user_msg = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):  
            st.write(prompt)
            st.caption(f"{datetime.fromisoformat(user_msg['timestamp']).strftime('%H:%M')}")
        
        # Generate and display response
        with st.spinner("Analyzing..."):  
            message_placeholder = st.empty()
            full_response = ""
            
            response = generate_response(
                prompt,
                context=doc_context if uploaded_file else None,
                mode=chatbot_mode,
                tone=chatbot_tone
            )

            # Simulate typing
            for chunk in response.split('\n'):
                full_response += chunk + '\n'
                time.sleep(0.1)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Add assistant response
            assistant_msg = {
                "role": "assistant", 
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_msg)
    
    # Chat management buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant", 
                "content": "Hello! I'm LegalEase AI. How can I help you with Ethiopian law today?",
                "timestamp": datetime.now().isoformat()
            }]
            st.rerun()
    with col2:
        if st.session_state.messages:
            chat_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])  
            st.download_button(
                "📥 Save Chat", 
                data=chat_text, 
                file_name="legalease_chat.txt",
                use_container_width=True
            )
    with col3:
        example_questions = [
            "What makes a contract valid in Ethiopia?",
            "How can I terminate a lease agreement?",
            "What are my property rights?"
        ]
        example = st.selectbox(
            "💡 Example Questions",
            ["Select an example..."] + example_questions,
            index=0
        )
        if example and example != "Select an example...":
            if "example_used" not in st.session_state or st.session_state.example_used != example:
                st.session_state.example_used = example
                st.session_state.messages.append({
                    "role": "user", 
                    "content": example,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

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

            st.markdown("### 🎯 Filter Risks")  
            selected_levels = st.multiselect("Select risk levels to show", ["high", "medium"], default=["high", "medium"])  
            filtered = [f for f in findings if f["risk"] in selected_levels]  

            if filtered:  
                st.metric("Total Issues", len(filtered))  
                for i, f in enumerate(filtered):  
                    with st.expander(f"📌 {f['issue']} (Line {f['line']})"):  
                        st.write(f"**Clause**: {f['matched_text']}")
                        st.write(f"**Risk Level**: {'🔴 High' if f['risk']=='high' else '🟡 Medium'}")  
                        st.info(f"💡 **Suggestion**: {f['suggestion']}")  
            else:  
                st.success("✅ No major risks detected!")  

            try:  
                pdf_bytes = generate_report_pdf(summary, findings)  
                st.markdown("### 📄 Download Report")  
                st.download_button(  
                    label="📥 Download Full Report (PDF)",  
                    data=pdf_bytes,  
                    file_name=f"legalease_analysis_{uploaded_file.name.split('.')[0]}.pdf",  
                    mime="application/pdf"  
                )  
            except Exception as e:  
                st.error(f"Report generation failed: {e}")

# -----------------------------
# TAB 4: Clause Extractor
# -----------------------------

with tab4:
    st.header("📚 Clause Extractor")
    uploaded_file = st.file_uploader("Upload Legal PDF/DOCX", type=["pdf", "docx"], key="clause")

    if uploaded_file:  
        file_bytes = uploaded_file.read()  
        file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"  
        text = extract_text_from_file(file_bytes, file_type)  

        if len(text.strip()) < 50:  
            st.warning("⚠️ Not enough text extracted.")  
        else:  
            st.success("✅ Extracted contract text.")  
            st.markdown("### 📑 Detected Clauses")  

            clauses = re.split(r"\n(?=\d+\.\s)", text)  
            for i, clause in enumerate(clauses):  
                if len(clause.strip()) > 20:  
                    with st.expander(f"📄 Clause {i+1}"):  
                        st.write(clause.strip())

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.caption("Built by ASET (Alpha Software Engineering Tutorial) for Ethiopian Legal Empowerment • Capstone 2025")
