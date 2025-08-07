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

# -----------------------------
# Services
# -----------------------------

def extract_text_from_pdf(file_bytes):
    try:
        pdf_stream = BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"PDF extraction error: {str(e)}"

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
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        summary = '. '.join(sentences[:4]) + '...'
        return summary if len(summary) > min_length else "Summary could not be generated."
    except Exception as e:
        return f"Summary generation failed: {e}"

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
            if st.session_state.get("debug"):
                print(f"Checking pattern: {pattern} against line: {line_lower}")
            if re.search(pattern, line_lower):
                findings.append({
                    "line": i + 1,
                    "text": line.strip(),
                    "issue": issue,
                    "risk": risk,
                    "suggestion": suggestion
                })
    return findings

def clean_text_for_pdf(text: str) -> str:
    text = text.replace('—', '-').replace('–', '-')
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

    return bytes(pdf.output(dest='S').encode('latin-1'))

import streamlit as st

# -----------------------------
# Modern UI Setup with Day/Night Theme
# -----------------------------

# Set page config
st.set_page_config(
    page_title="LegalEase AI",
    layout="centered",
    page_icon="⚖️"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* General Reset & Theme Colors */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --text-primary: #1a1a1a;
        --text-secondary: #4a4a4a;
        --accent: #4a6fa5; /* Ethiopian blue-inspired */
        --border: #e0e0e0;
        --shadow: rgba(0, 0, 0, 0.05);
    }

    [data-theme="dark"] {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --text-primary: #f8f9fa;
        --text-secondary: #cbd5e1;
        --accent: #65a3e0;
        --border: #334155;
        --shadow: rgba(0, 0, 0, 0.2);
    }

    body {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        transition: background-color 0.3s, color 0.3s;
    }

    .stApp {
        background-color: var(--bg-primary);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 111, 165, 0.3);
    }

    /* Input fields */
    .stTextInput > div > input,
    .stTextArea > div > textarea {
        border-radius: 8px;
        border: 1px solid var(--border);
        padding: 0.75rem;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Chat bubbles */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
        line-height: 1.5;
    }
    .chat-message-user {
        background-color: #e6f2ff;
        color: #000;
        align-self: flex-end;
        margin-left: auto;
    }
    .chat-message-assistant {
        background-color: #f0f4f8;
        color: #000;
        align-self: flex-start;
    }
    [data-theme="dark"] .chat-message-user {
        background-color: #1d4ed8;
        color: #fff;
    }
    [data-theme="dark"] .chat-message-assistant {
        background-color: #334155;
        color: #fff;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 1rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
    }
    .stSidebar [data-testid="stMarkdownContainer"] > p,
    .stSidebar [data-testid="stMarkdownContainer"] > h2 {
        color: var(--text-primary);
    }

    /* Divider */
    hr {
        border-color: var(--border);
    }

    /* Toggle Switch */
    .theme-toggle {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.5rem 0;
    }
    .switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 30px;
    }
    .switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 30px;
    }
    .slider:before {
        position: absolute;
        content: "";
        height: 22px;
        width: 22px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    input:checked + .slider {
        background-color: #4a6fa5;
    }
    input:checked + .slider:before {
        transform: translateX(30px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 16px;
        color: var(--text-secondary);
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
    }
    </style>
""", unsafe_allow_html=True)

# Theme toggle in sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Settings</h2>", unsafe_allow_html=True)

    # Theme toggle (modern switch)
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    theme = st.checkbox("🌙 Dark Mode", key="dark_mode")
    st.markdown('</div>', unsafe_allow_html=True)

    # Apply theme via HTML class
    if st.session_state.dark_mode:
        st.markdown('<div id="theme-root" data-theme="dark"></div>', unsafe_allow_html=True)
        st.markdown("""
        <script>
        document.querySelector('html').setAttribute('data-theme', 'dark');
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div id="theme-root" data-theme="light"></div>', unsafe_allow_html=True)
        st.markdown("""
        <script>
        document.querySelector('html').removeAttribute('data-theme');
        </script>
        """, unsafe_allow_html=True)

    # Debug mode
    st.checkbox("🔧 Debug Mode", key="debug")

    # Chatbot settings
    st.markdown("### 🤖 Chatbot Settings")
    chatbot_mode = st.selectbox(
        "Chat Mode",
        ["General Legal", "Contract Analysis", "Document Review"],
        help="Select the type of assistance you need"
    )
    chatbot_tone = st.select_slider(
        "Response Tone",
        options=["Formal", "Balanced", "Simple"],
        value="Balanced"
    )

    st.divider()

    st.markdown("### 📚 Legal Resources")
    st.markdown("- [📘 Ethiopian Civil Code](https://www.trans-lex.org/604600/_/ethiopian-civil-code/)")
    st.markdown("- [📘 Contract Law Guidelines](https://legal.thomsonreuters.com/blog/the-principles-of-contract-law/)")
    st.markdown("- [📘 Legal Aid Services](https://help.unhcr.org/ethiopia/services/legal/)")

# Main Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: var(--accent);'>⚖️ LegalEase AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: var(--text-secondary);'>Your AI Legal Assistant for Ethiopian Law</p>", unsafe_allow_html=True)
    st.caption("⚠️ This tool does not provide legal advice. Always consult a licensed attorney.")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗨️ Smart Chat", 
    "📄 Summarize Document", 
    "🔍 Analyze Risks", 
    "📚 Extract Clauses"
])
# -----------------------------
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
                        st.write(f"**Clause**: {f['text']}")  
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
