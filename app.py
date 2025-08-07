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

# -----------------------------
# Modern UI Setup with Themes
# -----------------------------
import streamlit as st

st.set_page_config(
    page_title="LegalEase AI",
    layout="wide",
    page_icon="⚖️"
)

# ---- Theming ----
theme = st.sidebar.radio("🌓 Theme Mode", ["🌞 Light", "🌙 Dark"], key="theme_mode")

# Dynamic CSS styling based on theme
if theme == "🌙 Dark":
    st.markdown("""
        <style>
            body, .stApp { background-color: #0e1117; color: #fafafa; }
            .stButton>button, .stDownloadButton>button {
                background-color: #1f2229;
                color: white;
                border-radius: 8px;
                padding: 0.5em 1em;
            }
            .stTextInput>div>input {
                background-color: #1f2229;
                color: white;
                border-radius: 5px;
            }
            .chat-message-user {
                background-color: #1a3a4a !important;
                padding: 1em;
                border-radius: 10px;
            }
            .chat-message-assistant {
                background-color: #2a2a3a !important;
                padding: 1em;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: #ffffff; color: #000000; }
            .stButton>button, .stDownloadButton>button {
                background-color: #f5f5f5;
                color: #000;
                border-radius: 8px;
                padding: 0.5em 1em;
            }
            .stTextInput>div>input {
                background-color: #ffffff;
                color: #000000;
                border-radius: 5px;
            }
            .chat-message-user {
                background-color: #e6f2ff !important;
                padding: 1em;
                border-radius: 10px;
            }
            .chat-message-assistant {
                background-color: #f0f0f0 !important;
                padding: 1em;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# ---- Header Section ----
st.markdown("<h1 style='font-size: 2.5em;'>💼 LegalEase AI</h1>", unsafe_allow_html=True)
st.markdown("#### Empowering Ethiopian Legal Access Through AI")
st.markdown("🚨 *This tool does not provide legal advice. Always consult a licensed attorney.*")
st.markdown("---")

# ---- Sidebar: Settings ----
with st.sidebar:
    st.markdown("## ⚙️ App Settings")
    st.checkbox("🛠️ Debug Mode", key="debug_mode")

    st.markdown("## 🤖 Chatbot Preferences")
    chatbot_mode = st.selectbox(
        "AI Mode",
        ["General Legal", "Contract Analysis", "Document Review"],
        help="Choose the AI assistant's focus"
    )

    chatbot_tone = st.select_slider(
        "Tone of Responses",
        options=["Formal", "Balanced", "Simple"],
        value="Balanced"
    )

    st.markdown("---")
    st.markdown("### 📚 Legal Resources")
    st.markdown("- [Ethiopian Civil Code](https://chilot.me)")
    st.markdown("- [Contract Law Guidelines](https://chilot.me)")
    st.markdown("- [Legal Aid Services](https://ethiopianlaw.org)")

# ---- Main Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🗨️ Smart Chat",
    "📄 Summarize Document",
    "🔍 Analyze Legal Risks",
    "📚 Extract Contract Clauses"
])

# Optional: Set wide container spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

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
