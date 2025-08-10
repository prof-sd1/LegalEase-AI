import os
import json
import math
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# Optional libs for file parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

# Use OpenAI Python client - keep this generic so users can adapt
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# ---------------------- Utilities ----------------------

def ensure_openai():
    if openai is None:
        raise RuntimeError("openai library is required. Install with `pip install openai`.")
    if not (os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key")):
        raise RuntimeError("OpenAI API key not found. Please set OPENAI_API_KEY in environment or enter it in the sidebar.")
    openai.api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key")


def read_pdf(file) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is required to read PDF files. Install with `pip install PyPDF2`.")
    reader = PyPDF2.PdfReader(file)
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n\n".join(text)


def read_docx(file) -> str:
    if docx is None:
        raise RuntimeError("python-docx is required to read .docx files. Install with `pip install python-docx`.")
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n\n".join(paragraphs)


def simple_chunk_text(text: str, approx_chars: int = 3000) -> List[str]:
    """Split text into chunks of approx_chars characters trying to keep paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        if current_len + len(p) + 2 <= approx_chars:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                chunks.append('\n\n'.join(current))
            # if paragraph itself is huge, break it
            if len(p) > approx_chars:
                for i in range(0, len(p), approx_chars):
                    chunks.append(p[i:i+approx_chars])
                current = []
                current_len = 0
            else:
                current = [p]
                current_len = len(p) + 2
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


# ---------------------- OpenAI helpers ----------------------

def call_chat_model(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 1000) -> str:
    ensure_openai()
    # Use Chat Completions. Users can update to the exact API patterns they prefer.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


def analyze_chunks(chunks: List[str], tool_prompt: str, model: str, temperature: float) -> List[str]:
    results = []
    for i, chunk in enumerate(chunks):
        prompt = tool_prompt.replace("{{chunk}}", chunk)
        with st.spinner(f"Analyzing chunk {i+1}/{len(chunks)}..."):
            out = call_chat_model(
                system_prompt="You are a helpful legal assistant that explains and simplifies legal text.",
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=1000,
            )
        results.append(out)
    return results


# ---------------------- Prompts ----------------------
SIMPLIFY_PROMPT = (
    "Simplify the following legal text into plain English. Keep the meaning exact but use short, clear sentences. "
    "Then provide a one-sentence summary, potential obligations created, who is protected, and any risky or unusual terms. Return the result as JSON with keys: 'plain_english', 'summary', 'obligations', 'protected_party', 'risks'.\n\nText:\n{{chunk}}"
)

CLAUSE_BREAKDOWN_PROMPT = (
    "Break the following contract excerpt into numbered clauses. For each clause return: clause_number, clause_text (short), plain_english_explanation, parties_impacted, recommended_action (if any). Return JSON list.\n\nText:\n{{chunk}}"
)

QNA_PROMPT = (
    "You are a precise legal assistant. Given the document below, answer the user's question. If the document does not contain enough information say 'Insufficient information in document to answer'.\n\nDocument:\n{{chunk}}\n\nQuestion:\n{question}" 
)

EXTRACT_OBLIGATIONS_PROMPT = (
    "From the following text, extract obligations (who must do what, by when), deadlines, notice periods, termination triggers, and monetary amounts. Return JSON with keys: obligations (list), deadlines (list), notice_periods (list), termination_triggers (list), monetary_items (list).\n\nText:\n{{chunk}}"
)


# ---------------------- Streamlit UI ----------------------

def main():
    st.set_page_config(page_title="Legalease AI — Simplify contracts", layout="wide")
    st.title("Legalease AI — Simplify contracts & explain legal text")

    # Sidebar - configuration
    st.sidebar.header("Settings & API")
    api_key = st.sidebar.text_input("OpenAI API key (or set OPENAI_API_KEY env)", type="password")
    if api_key:
        st.session_state["openai_key"] = api_key

    model = st.sidebar.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    approx_chunk_chars = st.sidebar.number_input("Chunk size (approx characters)", value=3000, step=500)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for speed: paste text or upload a PDF / DOCX. For long contracts, Legalease splits into chunks automatically.")

    # Input panel
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Input")
        input_mode = st.radio("Input method", ["Paste text", "Upload file"], index=0)
        raw_text = ""
        if input_mode == "Paste text":
            raw_text = st.text_area("Paste contract or clause here", height=300)
        else:
            uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx", "txt", "md"])
            if uploaded is not None:
                fname = uploaded.name.lower()
                if fname.endswith(".pdf"):
                    try:
                        raw_text = read_pdf(uploaded)
                    except Exception as e:
                        st.error(f"Failed to read PDF: {e}")
                elif fname.endswith(".docx"):
                    try:
                        raw_text = read_docx(uploaded)
                    except Exception as e:
                        st.error(f"Failed to read docx: {e}")
                else:
                    raw_text = uploaded.read().decode("utf-8")

        st.write("Upload or paste. For best results, give us a short question or tell us what you want (simplify, summarize, obligations, explain clause #3, etc.).")
        user_instruction = st.text_input("What would you like Legalease to do? (e.g. 'Simplify', 'Summarize', 'Extract obligations', 'Explain clause 3', or ask a question')")

        run_button = st.button("Run")

        st.markdown("---")
        st.markdown("Example uses:")
        st.markdown("- Simplify: make contract readable for non-lawyers.\n- Clause breakdown: numbered clauses with plain-language explanations.\n- Extract obligations & dates.\n- Ask a question about whether a clause allows early termination.")

    with col2:
        st.subheader("Output")
        output_area = st.empty()

    # Actions
    if run_button:
        if not raw_text.strip():
            st.warning("Please paste or upload some text first.")
            return
        try:
            ensure_openai()
        except Exception as e:
            st.error(str(e))
            return

        chunks = simple_chunk_text(raw_text, approx_chars=approx_chunk_chars)
        st.info(f"Document split into {len(chunks)} chunk(s).")

        instr = (user_instruction or "Simplify").lower()
        if "simpl" in instr or instr.strip() == "simplify":
            results = analyze_chunks(chunks, SIMPLIFY_PROMPT, model=model, temperature=temperature)
            # try to parse each chunk's JSON (best-effort)
            combined = []
            for r in results:
                try:
                    j = json.loads(r)
                except Exception:
                    # wrap as plain text fallback
                    j = {"raw": r}
                combined.append(j)
            st.session_state["last_result"] = combined
            output_area.json(combined)

        elif "oblig" in instr or "extract" in instr:
            results = analyze_chunks(chunks, EXTRACT_OBLIGATIONS_PROMPT, model=model, temperature=temperature)
            combined = []
            for r in results:
                try:
                    j = json.loads(r)
                except Exception:
                    j = {"raw": r}
                combined.append(j)
            st.session_state["last_result"] = combined
            output_area.json(combined)

        elif "clause" in instr or "break" in instr or "explain" in instr:
            results = analyze_chunks(chunks, CLAUSE_BREAKDOWN_PROMPT, model=model, temperature=temperature)
            combined = []
            for r in results:
                try:
                    j = json.loads(r)
                except Exception:
                    j = {"raw": r}
                combined.append(j)
            st.session_state["last_result"] = combined
            output_area.json(combined)

        else:
            # treat as question — send whole document + question, but chunk-aware: answer per chunk and then synthesize
            question = user_instruction
            q_results = []
            for chunk in chunks:
                prompt = QNA_PROMPT.replace("{{chunk}}", chunk).replace("{question}", question)
                with st.spinner("Asking model..."):
                    out = call_chat_model(system_prompt="You are a helpful legal assistant.", user_prompt=prompt, model=model, temperature=temperature, max_tokens=800)
                q_results.append(out)

            # Synthesize answers
            synth_prompt = (
                "You are a legal assistant. Given these partial answers from document chunks, synthesize a single final answer, state when information is missing, and cite which chunk answers contributed.\n\n"
                "Partial answers:\n" + "\n---\n".join(q_results)
            )
            final = call_chat_model(system_prompt="Synthesize the partial answers carefully.", user_prompt=synth_prompt, model=model, temperature=0.0, max_tokens=500)
            st.session_state["last_result"] = {"question": question, "answers": q_results, "final": final}
            output_area.text_area("Answer", final, height=400)

        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if st.button("Download last result as JSON"):
                data = st.session_state.get("last_result", {})
                st.download_button("Download JSON", json.dumps(data, indent=2), file_name="legalease_result.json")
        with col_dl2:
            if st.button("Copy plain text summary"):
                last = st.session_state.get("last_result")
                st.write("Copied to clipboard is not available in Streamlit server — please download the JSON or copy manually.")

    # Footer / help
    st.markdown("---")
    st.write("⚖️ Legalease AI is for informational and educational use only. This app does not provide legal advice. Consult a qualified lawyer for legal decisions.")


if __name__ == '__main__':
    main()
