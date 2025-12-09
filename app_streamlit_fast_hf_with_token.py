# app_streamlit_fast_hf_no_langchain_memory.py
"""
Streamlit RAG app using HF Inference embeddings + Ollama.
This version avoids importing langchain.memory (uses a simple memory wrapper).
Keep HF_API_TOKEN as env var (or fallback embedded token).
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import requests
import streamlit as st

# Ollama & LangChain-ish imports
import ollama
from langchain_ollama import ChatOllama

# Keep only the langchain pieces we need for loaders/splitters/vectorstore
# (these are commonly available in modern langchain packages)
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document

# -------------------------
# HF TOKEN (FALLBACK EMBEDDED)
# -------------------------
HF_TOKEN_FALLBACK = "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"

def get_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if token and token.strip():
        return token.strip()
    return HF_TOKEN_FALLBACK

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Fast RAG (HF Inference) - No langchain.memory", layout="wide")
TMP_DIR = Path(os.environ.get("STREAMLIT_CHROMA_DIR", tempfile.gettempdir())) / "chroma_persist"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOADS = 5
MAX_UPLOAD_MB = 100
DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_CHUNKS = 1200
HF_BATCH_SIZE = 16
HF_TIMEOUT = 60

# -------------------------
# Simple memory wrapper (replaces langchain.memory.ConversationBufferWindowMemory)
# -------------------------
class SimpleMemory:
    """
    Very small memory wrapper that returns the last `k` messages from st.session_state['messages'].
    load_memory_variables({}) -> {'chat_history': [ { 'type': role, 'content': text }, ... ] }
    """
    def __init__(self, k: int = 4):
        self.k = max(1, int(k))

    def load_memory_variables(self, _: Dict = None) -> Dict[str, List[Dict[str, str]]]:
        msgs = st.session_state.get("messages", [])
        # Keep last k messages (most recent)
        last_msgs = msgs[-self.k:] if msgs else []
        # Normalize into simple dicts: { type: role, content: text }
        chat_history = []
        for m in last_msgs:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            chat_history.append({"type": role, "content": content})
        return {"chat_history": chat_history}

# -------------------------
# Simple QA template (string) - replaces PromptTemplate import
# -------------------------
qa_template_str = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely."

# -------------------------
# HF Inference embeddings wrapper
# -------------------------
class HFInferenceEmbeddings:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        token = get_hf_token()
        if not token:
            raise RuntimeError("HF API token not available.")
        self.url = f"https://api-inference.huggingface.co/embeddings/{model}"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _post(self, inputs: List[str]) -> List[List[float]]:
        payload = {"inputs": inputs}
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=HF_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), HF_BATCH_SIZE):
            batch = texts[i : i + HF_BATCH_SIZE]
            out = self._post(batch)
            embeddings.extend(out)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]

# -------------------------
# Cached LLMs (Ollama)
# -------------------------
@st.cache_resource
def get_llms(text_model: str = "llama3", vision_model: str = "moondream"):
    text_llm = ChatOllama(model=text_model)
    vision_llm = ChatOllama(model=vision_model)
    return text_llm, vision_llm

# -------------------------
# Helpers: fingerprint & chroma path
# -------------------------
def fingerprint_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode("utf-8"))
        try:
            size = getattr(f, "size", None)
            if size is None:
                size = len(f.getbuffer())
        except Exception:
            size = 0
        h.update(str(size).encode("utf-8"))
    return h.hexdigest()[:16]

def chroma_path_for_fp(fp: str) -> Path:
    p = TMP_DIR / f"chroma_{fp}"
    p.mkdir(parents=True, exist_ok=True)
    return p

# -------------------------
# Ingest & index
# -------------------------
def ingest_files_to_chroma(
    uploaded_files,
    hf_embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
):
    if not uploaded_files:
        return None

    fp = fingerprint_files(uploaded_files)
    chroma_dir = chroma_path_for_fp(fp)

    # reuse existing chroma if present
    if any(chroma_dir.iterdir()):
        try:
            vectordb = Chroma(persist_directory=str(chroma_dir))
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            return {"retriever": retriever, "chroma_path": str(chroma_dir), "chunks_indexed": None}
        except Exception:
            pass

    docs: List[Document] = []
    snippets = []
    raw_dir = chroma_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for uploaded in uploaded_files[:MAX_UPLOADS]:
        try:
            size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
        except Exception:
            size = 0
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipping {uploaded.name} > {MAX_UPLOAD_MB}MB")
            continue

        target = raw_dir / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())

        if uploaded.name.lower().endswith(".pdf"):
            # try PyPDFLoader
            try:
                loader = PyPDFLoader(str(target))
                loaded = loader.load()
                if loaded:
                    docs.extend(loaded)
                    snippets.append({"name": uploaded.name, "preview": loaded[0].page_content[:300]})
                    continue
            except Exception:
                pass
            # fallback to pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(target))
                for i, page in enumerate(reader.pages):
                    txt = page.extract_text() or ""
                    docs.append(Document(page_content=txt, metadata={"source": uploaded.name, "page": i + 1}))
                if txt:
                    snippets.append({"name": uploaded.name, "preview": txt[:300]})
                    continue
            except Exception:
                st.info(f"No text extracted from PDF {uploaded.name} (likely scanned). Skipping.")
        else:
            # text file with encoding fallback
            for enc in ("utf-8", "latin-1"):
                try:
                    loader = TextLoader(str(target), encoding=enc)
                    loaded = loader.load()
                    docs.extend(loaded)
                    if loaded:
                        snippets.append({"name": uploaded.name, "preview": loaded[0].page_content[:300]})
                    break
                except Exception:
                    continue

    if not docs:
        st.warning("No documents extracted for indexing.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if not chunks:
        st.warning("No chunks produced.")
        return None

    if max_chunks and len(chunks) > max_chunks:
        st.info(f"Indexing limited to first {max_chunks} chunks (from {len(chunks)})")
        chunks = chunks[:max_chunks]

    texts = [getattr(d, "page_content", str(d)) for d in chunks]

    # HF embeddings batched
    hf = HFInferenceEmbeddings(model=hf_embed_model)
    embeddings = []
    progress = st.progress(0)
    n = len(texts)
    for i in range(0, n, HF_BATCH_SIZE):
        batch = texts[i : i + HF_BATCH_SIZE]
        try:
            out = hf.embed_documents(batch)
        except Exception as e:
            st.error(f"HF embeddings error: {e}")
            return None
        embeddings.extend(out)
        progress.progress(min(100, int(100 * (i + len(batch)) / n)))
    progress.empty()

    # wrapper for Chroma
    class SimpleEmbedFn:
        def __init__(self, vectors, hf_instance):
            self._vectors = vectors
            self._hf = hf_instance
        def embed_documents(self, texts_in):
            return self._vectors
        def embed_query(self, text):
            return self._hf.embed_query(text)

    try:
        embed_fn = SimpleEmbedFn(embeddings, hf)
        vectordb = Chroma.from_documents(chunks, embed_fn, persist_directory=str(chroma_dir))
        vectordb.persist()
    except Exception as e:
        st.error(f"Failed to create/persist Chroma DB: {e}")
        return None

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return {"retriever": retriever, "chroma_path": str(chroma_dir), "chunks_indexed": len(chunks), "snippets": snippets}

# -------------------------
# Simple RAG function (no langchain.chains)
# -------------------------
def run_rag_query(retriever, llm, memory: SimpleMemory, user_question: str, qa_template: str, k: int = 3, max_chars: int = 1500):
    mem = memory.load_memory_variables({}).get("chat_history", [])
    if isinstance(mem, list) and mem:
        mem_text = "\n".join(f"{m.get('type','msg')}: {m.get('content','')}" for m in mem)
    else:
        mem_text = ""

    retrieved = []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            retrieved = retriever.get_relevant_documents(user_question)
        elif hasattr(retriever, "similarity_search"):
            retrieved = retriever.similarity_search(user_question, k=k)
        else:
            retrieved = retriever(user_question)
    except Exception:
        retrieved = []

    doc_texts = []
    for d in (retrieved or [])[:k]:
        txt = getattr(d, "page_content", getattr(d, "content", str(d)))
        if txt:
            doc_texts.append(txt)
    context = "\n\n---\n\n".join(doc_texts) if doc_texts else ""
    if mem_text:
        context = f"Conversation memory:\n{mem_text}\n\nRetrieved docs:\n{context}"

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n...[truncated]"

    prompt_text = qa_template.format(context=context, question=user_question)

    try:
        out = llm.invoke(prompt_text)
        answer = getattr(out, "content", str(out))
    except Exception as e:
        answer = f"LLM invocation error: {e}"

    return answer, retrieved

# -------------------------
# Sidebar UI
# -------------------------
with st.sidebar:
    st.header("Settings")
    text_model = st.selectbox("Text model (ollama)", ["llama3", "mistral"], index=0)
    vision_model = st.selectbox("Vision model (ollama)", ["moondream"], index=0)
    memory_window = st.slider("Memory window (messages)", 1, 10, 4)

    st.markdown("---")
    st.subheader("RAG / Indexing")
    hf_embed_model = st.text_input("HF embed model", value="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.number_input("Chunk size", value=DEFAULT_CHUNK_SIZE, min_value=512)
    chunk_overlap = st.number_input("Chunk overlap", value=DEFAULT_CHUNK_OVERLAP, min_value=0)
    max_chunks = st.number_input("Max chunks to index (0 = no limit)", value=DEFAULT_MAX_CHUNKS, min_value=0)
    st.caption("HF API token: set HF_API_TOKEN env var or rely on embedded fallback (not recommended).")

    st.markdown("---")
    st.subheader("Uploads")
    uploaded_files = st.file_uploader("Upload PDF/TXT (max {})".format(MAX_UPLOADS), type=["pdf", "txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Image (optional)", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    if st.button("Clear conversation & retriever"):
        for k in ["messages", "retriever_info"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

# -------------------------
# App init & LLM init
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever_info" not in st.session_state:
    st.session_state.retriever_info = None

try:
    llm, vision_llm = get_llms(text_model, vision_model)
except Exception as e:
    st.error(f"LLM init error: {e}")
    st.stop()

# -------------------------
# Indexing button
# -------------------------
if uploaded_files:
    st.sidebar.markdown("### Uploaded files preview")
    for f in uploaded_files[:MAX_UPLOADS]:
        try:
            b = f.getbuffer()
            preview_text = b.tobytes()[:400].decode("utf-8", errors="ignore") if not f.name.lower().endswith(".pdf") else "<PDF>"
        except Exception:
            preview_text = "<Could not preview>"
        st.sidebar.markdown(f"- **{f.name}** — `{preview_text}`")

    if st.sidebar.button("Index uploaded files (HF inference)"):
        with st.spinner("Indexing (this may take a few moments)..."):
            info = ingest_files_to_chroma(
                uploaded_files,
                hf_embed_model=hf_embed_model,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_chunks=(None if int(max_chunks) == 0 else int(max_chunks))
            )
            if info:
                st.session_state.retriever_info = info
                st.success("Index ready.")
            else:
                st.error("Indexing failed; check HF token and try again.")

# -------------------------
# Use retriever & memory
# -------------------------
use_rag = st.session_state.get("retriever_info") is not None
memory = SimpleMemory(k=memory_window)

# -------------------------
# Main UI (chat)
# -------------------------
st.title("Fast RAG — HF Inference (langchain.memory removed)")

col1, col2 = st.columns([3, 1])
with col1:
    # render messages
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>',
                unsafe_allow_html=True,
            )

    # image analysis
    if image_file:
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask about this image:")
        if q_img and st.button("Analyze image"):
            img_bytes = image_file.getvalue()
            with st.spinner("Calling vision model..."):
                try:
                    resp = ollama.generate(model=vision_model, prompt=q_img, images=[img_bytes])
                    ans = resp.get("response") if isinstance(resp, dict) else str(resp)
                except Exception as e:
                    ans = f"Vision model error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

    # chat form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        send = st.form_submit_button("Send")

    if send and user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                if use_rag:
                    retriever = st.session_state["retriever_info"]["retriever"]
                    answer, retrieved_docs = run_rag_query(
                        retriever=retriever,
                        llm=llm,
                        memory=memory,
                        user_question=user_input,
                        qa_template=qa_template_str,
                        k=3,
                    )
                else:
                    ctx = ""
                    prompt_str = qa_template_str.format(context=ctx, question=user_input)
                    out = llm.invoke(prompt_str)
                    answer = getattr(out, "content", str(out))
                    retrieved_docs = []
            except Exception as e:
                answer = f"Error generating answer: {e}"
                retrieved_docs = []

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if retrieved_docs:
            snippets = []
            for d in retrieved_docs:
                txt = getattr(d, "page_content", getattr(d, "content", str(d)))[:400]
                snippets.append(txt)
            st.session_state.messages.append({"role": "assistant", "content": "\n\nRetrieved snippets:\n\n" + "\n\n---\n\n".join(snippets)})

        st.experimental_rerun()

with col2:
    st.markdown("### RAG / Status")
    if use_rag:
        st.success("RAG enabled (Chroma available).")
        info = st.session_state["retriever_info"]
        st.write({"chroma_path": info.get("chroma_path"), "chunks_indexed": info.get("chunks_indexed")})
        if info.get("snippets"):
            st.markdown("**Snippets:**")
            for s in info["snippets"]:
                st.markdown(f"- **{s['name']}** — `{s['preview'][:160]}`")
    else:
        st.info("No RAG index loaded. Upload files and click 'Index uploaded files (HF inference)' in the sidebar.")

    st.markdown("---")
    if st.button("Show memory"):
        st.write(memory.load_memory_variables({}).get("chat_history", []))

    st.markdown("---")
    st.caption("Set HF_API_TOKEN in your environment to override embedded token. Keep token private.")

# End of file
