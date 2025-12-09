# app_streamlit_chroma_hf_ollama.py
"""
Streamlit app: RAG with Chromadb + HuggingFace Inference embeddings + Ollama LLM (no LangChain).
- Extracts text from PDFs using pypdf
- Chunks text with sliding window
- Embeds via HF Inference API
- Stores embeddings + docs in chromadb collection (persisted)
- Queries chromadb with HF query embedding
- Calls Ollama for generation / image analysis
WARNING: keep HF token private; remove fallback before publishing.
"""

import os
import hashlib
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import ollama

# -------------------------
# CONFIG & CONSTANTS
# -------------------------
HF_TOKEN_FALLBACK = "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"  # remove before publishing
DEFAULT_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TMP_DIR = Path(os.environ.get("STREAMLIT_CHROMA_DIR", tempfile.gettempdir())) / "chroma_persist"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOADS = 6
MAX_UPLOAD_MB = 200
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
HF_BATCH_SIZE = 16
HF_TIMEOUT = 60

st.set_page_config(page_title="RAG — Chroma + HF + Ollama", layout="wide")

# -------------------------
# Utilities
# -------------------------
def get_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if token and token.strip():
        return token.strip()
    return HF_TOKEN_FALLBACK

class HFInferenceEmbeddings:
    """Simple HF Inference embeddings wrapper (embeddings endpoint)."""
    def __init__(self, model: str = DEFAULT_HF_EMBED_MODEL):
        token = get_hf_token()
        if not token:
            raise RuntimeError("HF API token missing (set HF_API_TOKEN env var).")
        self.url = f"https://api-inference.huggingface.co/embeddings/{model}"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _post(self, inputs: List[str]) -> List[List[float]]:
        payload = {"inputs": inputs}
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=HF_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), HF_BATCH_SIZE):
            batch = texts[i : i + HF_BATCH_SIZE]
            vectors = self._post(batch)
            out.extend(vectors)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]

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

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks

# -------------------------
# Chromadb helpers (direct client)
# -------------------------
def chroma_client(persist_directory: str):
    # Use local disk persistence
    return chromadb.Client(Settings(persist_directory=persist_directory, chroma_db_impl="duckdb+parquet"))

def ensure_collection_for_fp(client: chromadb.Client, fp: str):
    col_name = f"col_{fp}"
    try:
        return client.get_collection(name=col_name)
    except Exception:
        # create new collection
        return client.create_collection(name=col_name)

def index_documents_to_chroma(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    hf_embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> Dict[str, Any]:
    if not uploaded_files:
        return None

    fp = fingerprint_files(uploaded_files)
    persist_dir = str(TMP_DIR / fp)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chroma_client(persist_directory=persist_dir)
    collection = ensure_collection_for_fp(client, fp)

    # Check if collection already has data -> reuse
    try:
        existing_count = collection.count()
        if existing_count and existing_count > 0:
            return {"collection": collection, "persist_dir": persist_dir, "indexed_chunks": existing_count}
    except Exception:
        existing_count = None

    texts = []
    metadatas = []
    ids = []
    doc_refs = []  # human readable doc references for previews

    for uploaded in uploaded_files[:MAX_UPLOADS]:
        # size guard
        try:
            size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
        except Exception:
            size = 0
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipping {uploaded.name} (> {MAX_UPLOAD_MB} MB)")
            continue

        # save file to raw directory
        raw_dir = Path(persist_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())

        # extract text
        text = ""
        if uploaded.name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(str(target))
                pages = []
                for p in reader.pages:
                    pages.append(p.extract_text() or "")
                text = "\n\n".join(pages)
            except Exception as e:
                st.info(f"Could not read PDF {uploaded.name}: {e}")
                text = ""
        else:
            try:
                text = target.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    text = target.read_text(encoding="latin-1", errors="ignore")
                except Exception:
                    text = ""

        if not text or not text.strip():
            st.info(f"No text extracted from {uploaded.name}; skipping.")
            continue

        # chunk
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            uid = f"{uploaded.name}__{i}"
            ids.append(uid)
            texts.append(c)
            metadatas.append({"source": uploaded.name, "chunk_index": i})
            doc_refs.append({"name": uploaded.name, "preview": c[:300]})

    if not texts:
        st.warning("No text found to index.")
        return None

    # cap chunks
    if max_chunks and len(texts) > max_chunks:
        st.info(f"Indexing first {max_chunks} chunks out of {len(texts)} for speed.")
        ids = ids[:max_chunks]
        texts = texts[:max_chunks]
        metadatas = metadatas[:max_chunks]

    # embeddings (HF Inference)
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

    # add to chroma collection
    try:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        client.persist()
    except Exception as e:
        st.error(f"Failed to add to Chroma: {e}")
        return None

    return {"collection": collection, "persist_dir": persist_dir, "indexed_chunks": len(texts), "snippets": doc_refs}

def query_chroma(collection: chromadb.api.models.Collection, query: str, hf_model_for_query: str, k: int = 3) -> List[Dict[str, Any]]:
    if collection is None:
        return []
    hf = HFInferenceEmbeddings(model=hf_model_for_query)
    q_emb = hf.embed_query(query)
    try:
        res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    except Exception as e:
        st.error(f"Chroma query error: {e}")
        return []
    docs = []
    docs_list = res.get("documents", [[]])[0]
    metas_list = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs_list, metas_list, dists):
        docs.append({"document": doc, "metadata": meta, "distance": dist})
    return docs

# -------------------------
# Simple memory (windowed) using session_state
# -------------------------
class SimpleMemory:
    def __init__(self, k: int = 4):
        self.k = max(1, int(k))

    def load_memory_variables(self, _) -> Dict[str, List[Dict[str, str]]]:
        msgs = st.session_state.get("messages", [])
        last = msgs[-self.k:] if msgs else []
        return {"chat_history": [{"type": m.get("role", "assistant"), "content": m.get("content", "")} for m in last]}

# -------------------------
# UI / App
# -------------------------
with st.sidebar:
    st.header("Settings")
    text_model = st.selectbox("Text model (ollama)", ["llama3", "mistral"], index=0)
    vision_model = st.selectbox("Vision model (ollama)", ["moondream"], index=0)
    memory_window = st.slider("Memory window", 1, 10, 4)
    st.markdown("---")
    st.subheader("RAG / Indexing")
    hf_embed_model = st.text_input("HF embed model", value=DEFAULT_HF_EMBED_MODEL)
    chunk_size = st.number_input("Chunk size", value=DEFAULT_CHUNK_SIZE, min_value=256)
    chunk_overlap = st.number_input("Chunk overlap", value=DEFAULT_CHUNK_OVERLAP, min_value=0)
    max_chunks = st.number_input("Max chunks to index (0=no limit)", value=0, min_value=0)
    st.markdown("---")
    st.subheader("Uploads")
    uploaded_files = st.file_uploader("Upload PDF / TXT (multiple)", type=["pdf", "txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    if st.button("Clear conversation & retriever"):
        for k in ["messages", "chroma_info"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# session init
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chroma_info" not in st.session_state:
    st.session_state["chroma_info"] = None

# init LLMs (Ollama)
@st.cache_resource
def get_ollama_models(text_model_name: str, vision_model_name: str):
    text_llm = ollama.Chat(model=text_model_name) if hasattr(ollama, "Chat") else ollama  # compatibility
    vision_llm = ollama.Chat(model=vision_model_name) if hasattr(ollama, "Chat") else ollama
    return text_llm, vision_llm

try:
    llm, vision_llm = get_ollama_models(text_model, vision_model)
except Exception as e:
    st.error(f"Ollama client init error: {e}")
    st.stop()

st.title("RAG — Chroma + HF Inference + Ollama")

col1, col2 = st.columns([3, 1])
with col1:
    # render chat
    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

    # image analysis
    if image_file:
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask something about this image:")
        if q_img and st.button("Analyze image"):
            img_bytes = image_file.getvalue()
            with st.spinner("Analyzing image..."):
                try:
                    resp = ollama.generate(model=vision_model, prompt=q_img, images=[img_bytes])
                    answer = resp.get("response") if isinstance(resp, dict) else str(resp)
                except Exception as e:
                    answer = f"Vision model error: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.rerun()

    # text chat form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        submitted = st.form_submit_button("Send")

    if submitted and user_input and user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                chroma_info = st.session_state.get("chroma_info")
                memory = SimpleMemory(k=memory_window)
                if chroma_info and chroma_info.get("collection"):
                    # RAG flow: query chroma
                    collection = chroma_info["collection"]
                    docs = query_chroma(collection, user_input, hf_embed_model, k=3)
                    # build context from docs
                    context_pieces = []
                    for d in docs:
                        context_pieces.append(d.get("document", "")[:1000])
                    context = "\n\n---\n\n".join(context_pieces)
                    # include memory
                    mem = memory.load_memory_variables({}).get("chat_history", [])
                    mem_text = "\n".join(f'{m.get("type")}: {m.get("content")}' for m in mem) if mem else ""
                    prompt = f"Conversation memory:\n{mem_text}\n\nRetrieved docs:\n{context}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
                    # call ollama
                    try:
                        resp = ollama.generate(model=text_model, prompt=prompt)
                        answer = resp.get("response") if isinstance(resp, dict) else str(resp)
                    except Exception as e:
                        answer = f"LLM (Ollama) error: {e}"
                else:
                    # non-RAG: direct LLM call
                    prompt = f"Question:\n{user_input}\n\nAnswer concisely."
                    try:
                        resp = ollama.generate(model=text_model, prompt=prompt)
                        answer = resp.get("response") if isinstance(resp, dict) else str(resp)
                    except Exception as e:
                        answer = f"LLM (Ollama) error: {e}"
            except Exception as e:
                answer = f"Unhandled error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

with col2:
    st.markdown("### RAG / Indexing")
    chroma_info = st.session_state.get("chroma_info")
    if chroma_info:
        st.success("Chroma index loaded.")
        st.write({"persist_dir": chroma_info.get("persist_dir"), "indexed_chunks": chroma_info.get("indexed_chunks")})
        if chroma_info.get("snippets"):
            st.markdown("**Document previews:**")
            for s in chroma_info["snippets"][:5]:
                st.markdown(f"- **{s['name']}** — `{s['preview'][:160]}...`")
    else:
        st.info("No Chroma index loaded. Upload files and click 'Index uploaded files' in the sidebar.")

    st.markdown("---")
    if uploaded_files and st.button("Index uploaded files (HF inference)"):
        with st.spinner("Indexing (HF embeddings -> Chroma)..."):
            info = index_documents_to_chroma(
                uploaded_files=uploaded_files,
                hf_embed_model=hf_embed_model,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_chunks=(None if int(max_chunks) == 0 else int(max_chunks)),
            )
            if info:
                st.session_state["chroma_info"] = info
                st.success(f"Indexed {info.get('indexed_chunks')} chunks.")
            else:
                st.error("Indexing failed. See messages above.")

    st.markdown("---")
    if st.button("Show memory"):
        memory = SimpleMemory(k=memory_window)
        st.write(memory.load_memory_variables({}).get("chat_history", []))

    st.markdown("---")
    st.caption("Set HF_API_TOKEN in environment to override embedded token. Keep token private; remove fallback before publishing.")
