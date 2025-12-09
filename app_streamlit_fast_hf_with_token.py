# app_streamlit_resilient_hf.py
"""
Resilient Streamlit RAG app — HF InferenceClient preferred, robust fallbacks.
Focus: fix 410 errors for embeddings & image caption by using huggingface_hub.InferenceClient
and trying multiple candidate models before failing gracefully.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import requests
import streamlit as st
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# Try to import huggingface_hub's InferenceClient (preferred)
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    InferenceClient = None
    HF_HUB_AVAILABLE = False

# OCR optional
try:
    from pdf2image import convert_from_bytes, convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -------------------------
# Keys (use Streamlit Secrets / env in production)
# -------------------------
HF_KEY_FALLBACK = "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"
GROQ_KEY_FALLBACK = "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"

def get_hf_token() -> str:
    t = os.getenv("HF_API_TOKEN")
    if t and t.strip(): return t.strip()
    try:
        t = st.secrets.get("HF_API_TOKEN")
        if t and t.strip(): return t.strip()
    except Exception:
        pass
    return HF_KEY_FALLBACK

def get_groq_key() -> str:
    t = os.getenv("GROQ_API_KEY")
    if t and t.strip(): return t.strip()
    try:
        t = st.secrets.get("GROQ_API_KEY")
        if t and t.strip(): return t.strip()
    except Exception:
        pass
    return GROQ_KEY_FALLBACK

# -------------------------
# Config defaults
# -------------------------
DEFAULT_EMBED_MODELS = [
    "intfloat/e5-small",                      # often available for embeddings
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]
DEFAULT_IMAGE_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "microsoft/git-base-coco"
]

TMP_DIR = Path(tempfile.gettempdir()) / "chroma_persist_v_latest"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 200
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HF_BATCH_SIZE = 16
HF_TIMEOUT = 60

st.set_page_config(page_title="RAG — Resilient HF", layout="wide")

# -------------------------
# HF helpers (InferenceClient + HTTP fallback)
# -------------------------
class HFClientFacade:
    """
    Wrapper that prefers huggingface_hub.InferenceClient (if installed).
    Provides:
      - embeddings(model, texts)
      - text_generation(model, prompt)
      - image_caption(model, image_bytes)
    Tries multiple candidate endpoints for robustness.
    """
    def __init__(self, token: str):
        self.token = token
        self.hf_client = None
        if HF_HUB_AVAILABLE and token:
            try:
                self.hf_client = InferenceClient(token=token)
            except Exception:
                self.hf_client = None
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def try_embeddings_http(self, model: str, inputs: List[str]) -> Optional[List[List[float]]]:
        # HTTP endpoints to try (ordered)
        endpoints = [
            f"https://api-inference.huggingface.co/embeddings/{model}",
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
            f"https://api-inference.huggingface.co/models/{model}/pipeline/feature-extraction",
        ]
        payload = {"inputs": inputs}
        for url in endpoints:
            try:
                r = requests.post(url, headers={**self.headers, "Content-Type":"application/json"}, json=payload, timeout=HF_TIMEOUT)
                if r.status_code in (404, 410, 403):
                    # model/endpoint not supported — try next
                    continue
                r.raise_for_status()
                j = r.json()
                # normalize: expect list of vectors
                if isinstance(j, list) and j and all(isinstance(x, list) for x in j):
                    return j
                # sometimes returns list of dicts with "embedding" key
                if isinstance(j, list) and j and isinstance(j[0], dict) and "embedding" in j[0]:
                    return [item["embedding"] for item in j]
                # if it returned a dict with embeddings
                if isinstance(j, dict):
                    for k in ("embedding","embeddings","vector","vectors"):
                        if k in j and isinstance(j[k], list) and all(isinstance(x, list) for x in j[k]):
                            return j[k]
                # else not recognized -> continue
            except requests.HTTPError as he:
                status = getattr(he.response, "status_code", None)
                if status in (404, 410, 403):
                    continue
                raise
            except Exception:
                continue
        return None

    def embeddings(self, model: str, texts: List[str]) -> List[List[float]]:
        # 1) try InferenceClient.embeddings
        if self.hf_client:
            try:
                out = self.hf_client.embeddings(model=model, inputs=texts)
                if isinstance(out, list) and out and all(isinstance(v, list) for v in out):
                    return out
            except Exception:
                pass
        # 2) try HTTP endpoints
        out = self.try_embeddings_http(model, texts)
        if out:
            return out
        # 3) nothing worked -> raise informative error
        raise RuntimeError(f"HF embeddings failed for model '{model}' (tried InferenceClient + HTTP endpoints).")

    def image_caption(self, model: str, image_bytes: bytes) -> str:
        # InferenceClient (preferred) often supports image tasks via client.call
        if self.hf_client:
            try:
                # many HF image caption models accept {"inputs": image_bytes} or files; InferenceClient has .image generation/call
                res = self.hf_client(model=model, inputs=image_bytes)
                # Normalize responses
                if isinstance(res, dict) and "generated_text" in res:
                    return res["generated_text"]
                if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
                    return res[0]["generated_text"]
                if isinstance(res, str):
                    return res
                # try common keys
                if isinstance(res, dict) and "caption" in res:
                    return res["caption"]
            except Exception:
                pass

        # HTTP fallback to model endpoint (file upload)
        url = f"https://api-inference.huggingface.co/models/{model}"
        files = {"inputs": ("image", image_bytes, "image/jpeg")}
        try:
            r = requests.post(url, headers=self.headers, files=files, timeout=HF_TIMEOUT)
            if r.status_code in (404, 410, 403):
                raise RuntimeError(f"Model/endpoint {model} not available (status {r.status_code}).")
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j and isinstance(j[0], dict):
                # typical output contains 'generated_text'
                if "generated_text" in j[0]:
                    return j[0]["generated_text"]
                # some models return {'generated_text': ...} at top-level
            if isinstance(j, dict):
                for k in ("generated_text","caption","text","result"):
                    if k in j:
                        return j[k]
            return str(j)
        except requests.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            raise RuntimeError(f"Image caption HTTP error {status}: {he}")
        except Exception as e:
            raise RuntimeError(f"Image caption error: {e}")

# -------------------------
# Chroma helpers (robust)
# -------------------------
def chroma_client(persist_directory: str) -> chromadb.Client:
    persist_directory = str(persist_directory)
    candidates = [
        {"persist_directory": persist_directory, "chroma_api_impl": "duckdb+parquet"},
        {"persist_directory": persist_directory, "chroma_db_impl": "duckdb+parquet"},
        {"persist_directory": persist_directory},
    ]
    for s in candidates:
        try:
            settings = Settings(**s)
            client = chromadb.Client(settings)
            try:
                _ = client.list_collections()
            except Exception:
                pass
            st.info(f"Chroma client initialized with settings: {list(s.keys())}")
            return client
        except Exception:
            continue
    # fallback
    st.warning("Falling back to in-memory Chroma client (no persistence).")
    return chromadb.Client()

def ensure_collection(client: chromadb.Client, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        try:
            return client.create_collection(name=name)
        except Exception:
            try:
                cols = client.list_collections()
                for c in cols:
                    if isinstance(c, dict) and c.get("name") == name:
                        return client.get_collection(name=name)
            except Exception:
                pass
            raise RuntimeError(f"Could not create/get collection '{name}'")

# -------------------------
# File utilities & chunking
# -------------------------
def fingerprint_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode("utf-8"))
        try:
            size = getattr(f, "size", None) or len(f.getbuffer())
        except Exception:
            size = 0
        h.update(str(size).encode("utf-8"))
    return h.hexdigest()[:16]

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end >= L:
            break
        start = end - overlap
    return chunks

def ocr_pdf_to_text_bytes(pdf_path: Path) -> str:
    if not OCR_AVAILABLE:
        return ""
    pages_txt = []
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
        for im in images:
            pages_txt.append(pytesseract.image_to_string(im))
    except Exception:
        try:
            images = convert_from_bytes(pdf_path.read_bytes(), dpi=200)
            for im in images:
                pages_txt.append(pytesseract.image_to_string(im))
        except Exception:
            return ""
    return "\n\n".join(pages_txt)

# -------------------------
# Indexing & querying
# -------------------------
def index_uploaded_files(uploaded_files, hf_facade: HFClientFacade, embed_models: List[str], max_chunks: Optional[int] = None):
    if not uploaded_files:
        return None
    fp = fingerprint_files(uploaded_files)
    persist_dir = TMP_DIR / fp
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chroma_client(str(persist_dir))
    col = ensure_collection(client, f"col_{fp}")

    # if already indexed, return quickly
    try:
        if getattr(col, "count", lambda: 0)() > 0:
            return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": col.count(), "snippets": []}
    except Exception:
        pass

    texts, ids, metadatas, snippets = [], [], [], []
    for uploaded in uploaded_files:
        # size guard
        try:
            size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
        except Exception:
            size = 0
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipping {uploaded.name} (> {MAX_UPLOAD_MB}MB)")
            continue

        path = persist_dir / uploaded.name
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        # extract text
        txt = ""
        if uploaded.name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(str(path))
                pages = [p.extract_text() or "" for p in reader.pages]
                txt = "\n\n".join(pages).strip()
            except Exception:
                txt = ""
            if (not txt or not txt.strip()) and OCR_AVAILABLE:
                txt = ocr_pdf_to_text_bytes(path)
        else:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    txt = path.read_text(encoding="latin-1", errors="ignore")
                except Exception:
                    txt = ""

        if not txt or not txt.strip():
            st.info(f"No text extracted from {uploaded.name}; skipping.")
            continue

        chunks = chunk_text(txt)
        for i, c in enumerate(chunks):
            uid = f"{uploaded.name}__{i}"
            ids.append(uid)
            texts.append(c)
            metadatas.append({"source": uploaded.name, "chunk": i})
            snippets.append({"name": uploaded.name, "preview": c[:300]})

    if not texts:
        st.warning("No text to index.")
        return None

    if max_chunks and len(texts) > max_chunks:
        texts = texts[:max_chunks]; ids = ids[:max_chunks]; metadatas = metadatas[:max_chunks]

    # compute embeddings (try candidate models in order; stop at first that works)
    embeddings = None
    used_model = None
    for m in embed_models:
        try:
            st.info(f"Trying embeddings with HF model: {m}")
            embeddings = hf_facade.embeddings(m, texts)
            used_model = m
            break
        except Exception as e:
            st.info(f"Model {m} failed for embeddings: {e}")
            continue

    if not embeddings:
        st.error("All embedding models failed. See logs for details.")
        return None

    # add to chroma
    try:
        col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        # persist if supported
        try:
            client.persist()
        except Exception:
            pass
    except Exception as e:
        st.error(f"Failed to add to Chroma: {e}")
        return None

    return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": len(texts), "snippets": snippets, "embed_model": used_model}

def query_collection(collection, hf_facade: HFClientFacade, embed_models: List[str], query: str, k: int = 3):
    if not collection:
        return []
    # try to get query embedding with the same models order
    qv = None
    used_model = None
    for m in embed_models:
        try:
            qv = hf_facade.embeddings(m, [query])[0]
            used_model = m
            break
        except Exception:
            continue
    if qv is None:
        st.error("Failed to compute query embedding.")
        return []
    try:
        res = collection.query(query_embeddings=[qv], n_results=k, include=["documents","metadatas","distances"])
    except Exception as e:
        st.error(f"Chroma query error: {e}")
        return []
    docs_list = res.get("documents", [[]])[0]
    metas_list = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for doc, meta, dist in zip(docs_list, metas_list, dists):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out

# -------------------------
# Sidebar + UI
# -------------------------
with st.sidebar:
    st.header("Settings")
    st.markdown("**Embeddings candidate models (tried in order)**")
    models_txt = st.text_area("HF embed models (comma-separated)", value=",".join(DEFAULT_EMBED_MODELS), height=80)
    embed_models = [m.strip() for m in models_txt.split(",") if m.strip()]
    st.markdown("**Image caption candidate models (tried in order)**")
    img_txt = st.text_area("HF image caption models (comma-separated)", value=",".join(DEFAULT_IMAGE_MODELS), height=80)
    image_models = [m.strip() for m in img_txt.split(",") if m.strip()]
    uploaded_files = st.file_uploader("Upload PDF/TXT (multiple)", type=["pdf","txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Upload image (optional)", type=["png","jpg","jpeg"])
    if st.button("Clear conversation & index"):
        for k in ["messages", "chroma_info", "last_upload_fp"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

if "messages" not in st.session_state: st.session_state["messages"] = []
if "chroma_info" not in st.session_state: st.session_state["chroma_info"] = None

st.title("RAG — Resilient HF InferenceClient (fix 410)")

# init facade
hf_token = get_hf_token()
hf_facade = HFClientFacade(hf_token)

# Auto-index uploaded files once per fingerprint
if uploaded_files and st.session_state.get("chroma_info") is None:
    try:
        fp = fingerprint_files(uploaded_files)
        if st.session_state.get("last_upload_fp") != fp:
            st.session_state["last_upload_fp"] = fp
            with st.spinner("Auto-indexing uploaded files (this may take some time)..."):
                info = index_uploaded_files(uploaded_files, hf_facade=hf_facade, embed_models=embed_models, max_chunks=None)
                if info:
                    st.session_state["chroma_info"] = info
                    st.success(f"Indexed {info.get('indexed_chunks')} chunks using model {info.get('embed_model')}")
                else:
                    st.error("Auto-indexing failed; check logs and HF token / model availability.")
    except Exception as e:
        st.error(f"Auto-indexing exception: {e}")

# Main UI columns
col1, col2 = st.columns([3,1])
with col1:
    for msg in st.session_state["messages"]:
        role = msg.get("role","assistant"); content = msg.get("content","")
        if role == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:8px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:8px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

    # Image caption block
    if image_file:
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask something about this image (optional):")
        if q_img and st.button("Analyze image"):
            img_bytes = image_file.getvalue()
            caption = None
            # try candidate image models
            for imodel in image_models:
                try:
                    st.info(f"Trying image caption model: {imodel}")
                    caption = hf_facade.image_caption(imodel, img_bytes)
                    if caption:
                        st.success(f"Caption from {imodel}")
                        break
                except Exception as e:
                    st.info(f"Model {imodel} failed: {e}")
                    continue
            if not caption:
                caption = "Image caption unavailable (tried HF InferenceClient + HTTP endpoints)."
            prompt = f"Image caption: {caption}\n\nQuestion: {q_img}\nAnswer concisely."
            # generate short answer via HF text model (use InferenceClient text model or simple static message)
            answer = None
            if hf_facade.hf_client:
                try:
                    gen = hf_facade.hf_client(model="google/flan-t5-large", inputs=prompt)
                    # attempt to extract generated_text
                    if isinstance(gen, dict) and "generated_text" in gen:
                        answer = gen["generated_text"]
                    elif isinstance(gen, list) and gen and isinstance(gen[0], dict) and "generated_text" in gen[0]:
                        answer = gen[0]["generated_text"]
                    elif isinstance(gen, str):
                        answer = gen
                except Exception:
                    answer = None
            if not answer:
                # fallback static
                answer = f"[Answer generated locally] Based on caption: {caption}\n\nQ: {q_img}\nA: (short answer)"
            st.session_state["messages"].append({"role":"assistant","content": answer})
            st.rerun()

    # Text chat form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        send = st.form_submit_button("Send")

    if send and user_input and user_input.strip():
        st.session_state["messages"].append({"role":"user","content": user_input})
        with st.spinner("Thinking..."):
            # If we have chroma index, run RAG; else do a simple reply
            chroma_info = st.session_state.get("chroma_info")
            reply = ""
            if chroma_info and chroma_info.get("collection"):
                docs = query_collection(chroma_info["collection"], hf_facade=hf_facade, embed_models=embed_models, query=user_input, k=3)
                ctx = "\n\n---\n\n".join([d["document"][:800] for d in docs])
                prompt = f"Context:\n{ctx}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
                # try InferenceClient to generate answer
                if hf_facade.hf_client:
                    try:
                        gen = hf_facade.hf_client(model="google/flan-t5-large", inputs=prompt)
                        if isinstance(gen, dict) and "generated_text" in gen:
                            reply = gen["generated_text"]
                        elif isinstance(gen, list) and gen and isinstance(gen[0], dict) and "generated_text" in gen[0]:
                            reply = gen[0]["generated_text"]
                        elif isinstance(gen, str):
                            reply = gen
                    except Exception:
                        reply = "Unable to generate a text reply using HF Inference. Please try again."
                else:
                    reply = "No HF Inference client available to generate answers. Please set HF_API_TOKEN or use a different deployment."
            else:
                reply = "No RAG index loaded. Upload PDFs/TXT in the sidebar to enable RAG."
            st.session_state["messages"].append({"role":"assistant","content": reply})
            st.rerun()

with col2:
    st.markdown("### Status")
    if st.session_state.get("chroma_info"):
        info = st.session_state["chroma_info"]
        st.success("Chroma index ready")
        st.write({"persist_dir": info.get("persist_dir"), "indexed_chunks": info.get("indexed_chunks"), "embed_model": info.get("embed_model")})
        if info.get("snippets"):
            st.markdown("**Document previews**")
            for s in info["snippets"][:6]:
                st.markdown(f"- **{s['name']}** `{s['preview'][:120]}...`")
    else:
        st.info("No Chroma index. Upload files and allow auto-indexing.")

    st.markdown("---")
    if st.button("Index uploaded files (manual)"):
        if uploaded_files:
            with st.spinner("Indexing..."):
                info = index_uploaded_files(uploaded_files, hf_facade=hf_facade, embed_models=embed_models, max_chunks=None)
                if info:
                    st.session_state["chroma_info"] = info
                    st.success("Indexing complete.")
                else:
                    st.error("Indexing failed. Check HF token & model availability.")
        else:
            st.info("No uploaded files to index.")

    st.markdown("---")
    if st.button("Show memory"):
        st.write(st.session_state.get("messages", [])[-8:])

    st.caption("Set HF_API_TOKEN in Streamlit Secrets or environment variables for reliable HF InferenceClient access. If HF Inference endpoints/model IDs return 410, try different candidate models (sidebar).")
