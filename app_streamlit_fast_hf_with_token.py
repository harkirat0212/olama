# app_streamlit_groq_chroma_hf.py
"""
Streamlit RAG chatbot — updated resilient embeddings flow.

Features:
- Embeddings backend selector: "HF Inference API" (default) or "Local sentence-transformers"
- HF embeddings tries multiple candidate models and uses huggingface_hub.InferenceClient if available
- If HF fails, optionally fall back to local sentence-transformers (if installed)
- Robust Chroma client (tries multiple constructors, falls back to in-memory)
- OCR fallback for scanned PDFs using pdf2image + pytesseract if available
- Groq (OpenAI-compatible) for generation if GROQ key is present; HF text model fallback
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

# optional OCR libs
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# optional local embeddings
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_S2_AVAILABLE = True
except Exception:
    LOCAL_S2_AVAILABLE = False

# optional HF client
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# -------------------------
# USER KEYS (FALLBACKS)
# -------------------------
GROQ_KEY_FALLBACK = "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"
HF_KEY_FALLBACK = "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"
# -------------------------

# -------------------------
# CONFIG
# -------------------------
DEFAULT_HF_EMBED_MODELS = [
    "intfloat/e5-small",                      # often supported for HF inference embeddings
    "sentence-transformers/all-MiniLM-L6-v2", # widely used
    "sentence-transformers/all-mpnet-base-v2" # higher quality
]
DEFAULT_HF_IMAGE_MODEL = "Salesforce/blip-image-captioning-large"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

TMP_DIR = Path(os.environ.get("STREAMLIT_CHROMA_DIR", tempfile.gettempdir())) / "chroma_persist_v4"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOADS = 6
MAX_UPLOAD_MB = 200
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HF_BATCH_SIZE = 16
HF_TIMEOUT = 60

st.set_page_config(page_title="RAG — Resilient Embeddings + Chroma", layout="wide")

# -------------------------
# helpers: secrets
# -------------------------
def get_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if token and token.strip():
        return token.strip()
    try:
        token = st.secrets.get("HF_API_TOKEN")
        if token and token.strip():
            return token.strip()
    except Exception:
        pass
    return HF_KEY_FALLBACK

def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key and key.strip():
            return key.strip()
    except Exception:
        pass
    return GROQ_KEY_FALLBACK

# -------------------------
# Robust HF embeddings (tries many routes) + local fallback
# -------------------------
class HFAndLocalEmbeddings:
    """
    Attempts:
      1) huggingface_hub.InferenceClient.embeddings() (if huggingface_hub available)
      2) HF inference endpoints (embeddings / feature-extraction) for a list of candidate models
      3) Local sentence-transformers model (if installed/allowed)
    """

    def __init__(self, candidate_models: List[str], use_local_if_failed: bool = True, timeout: int = HF_TIMEOUT):
        self.candidate_models = candidate_models
        self.timeout = timeout
        self.token = get_hf_token()
        self.use_local_if_failed = use_local_if_failed
        self.hf_client = None
        if HF_HUB_AVAILABLE:
            try:
                self.hf_client = InferenceClient(token=self.token)
            except Exception:
                self.hf_client = None

        # prepare HTTP endpoints templates
        self.endpoint_templates = [
            "https://api-inference.huggingface.co/embeddings/{model}",
            "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
            "https://api-inference.huggingface.co/models/{model}/pipeline/feature-extraction",
            "https://api-inference.huggingface.co/models/{model}"
        ]
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"} if self.token else {}

        # local model attr (lazy)
        self.local_model = None

    def _normalize_response(self, resp):
        if resp is None:
            return None
        if isinstance(resp, list) and resp and all(isinstance(x, list) for x in resp):
            return resp
        if isinstance(resp, list) and resp and all(isinstance(x, dict) for x in resp):
            out = []
            for item in resp:
                if "embedding" in item:
                    out.append(item["embedding"])
                elif "vector" in item:
                    out.append(item["vector"])
                else:
                    # some endpoints may return nested lists in dict; attempt extraction
                    # try to find first list-of-floats in dict values
                    found = None
                    for v in item.values():
                        if isinstance(v, list) and all(isinstance(x, float) or isinstance(x, int) for x in v):
                            found = v
                            break
                    if found:
                        out.append(found)
                    else:
                        return None
            return out
        if isinstance(resp, dict):
            # attempt to extract embeddings list
            for k in ("embedding","embeddings","vector","vectors"):
                if k in resp:
                    val = resp[k]
                    if isinstance(val, list) and all(isinstance(x, list) for x in val):
                        return val
            return None
        return None

    def _call_hf_client(self, model: str, inputs: Iterable[str]):
        if not self.hf_client:
            return None
        try:
            return self.hf_client.embeddings(model=model, inputs=list(inputs))
        except Exception:
            try:
                # fallback to direct call
                res = self.hf_client(list(inputs), model=model)
                return self._normalize_response(res)
            except Exception:
                return None

    def _http_try(self, model: str, inputs: Iterable[str]):
        for templ in self.endpoint_templates:
            url = templ.format(model=model)
            try:
                data = {"inputs": list(inputs)}
                r = requests.post(url, headers=self.headers, json=data, timeout=self.timeout)
                if r.status_code == 404 or r.status_code == 410 or r.status_code == 403:
                    # unsupported endpoint/model — try next
                    continue
                r.raise_for_status()
                jsonr = r.json()
                norm = self._normalize_response(jsonr)
                if norm:
                    return norm
                # if not normalized, continue to try next endpoint
            except requests.HTTPError as he:
                status = getattr(he.response, "status_code", None)
                if status in (404, 410, 403):
                    continue
                else:
                    # re-raise other HTTP errors so they are visible
                    raise
            except Exception:
                continue
        return None

    def _try_models_hf(self, inputs: Iterable[str]):
        # Try HF client first with each model
        for model in self.candidate_models:
            try:
                if self.hf_client:
                    out = self._call_hf_client(model, inputs)
                    if out:
                        st.info(f"HF InferenceClient succeeded with model {model}")
                        return out, model
            except Exception as e:
                # continue to next model
                st.info(f"HF client try failed for model {model}: {e}")

            # Try raw HTTP endpoints for that model
            try:
                out = self._http_try(model, inputs)
                if out:
                    st.info(f"HF HTTP endpoint succeeded with model {model}")
                    return out, model
            except Exception as e:
                st.info(f"HF HTTP try failed for model {model}: {e}")
                continue
        return None, None

    def _init_local_model(self):
        if not LOCAL_S2_AVAILABLE:
            raise RuntimeError("Local sentence-transformers not installed.")
        if self.local_model is None:
            # choose a small local model to reduce download size; fallback to all-MiniLM-L6-v2 if available
            try:
                self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                # try a more explicit HF model id
                self.local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.local_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # First try HF model(s)
        try:
            out, used_model = self._try_models_hf(texts)
            if out:
                return out
        except Exception as e:
            st.info(f"HF model attempts raised error: {e}")

        # HF failed: fallback to local if allowed
        if self.use_local_if_failed:
            if LOCAL_S2_AVAILABLE:
                st.info("Falling back to local sentence-transformers embeddings (this may download a model on first run).")
                try:
                    model = self._init_local_model()
                    vecs = model.encode(texts, show_progress_bar=False, batch_size=HF_BATCH_SIZE, convert_to_numpy=True)
                    return [v.tolist() for v in vecs]
                except Exception as e:
                    raise RuntimeError(f"Local sentence-transformers embedding failed: {e}")
            else:
                raise RuntimeError("HF inference failed and local sentence-transformers not available.")
        else:
            raise RuntimeError("HF inference failed and local fallback disabled.")

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        if not res or not isinstance(res, list):
            raise RuntimeError("Failed to produce query embedding.")
        return res[0]

# -------------------------
# Groq & HF image caption (unchanged)
# -------------------------
class HFImageCaption:
    def __init__(self, model: str = DEFAULT_HF_IMAGE_MODEL, timeout: int = HF_TIMEOUT):
        token = get_hf_token()
        if not token:
            raise RuntimeError("HF API token not set (HF_API_TOKEN).")
        self.url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout = timeout

    def caption(self, image_bytes: bytes) -> str:
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        r = requests.post(self.url, headers=self.headers, files=files, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return out[0]["generated_text"]
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        return str(out)

class GroqClient:
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = (api_key or get_groq_key())
        if not self.api_key:
            raise RuntimeError("GROQ API key not set (GROQ_API_KEY).")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def chat(self, messages: List[Dict[str,str]], model: str = DEFAULT_GROQ_MODEL, max_tokens: int = 512, temperature: float = 0.0) -> str:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        r = requests.post(self.ENDPOINT, headers=self.headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        res = r.json()
        try:
            choice = res.get("choices", [None])[0]
            if not choice:
                return str(res)
            if "message" in choice and isinstance(choice["message"], dict):
                return choice["message"].get("content","") or str(choice)
            return choice.get("text","") or str(choice)
        except Exception:
            return str(res)

# -------------------------
# File helpers, OCR, chunking (unchanged)
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

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
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
        images = convert_from_path(str(pdf_path), dpi=300)
        for im in images:
            pages_txt.append(pytesseract.image_to_string(im))
        return "\n\n".join(pages_txt)
    except Exception:
        try:
            data = pdf_path.read_bytes()
            images = convert_from_bytes(data, dpi=300)
            for im in images:
                pages_txt.append(pytesseract.image_to_string(im))
            return "\n\n".join(pages_txt)
        except Exception:
            return ""

# -------------------------
# Robust Chroma client + ensure_collection (unchanged)
# -------------------------
def chroma_client(persist_directory: str) -> chromadb.Client:
    persist_directory = str(persist_directory)
    tried = []
    candidates = [
        {"persist_directory": persist_directory, "chroma_api_impl": "duckdb+parquet"},
        {"persist_directory": persist_directory, "chroma_db_impl": "duckdb+parquet"},
        {"persist_directory": persist_directory, "chroma_api_impl": "duckdb"},
        {"persist_directory": persist_directory, "chroma_db_impl": "duckdb"},
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
        except Exception as e:
            tried.append((s, str(e)))
            continue
    try:
        st.warning("Persistent Chroma initialization failed; falling back to in-memory Chroma (no persistence).")
        client = chromadb.Client()
        return client
    except Exception as e:
        err = f"Could not initialize any Chroma client. Tried: {tried}. Last error: {e}"
        st.error(err)
        raise RuntimeError(err)

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
            raise RuntimeError(f"Could not create or get collection '{name}'.")

# -------------------------
# Index & query code (uses HFAndLocalEmbeddings)
# -------------------------
def index_uploaded_files(uploaded_files, embed_backend: str, candidate_models: List[str], hf_embed_model_choice: str, use_local_fallback: bool, chunk_size: int, chunk_overlap: int, max_chunks: Optional[int]):
    if not uploaded_files:
        return None
    fp = fingerprint_files(uploaded_files)
    persist_dir = str(TMP_DIR / fp)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    try:
        client = chroma_client(persist_dir)
    except Exception as e:
        st.error(f"Could not initialize Chroma client: {e}")
        return None

    try:
        col = ensure_collection(client, f"col_{fp}")
    except Exception as e:
        st.error(f"Could not ensure Chroma collection: {e}")
        return None

    try:
        cnt = col.count()
        if cnt and cnt > 0:
            return {"collection": col, "persist_dir": persist_dir, "indexed_chunks": cnt, "snippets": []}
    except Exception:
        pass

    texts, ids, metadatas, snippets = [], [], [], []
    for uploaded in uploaded_files[:MAX_UPLOADS]:
        try:
            size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
        except Exception:
            size = 0
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipping {uploaded.name} (> {MAX_UPLOAD_MB}MB)")
            continue

        raw_dir = Path(persist_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / uploaded.name
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())

        text = ""
        if uploaded.name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(str(target))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n\n".join(pages).strip()
            except Exception as e:
                st.info(f"pypdf error for {uploaded.name}: {e}")
                text = ""
            if (not text or not text.strip()) and OCR_AVAILABLE:
                st.info(f"No text via pypdf for {uploaded.name}; trying OCR...")
                try:
                    ocr_text = ocr_pdf_to_text_bytes(target)
                    if ocr_text and ocr_text.strip():
                        text = ocr_text
                except Exception as e:
                    st.info(f"OCR error for {uploaded.name}: {e}")
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

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
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
        st.info(f"Indexing first {max_chunks} chunks (out of {len(texts)})")
        texts = texts[:max_chunks]; ids = ids[:max_chunks]; metadatas = metadatas[:max_chunks]

    # prepare embeddings helper
    use_local = (embed_backend == "local") or (embed_backend == "auto" and use_local_fallback)
    hf_helper = HFAndLocalEmbeddings(candidate_models=candidate_models, use_local_if_failed=use_local)
    try:
        embeddings = []
        progress = st.progress(0)
        total = len(texts)
        for i in range(0, total, HF_BATCH_SIZE):
            batch = texts[i : i + HF_BATCH_SIZE]
            ev = hf_helper.embed_documents(batch)
            embeddings.extend(ev)
            progress.progress(min(100, int(100 * (i + len(batch)) / total)))
        progress.empty()
    except Exception as e:
        st.error(f"HF embedding error: {e}")
        return None

    try:
        col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        client.persist()
    except Exception as e:
        st.error(f"Failed to save to Chroma: {e}")
        return None

    return {"collection": col, "persist_dir": persist_dir, "indexed_chunks": len(texts), "snippets": snippets}

def query_collection(collection: chromadb.api.models.Collection, query: str, candidate_models: List[str], use_local_fallback: bool, k: int = 3):
    if collection is None:
        return []
    hf_helper = HFAndLocalEmbeddings(candidate_models=candidate_models, use_local_if_failed=use_local_fallback)
    try:
        qv = hf_helper.embed_query(query)
    except Exception as e:
        st.error(f"HF embedding error for query: {e}")
        return []
    try:
        res = collection.query(query_embeddings=[qv], n_results=k, include=["documents","metadatas","distances"])
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
# UI and flow (similar to earlier app)
# -------------------------
with st.sidebar:
    st.header("Settings")
    # Embedding backend control
    embed_backend = st.selectbox("Embeddings backend", ["hf_inference", "local", "auto"], index=2,
                                help="hf_inference = HuggingFace Inference API only; local = sentence-transformers locally; auto = try HF then local fallback")
    st.markdown("**HF candidate models (tried in order)**")
    # show editable list as comma-separated
    models_txt = st.text_area("HF candidate models (comma separated)", value=",".join(DEFAULT_HF_EMBED_MODELS), height=70)
    candidate_models = [m.strip() for m in models_txt.split(",") if m.strip()]
    hf_image_model = st.text_input("HF image caption model", value=DEFAULT_HF_IMAGE_MODEL)
    groq_model = st.text_input("Groq model", value=DEFAULT_GROQ_MODEL)
    chunk_size = st.number_input("Chunk size", value=CHUNK_SIZE, min_value=256)
    chunk_overlap = st.number_input("Chunk overlap", value=CHUNK_OVERLAP, min_value=0)
    max_chunks = st.number_input("Max chunks to index (0=no limit)", value=0, min_value=0)
    memory_window = st.slider("Memory window", 1, 10, 4)
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload PDF/TXT (multiple)", type=["pdf","txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Upload image (optional)", type=["png","jpg","jpeg"])
    st.markdown("---")
    if st.button("Clear conversation & index"):
        for k in ["messages","chroma_info","last_upload_fp"]:
            if k in st.session_state: del st.session_state[k]
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chroma_info" not in st.session_state:
    st.session_state["chroma_info"] = None

# instantiate Groq client if possible (non-fatal)
groq_client = None
try:
    groq_client = GroqClient()
except Exception:
    groq_client = None

st.title("RAG Chat — Resilient Embeddings + Chroma")

# Auto-index when uploads detected (single-run per fingerprint)
if uploaded_files and (st.session_state.get("chroma_info") is None):
    try:
        uploaded_fp = fingerprint_files(uploaded_files)
        prev_fp = st.session_state.get("last_upload_fp")
        if uploaded_fp != prev_fp:
            st.session_state["last_upload_fp"] = uploaded_fp
            with st.spinner("Auto-indexing uploaded files..."):
                info = index_uploaded_files(
                    uploaded_files=uploaded_files,
                    embed_backend=embed_backend,
                    candidate_models=candidate_models,
                    hf_embed_model_choice=candidate_models[0] if candidate_models else DEFAULT_HF_EMBED_MODELS[0],
                    use_local_fallback=(embed_backend != "hf_inference"),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    max_chunks=(None if int(max_chunks) == 0 else int(max_chunks)),
                )
                if info:
                    st.session_state["chroma_info"] = info
                    st.success(f"Auto-indexed {info.get('indexed_chunks')} chunks.")
                else:
                    st.error("Auto-indexing failed; check logs or HF/Chroma initialization.")
    except Exception as e:
        st.error(f"Auto-indexing exception: {e}")

# Main UI
col1, col2 = st.columns([3,1])
with col1:
    for msg in st.session_state["messages"]:
        role = msg.get("role","assistant")
        content = msg.get("content","")
        if role == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

    # image analyze
    if image_file:
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask something about this image (optional):")
        if q_img and st.button("Analyze image"):
            img_bytes = image_file.getvalue()
            with st.spinner("Captioning image (HF)..."):
                try:
                    hf_img = HFImageCaption(model=hf_image_model)
                    caption = hf_img.caption(img_bytes)
                except Exception as e:
                    caption = f"[Image caption error] {e}"
            prompt = f"Image caption: {caption}\n\nQuestion: {q_img}\nAnswer concisely."
            if groq_client:
                try:
                    messages = [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}]
                    answer = groq_client.chat(messages=messages, model=groq_model, max_tokens=512, temperature=0.0)
                except Exception as e:
                    answer = f"[Groq error] {e}"
            else:
                try:
                    hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                    hf_headers = {"Authorization": f"Bearer {get_hf_token()}"}
                    resp = requests.post(hf_text_url, headers=hf_headers, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                    resp.raise_for_status()
                    out = resp.json()
                    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                        answer = out[0]["generated_text"]
                    else:
                        answer = str(out)
                except Exception as e:
                    answer = f"[HF fallback error] {e}"
            st.session_state["messages"].append({"role":"assistant","content":answer})
            st.experimental_rerun()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        send = st.form_submit_button("Send")

    if send and user_input and user_input.strip():
        st.session_state["messages"].append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            try:
                memory_window = memory_window if "memory_window" in locals() else 4
                mem_items = st.session_state.get("messages", [])[-memory_window:]
                mem_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in mem_items])
                chroma_info = st.session_state.get("chroma_info")
                final_answer = ""
                if chroma_info and chroma_info.get("collection"):
                    collection = chroma_info["collection"]
                    docs = query_collection(collection, user_input, candidate_models=candidate_models, use_local_fallback=(embed_backend != "hf_inference"), k=3)
                    retrieved_texts = [d.get("document","")[:1000] for d in docs]
                    context = "\n\n---\n\n".join(retrieved_texts)
                    system_prompt = "You are a helpful assistant. Use retrieved documents to answer concisely and cite sources."
                    user_prompt = f"Memory:\n{mem_text}\n\nRetrieved:\n{context}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
                    if groq_client:
                        messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
                        try:
                            final_answer = groq_client.chat(messages=messages, model=groq_model, max_tokens=512, temperature=0.0)
                        except Exception as e:
                            final_answer = f"[Groq error] {e}"
                    else:
                        try:
                            hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                            hf_headers = {"Authorization": f"Bearer {get_hf_token()}"}
                            resp = requests.post(hf_text_url, headers=hf_headers, json={"inputs": user_prompt}, timeout=HF_TIMEOUT)
                            resp.raise_for_status()
                            out = resp.json()
                            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                final_answer = out[0]["generated_text"]
                            else:
                                final_answer = str(out)
                        except Exception as e:
                            final_answer = f"[HF fallback error] {e}"
                else:
                    prompt = f"Question:\n{user_input}\n\nAnswer concisely."
                    if groq_client:
                        messages = [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}]
                        try:
                            final_answer = groq_client.chat(messages=messages, model=groq_model, max_tokens=512, temperature=0.0)
                        except Exception as e:
                            final_answer = f"[Groq error] {e}"
                    else:
                        try:
                            hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                            hf_headers = {"Authorization": f"Bearer {get_hf_token()}"}
                            resp = requests.post(hf_text_url, headers=hf_headers, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                            resp.raise_for_status()
                            out = resp.json()
                            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                final_answer = out[0]["generated_text"]
                            else:
                                final_answer = str(out)
                        except Exception as e:
                            final_answer = f"[HF fallback error] {e}"
            except Exception as e:
                final_answer = f"[Unhandled error] {e}"

        st.session_state["messages"].append({"role":"assistant","content": final_answer})
        st.experimental_rerun()

with col2:
    st.markdown("### RAG & Status")
    chroma_info = st.session_state.get("chroma_info")
    if chroma_info:
        st.success("Chroma index loaded")
        st.write({"persist_dir": chroma_info.get("persist_dir"), "indexed_chunks": chroma_info.get("indexed_chunks")})
        if chroma_info.get("snippets"):
            st.markdown("**Document previews:**")
            for s in chroma_info["snippets"][:6]:
                st.markdown(f"- **{s['name']}** — `{s['preview'][:160]}...`")
    else:
        st.info("No Chroma index. Upload files and the app will auto-index them (or click 'Index uploaded files').")

    st.markdown("---")
    if uploaded_files and st.button("Index uploaded files (HF embeddings -> Chroma)"):
        with st.spinner("Indexing..."):
            info = index_uploaded_files(uploaded_files=uploaded_files, embed_backend=embed_backend, candidate_models=candidate_models, hf_embed_model_choice=candidate_models[0] if candidate_models else DEFAULT_HF_EMBED_MODELS[0], use_local_fallback=(embed_backend != "hf_inference"), chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), max_chunks=(None if int(max_chunks)==0 else int(max_chunks)))
            if info:
                st.session_state["chroma_info"] = info
                st.success(f"Indexed {info.get('indexed_chunks')} chunks.")
            else:
                st.error("Indexing failed.")

    st.markdown("---")
    if st.button("Show memory"):
        mem = st.session_state.get("messages", [])[-memory_window:]
        st.write(mem)

    st.markdown("---")
    st.caption("Set HF_API_TOKEN and GROQ_API_KEY in Streamlit Secrets or environment variables (recommended). Installing sentence-transformers allows local embedding fallback (may download model on first run).")

# End file
