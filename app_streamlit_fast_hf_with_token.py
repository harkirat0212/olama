# app_streamlit_resilient.py
"""
Resilient Streamlit RAG app:
- Robust HF embeddings & image captioning (tries InferenceClient -> HTTP endpoints -> local fallback)
- Chroma vectorstore with robust constructor (persistent attempt, in-memory fallback)
- PDF text extraction (pypdf) + OCR fallback (pdf2image + pytesseract) if available
- @st.cache_resource for HF clients and local models to avoid repeated downloads
- Clear UI diagnostic messages about which backend/model succeeded
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

# Optional deps (handled gracefully)
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Optional local embedding model
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_S2_AVAILABLE = True
except Exception:
    LOCAL_S2_AVAILABLE = False

# Optional huggingface_hub client
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# -------------------------
# FALLBACK TOKENS (you provided)
# Replace or remove fallback tokens before publishing.
GROQ_KEY_FALLBACK = "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"
HF_KEY_FALLBACK = "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"
# -------------------------

# -------------------------
# DEFAULT MODELS & CONFIG
# -------------------------
DEFAULT_HF_EMBED_CANDIDATES = [
    "intfloat/e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
DEFAULT_HF_IMAGE_CANDIDATES = [
    "Salesforce/blip-image-captioning-large",
    "nlpconnect/vit-gpt2-image-captioning",
]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
TMP_DIR = Path(os.environ.get("STREAMLIT_CHROMA_DIR", tempfile.gettempdir())) / "chroma_persist_v_final"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOADS = 6
MAX_UPLOAD_MB = 200
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 16
HF_TIMEOUT = 60

st.set_page_config(page_title="RAG — resilient HF + Chroma", layout="wide")

# -------------------------
# Utilities: tokens
# -------------------------
def get_hf_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if token and token.strip():
        return token.strip()
    try:
        token = st.secrets.get("HF_API_TOKEN")  # recommended
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
# Cached clients & models
# -------------------------
@st.cache_resource
def get_hf_inference_client():
    """Return a huggingface_hub.InferenceClient if possible, else None."""
    if not HF_HUB_AVAILABLE:
        return None
    token = get_hf_token()
    if not token:
        return None
    try:
        return InferenceClient(token=token)
    except Exception:
        return None

@st.cache_resource
def get_local_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Load & cache a sentence-transformers model for local embeddings."""
    if not LOCAL_S2_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed.")
    # Accept either repo id or short name
    try:
        return SentenceTransformer(model_name)
    except Exception:
        # attempt with HF repo id
        return SentenceTransformer(f"sentence-transformers/{model_name}")

# -------------------------
# Robust HF embeddings + image caption helpers
# -------------------------
class HFResilient:
    """Try InferenceClient -> HTTP endpoints -> local fallback for embeddings & image captions."""

    def __init__(self, embed_models: List[str], image_models: List[str], local_fallback: bool = True, timeout: int = HF_TIMEOUT):
        self.embed_models = embed_models
        self.image_models = image_models
        self.local_fallback = local_fallback
        self.timeout = timeout
        self.hf_token = get_hf_token()
        self.hf_client = get_hf_inference_client()
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

    # --- embeddings ---
    def _http_embeddings(self, model: str, texts: List[str]) -> Optional[List[List[float]]]:
        endpoints = [
            f"https://api-inference.huggingface.co/embeddings/{model}",
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}",
            f"https://api-inference.huggingface.co/models/{model}/pipeline/feature-extraction",
        ]
        payload = {"inputs": texts}
        for url in endpoints:
            try:
                r = requests.post(url, headers={**self.headers, "Content-Type":"application/json"}, json=payload, timeout=self.timeout)
                if r.status_code in (404, 410, 403):
                    # unsupported route/model -> try next endpoint/model
                    continue
                r.raise_for_status()
                data = r.json()
                # Normalize list of list result
                if isinstance(data, list) and data and isinstance(data[0], list):
                    return data
                # some endpoints return list of dicts with 'embedding' key
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    out = []
                    for item in data:
                        if "embedding" in item:
                            out.append(item["embedding"])
                        elif "vector" in item:
                            out.append(item["vector"])
                        else:
                            # try to find first list of floats in values
                            found = None
                            for v in item.values():
                                if isinstance(v, list) and all(isinstance(x, (float,int)) for x in v):
                                    found = v; break
                            if found:
                                out.append(found)
                            else:
                                out = None
                                break
                    if out:
                        return out
                # otherwise continue trying
            except requests.HTTPError as he:
                status = getattr(he.response, "status_code", None)
                if status in (404, 410, 403):
                    continue
                raise
            except Exception:
                continue
        return None

    def _hf_client_embeddings(self, model: str, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.hf_client:
            return None
        try:
            # InferenceClient has embeddings(...) method in modern versions
            out = self.hf_client.embeddings(model=model, inputs=texts)
            # out expected to be list-of-vectors
            if isinstance(out, list) and out and isinstance(out[0], list):
                return out
            # fallback: call model pipeline
            resp = self.hf_client(texts, model=model)
            if isinstance(resp, list) and resp and isinstance(resp[0], list):
                return resp
        except Exception:
            return None
        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Try hf client first per model
        for model in self.embed_models:
            if self.hf_client:
                try:
                    out = self._hf_client_embeddings(model, texts)
                    if out:
                        st.info(f"Embeddings: HF InferenceClient success with model: {model}")
                        return out
                except Exception as e:
                    st.info(f"HF client embeddings attempt failed for {model}: {e}")

            # Try HTTP endpoints
            try:
                out = self._http_embeddings(model, texts)
                if out:
                    st.info(f"Embeddings: HF HTTP endpoint success with model: {model}")
                    return out
            except Exception as e:
                st.info(f"HF HTTP embeddings attempt failed for {model}: {e}")
                continue

        # Local fallback
        if self.local_fallback:
            if LOCAL_S2_AVAILABLE:
                st.info("Embeddings: falling back to local sentence-transformers (may download on first run).")
                model_name = self.embed_models[0] if self.embed_models else "all-MiniLM-L6-v2"
                # try robust local model loader
                try:
                    m = get_local_sentence_transformer(model_name if "all-" not in model_name else model_name)
                except Exception:
                    # fallback to a safe local id
                    m = get_local_sentence_transformer("all-MiniLM-L6-v2")
                arr = m.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE, convert_to_numpy=True)
                return [v.tolist() for v in arr]
            else:
                raise RuntimeError("HF embedding endpoints failed and local sentence-transformers is not installed.")
        raise RuntimeError("All embedding backends failed (HF endpoints + local fallback).")

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        if not res or not isinstance(res, list):
            raise RuntimeError("Failed to compute query embedding")
        return res[0]

    # --- image caption ---
    def _http_image_caption(self, model: str, image_bytes: bytes) -> Optional[str]:
        url = f"https://api-inference.huggingface.co/models/{model}"
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        for attempt in range(1):
            try:
                r = requests.post(url, headers=self.headers, files=files, timeout=self.timeout)
                if r.status_code in (404, 410, 403):
                    return None
                r.raise_for_status()
                out = r.json()
                # many HF image caption endpoints return list with dict->generated_text
                if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                    return out[0]["generated_text"]
                if isinstance(out, dict) and "generated_text" in out:
                    return out["generated_text"]
                # some return 'caption' or 'result'
                if isinstance(out, dict):
                    for k in ("caption","result","text"):
                        if k in out and isinstance(out[k], str):
                            return out[k]
                return str(out)
            except requests.HTTPError as he:
                status = getattr(he.response, "status_code", None)
                if status in (404,410,403):
                    return None
                continue
            except Exception:
                continue
        return None

    def _hf_client_image_caption(self, model: str, image_bytes: bytes) -> Optional[str]:
        if not self.hf_client:
            return None
        try:
            # InferenceClient usually supports passing files as bytes
            res = self.hf_client(inputs=image_bytes, model=model)
            # normalize
            if isinstance(res, dict) and "generated_text" in res:
                return res["generated_text"]
            if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
                return res[0]["generated_text"]
            return None
        except Exception:
            return None

    def caption_image(self, image_bytes: bytes) -> str:
        for model in self.image_models:
            # hf client
            if self.hf_client:
                try:
                    out = self._hf_client_image_caption(model, image_bytes)
                    if out:
                        st.info(f"Image caption: HF InferenceClient success with model {model}")
                        return out
                except Exception as e:
                    st.info(f"HF client image caption error for {model}: {e}")
            # http endpoint
            try:
                out = self._http_image_caption(model, image_bytes)
                if out:
                    st.info(f"Image caption: HF HTTP endpoint success with model {model}")
                    return out
            except Exception as e:
                st.info(f"HF HTTP image caption error for {model}: {e}")
                continue

        # Local fallback: very basic (if no HF available) - attempt to use transformers' pipeline if installed
        try:
            from transformers import pipeline
            # try a captioning pipeline with a known model - may download heavy model
            try_models = ["nlpconnect/vit-gpt2-image-captioning", "Salesforce/blip-image-captioning-base"]
            for m in try_models:
                try:
                    pipe = pipeline("image-to-text", model=m, device=-1)
                    out = pipe(image_bytes)
                    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                        return out[0]["generated_text"]
                    # pipeline may return list of strings
                    if isinstance(out, list) and out and isinstance(out[0], str):
                        return out[0]
                except Exception:
                    continue
        except Exception:
            pass

        raise RuntimeError("All image caption backends failed (HF endpoints + local fallback).")

# -------------------------
# Chroma helpers (robust)
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
    # fallback to in-memory
    try:
        st.warning("Persistent Chroma init failed; falling back to in-memory Chroma (no persistence).")
        client = chromadb.Client()
        return client
    except Exception as e:
        err = f"Could not initialize Chroma client. Tried: {tried}. Last error: {e}"
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
# File helpers: fingerprint, chunking, OCR
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
# Indexing & querying using HFResilient
# -------------------------
def index_uploaded_files(uploaded_files, hf_resilient: HFResilient, chunk_size: int, chunk_overlap: int, max_chunks: Optional[int]):
    if not uploaded_files:
        return None
    fp = fingerprint_files(uploaded_files)
    persist_dir = str(TMP_DIR / fp)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    try:
        client = chroma_client(persist_dir)
    except Exception as e:
        st.error(f"Chroma init error: {e}")
        return None

    try:
        col = ensure_collection(client, f"col_{fp}")
    except Exception as e:
        st.error(f"Chroma collection error: {e}")
        return None

    # If already indexed, skip
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
            st.warning(f"Skipping {uploaded.name}: >{MAX_UPLOAD_MB}MB")
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

    # compute embeddings
    try:
        embeddings = []
        progress = st.progress(0)
        total = len(texts)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            ev = hf_resilient.embed_documents(batch)
            embeddings.extend(ev)
            progress.progress(min(100, int(100 * (i + len(batch)) / total)))
        progress.empty()
    except Exception as e:
        st.error(f"Embedding step failed: {e}")
        return None

    try:
        col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        client.persist()
    except Exception as e:
        st.error(f"Failed to save to Chroma: {e}")
        return None

    return {"collection": col, "persist_dir": persist_dir, "indexed_chunks": len(texts), "snippets": snippets}

def query_collection(collection: chromadb.api.models.Collection, hf_resilient: HFResilient, query: str, k: int = 3):
    if collection is None:
        return []
    try:
        qv = hf_resilient.embed_query(query)
    except Exception as e:
        st.error(f"Query embedding error: {e}")
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
# UI
# -------------------------
with st.sidebar:
    st.header("Settings")
    st.markdown("**Embeddings candidate models (tried in order)**")
    embed_models_txt = st.text_area("HF embedding models (comma separated)", value=",".join(DEFAULT_HF_EMBED_CANDIDATES), height=80)
    embed_models = [m.strip() for m in embed_models_txt.split(",") if m.strip()]
    st.markdown("**Image caption candidate models (tried in order)**")
    image_models_txt = st.text_area("HF image models (comma separated)", value=",".join(DEFAULT_HF_IMAGE_CANDIDATES), height=60)
    image_models = [m.strip() for m in image_models_txt.split(",") if m.strip()]
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload PDF/TXT (multiple)", type=["pdf","txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Upload image (optional)", type=["png","jpg","jpeg"])
    chunk_size = st.number_input("Chunk size", value=CHUNK_SIZE, min_value=256)
    chunk_overlap = st.number_input("Chunk overlap", value=CHUNK_OVERLAP, min_value=0)
    max_chunks = st.number_input("Max chunks to index (0=no limit)", value=0, min_value=0)
    memory_window = st.slider("Memory window", 1, 10, 4)
    st.markdown("---")
    if st.button("Clear conversation & index"):
        for k in ["messages","chroma_info","last_upload_fp"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chroma_info" not in st.session_state:
    st.session_state["chroma_info"] = None

# Build HFResilient helper once (cached)
hf_res = HFResilient(embed_models=embed_models, image_models=image_models, local_fallback=True)

st.title("RAG Chat — resilient HF + Chroma")

# Auto-index uploaded files (once per fingerprint)
if uploaded_files and st.session_state.get("chroma_info") is None:
    try:
        uploaded_fp = fingerprint_files(uploaded_files)
        prev_fp = st.session_state.get("last_upload_fp")
        if uploaded_fp != prev_fp:
            st.session_state["last_upload_fp"] = uploaded_fp
            with st.spinner("Auto-indexing uploaded files..."):
                info = index_uploaded_files(uploaded_files=uploaded_files, hf_resilient=hf_res, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), max_chunks=(None if int(max_chunks) == 0 else int(max_chunks)))
                if info:
                    st.session_state["chroma_info"] = info
                    st.success(f"Auto-indexed {info.get('indexed_chunks')} chunks.")
                else:
                    st.error("Auto-indexing failed; check logs or HF/Chroma initialization.")
    except Exception as e:
        st.error(f"Auto-indexing exception: {e}")

col1, col2 = st.columns([3,1])
with col1:
    for msg in st.session_state["messages"]:
        role = msg.get("role","assistant")
        content = msg.get("content","")
        if role == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

    # Image analyze
    if image_file:
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask something about this image (optional):")
        if q_img and st.button("Analyze image"):
            img_bytes = image_file.getvalue()
            with st.spinner("Captioning image..."):
                try:
                    caption = hf_res.caption_image(img_bytes)
                except Exception as e:
                    caption = f"[Caption error] {e}"
            # Build prompt with caption
            prompt = f"Image caption: {caption}\n\nQuestion: {q_img}\nAnswer concisely."
            # Use Groq if available or HF text-model fallback
            groq_key = get_groq_key()
            if groq_key:
                try:
                    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                             headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
                                             json={"model": DEFAULT_GROQ_MODEL, "messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}], "max_tokens":512},
                                             timeout=HF_TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                    choice = data.get("choices",[{}])[0]
                    out_text = choice.get("message",{}).get("content") if isinstance(choice.get("message"),dict) else choice.get("text")
                    answer = out_text or str(data)
                except Exception as e:
                    # HF text fallback
                    try:
                        hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                        r = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                        r.raise_for_status()
                        out = r.json()
                        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                            answer = out[0]["generated_text"]
                        else:
                            answer = str(out)
                    except Exception as ee:
                        answer = f"[Generation error] {ee}"
            else:
                # HF text fallback
                try:
                    hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                    r = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                    r.raise_for_status()
                    out = r.json()
                    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                        answer = out[0]["generated_text"]
                    else:
                        answer = str(out)
                except Exception as ee:
                    answer = f"[Generation error] {ee}"

            st.session_state["messages"].append({"role":"assistant","content": answer})
            st.experimental_rerun()

    # Text chat
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        send = st.form_submit_button("Send")

    if send and user_input and user_input.strip():
        st.session_state["messages"].append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            try:
                chroma_info = st.session_state.get("chroma_info")
                final_answer = ""
                if chroma_info and chroma_info.get("collection"):
                    collection = chroma_info["collection"]
                    docs = query_collection(collection, hf_res, user_input, k=3)
                    retrieved_texts = [d.get("document","")[:1000] for d in docs]
                    context = "\n\n---\n\n".join(retrieved_texts)
                    mem = st.session_state.get("messages", [])[-memory_window:]
                    mem_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in mem])
                    prompt = f"Memory:\n{mem_text}\n\nRetrieved:\n{context}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
                    groq_key = get_groq_key()
                    if groq_key:
                        try:
                            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                              headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
                                              json={"model": DEFAULT_GROQ_MODEL, "messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}], "max_tokens":512},
                                              timeout=HF_TIMEOUT)
                            r.raise_for_status()
                            data = r.json()
                            choice = data.get("choices",[{}])[0]
                            out_text = choice.get("message",{}).get("content") if isinstance(choice.get("message"),dict) else choice.get("text")
                            final_answer = out_text or str(data)
                        except Exception as e:
                            # HF text fallback
                            try:
                                hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                                rr = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                                rr.raise_for_status()
                                out = rr.json()
                                if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                    final_answer = out[0]["generated_text"]
                                else:
                                    final_answer = str(out)
                            except Exception as ee:
                                final_answer = f"[Generation error] {ee}"
                    else:
                        try:
                            hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                            rr = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                            rr.raise_for_status()
                            out = rr.json()
                            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                final_answer = out[0]["generated_text"]
                            else:
                                final_answer = str(out)
                        except Exception as ee:
                            final_answer = f"[Generation error] {ee}"
                else:
                    # No RAG; simple generation
                    prompt = f"Question:\n{user_input}\n\nAnswer concisely."
                    groq_key = get_groq_key()
                    if groq_key:
                        try:
                            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                              headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
                                              json={"model": DEFAULT_GROQ_MODEL, "messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}], "max_tokens":512},
                                              timeout=HF_TIMEOUT)
                            r.raise_for_status()
                            data = r.json()
                            choice = data.get("choices",[{}])[0]
                            out_text = choice.get("message",{}).get("content") if isinstance(choice.get("message"),dict) else choice.get("text")
                            final_answer = out_text or str(data)
                        except Exception as e:
                            try:
                                hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                                rr = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                                rr.raise_for_status()
                                out = rr.json()
                                if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                    final_answer = out[0]["generated_text"]
                                else:
                                    final_answer = str(out)
                            except Exception as ee:
                                final_answer = f"[Generation error] {ee}"
                    else:
                        try:
                            hf_text_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
                            rr = requests.post(hf_text_url, headers={"Authorization": f"Bearer {get_hf_token()}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                            rr.raise_for_status()
                            out = rr.json()
                            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                final_answer = out[0]["generated_text"]
                            else:
                                final_answer = str(out)
                        except Exception as ee:
                            final_answer = f"[Generation error] {ee}"
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
        st.info("No Chroma index. Upload PDFs/TXT and the app will auto-index them.")

    st.markdown("---")
    if uploaded_files and st.button("Index uploaded files (manual)"):
        with st.spinner("Indexing..."):
            info = index_uploaded_files(uploaded_files=uploaded_files, hf_resilient=hf_res, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), max_chunks=(None if int(max_chunks)==0 else int(max_chunks)))
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
    st.caption("Set HF_API_TOKEN and GROQ_API_KEY in Streamlit Secrets (recommended). OCR requires poppler + tesseract installed on the host to work.")

# End app
