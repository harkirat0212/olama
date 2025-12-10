# # app_streamlit_final_v2.py
# """
# Updated final Streamlit app with more robust embedding backend logic.
# - Prioritizes local sentence-transformers if available.
# - Then tries Hugging Face InferenceClient.embeddings (requires HF token).
# - Then tries HF HTTP embeddings endpoint as fallback.
# - Includes clear error messages when embedding backends fail.
# """

# import os
# import io
# import hashlib
# import tempfile
# from pathlib import Path
# from typing import List, Optional, Dict, Any

# import requests
# import streamlit as st
# from pypdf import PdfReader
# from PIL import Image

# # Chroma
# import chromadb
# from chromadb.config import Settings

# # Graceful optional imports
# OCR_AVAILABLE = False
# TRANSFORMERS_AVAILABLE = False
# LOCAL_S2_AVAILABLE = False
# HUGGINGFACE_HUB_AVAILABLE = False

# try:
#     from pdf2image import convert_from_bytes
#     import pytesseract
#     OCR_AVAILABLE = True
# except Exception:
#     OCR_AVAILABLE = False

# try:
#     from transformers import pipeline
#     TRANSFORMERS_AVAILABLE = True
# except Exception:
#     TRANSFORMERS_AVAILABLE = False

# try:
#     from sentence_transformers import SentenceTransformer
#     LOCAL_S2_AVAILABLE = True
# except Exception:
#     LOCAL_S2_AVAILABLE = False

# try:
#     from huggingface_hub import InferenceClient
#     HUGGINGFACE_HUB_AVAILABLE = True
# except Exception:
#     HUGGINGFACE_HUB_AVAILABLE = False

# # -------------------------
# # FALLBACK KEYS (user provided)
# # -------------------------
# GROQ_KEY_FALLBACK = "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"
# HF_KEY_FALLBACK = "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"

# # -------------------------
# # CONFIG
# # -------------------------
# HF_TIMEOUT = 60
# TMP_BASE = Path(tempfile.gettempdir()) / "streamlit_rag_chroma_final_v2"
# TMP_BASE.mkdir(parents=True, exist_ok=True)
# MAX_UPLOAD_MB = 200
# BATCH_SIZE = 16

# HF_EMBED_MODELS = ["intfloat/e5-small", "sentence-transformers/all-MiniLM-L6-v2"]
# HF_IMAGE_MODELS = ["nlpconnect/vit-gpt2-image-captioning", "Salesforce/blip-image-captioning-base"]
# HF_TEXT_MODEL_FALLBACK = "google/flan-t5-large"

# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"

# st.set_page_config(page_title="RAG Chat — Final v2", layout="wide")

# # -------------------------
# # Tokens (secrets -> env -> fallback)
# # -------------------------
# def get_hf_token() -> Optional[str]:
#     token = None
#     try:
#         token = os.environ.get("HF_API_TOKEN") or (st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else None)
#     except Exception:
#         token = os.environ.get("HF_API_TOKEN")
#     if token and token.strip():
#         return token.strip()
#     return HF_KEY_FALLBACK

# def get_groq_key() -> Optional[str]:
#     key = None
#     try:
#         key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None)
#     except Exception:
#         key = os.environ.get("GROQ_API_KEY")
#     if key and key.strip():
#         return key.strip()
#     return GROQ_KEY_FALLBACK

# # -------------------------
# # cached clients & local models
# # -------------------------
# @st.cache_resource
# def get_hf_inference_client_cached():
#     token = get_hf_token()
#     if not token:
#         return None
#     if not HUGGINGFACE_HUB_AVAILABLE:
#         return None
#     try:
#         return InferenceClient(token=token)
#     except Exception:
#         return None

# @st.cache_resource
# def get_local_sentence_transformer_cached(model_name="all-MiniLM-L6-v2"):
#     if not LOCAL_S2_AVAILABLE:
#         raise RuntimeError("sentence-transformers not installed on server.")
#     return SentenceTransformer(model_name)

# @st.cache_resource
# def get_local_caption_pipeline_cached(model_name="nlpconnect/vit-gpt2-image-captioning"):
#     if not TRANSFORMERS_AVAILABLE:
#         raise RuntimeError("transformers not installed on server.")
#     return pipeline("image-to-text", model=model_name, device=-1)

# # -------------------------
# # Chroma client helper
# # -------------------------
# def make_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
#     persist_directory = str(persist_directory) if persist_directory else None
#     candidates = []
#     if persist_directory:
#         candidates.append({"persist_directory": persist_directory, "chroma_api_impl": "duckdb+parquet"})
#         candidates.append({"persist_directory": persist_directory, "chroma_db_impl": "duckdb+parquet"})
#         candidates.append({"persist_directory": persist_directory})
#     else:
#         candidates.append({})
#     last_exc = None
#     for cfg in candidates:
#         try:
#             settings = Settings(**cfg)
#             client = chromadb.Client(settings)
#             _ = client.list_collections()
#             st.info(f"Chroma client initialized with settings: {list(cfg.keys()) or ['default']}")
#             return client
#         except Exception as e:
#             last_exc = e
#             continue
#     try:
#         client = chromadb.Client()
#         st.warning("Falling back to in-memory Chroma (no persistence).")
#         return client
#     except Exception as e:
#         raise RuntimeError(f"Could not initialize Chroma client. Last error: {last_exc or e}")

# # -------------------------
# # Utilities
# # -------------------------
# def fingerprint_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
#     h = hashlib.sha256()
#     for f in sorted(files, key=lambda x: x.name):
#         h.update(f.name.encode("utf-8"))
#         try:
#             size = getattr(f, "size", None)
#             if size is None:
#                 size = len(f.getbuffer())
#         except Exception:
#             size = 0
#         h.update(str(size).encode("utf-8"))
#     return h.hexdigest()[:16]

# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#     if not text:
#         return []
#     chunks = []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = min(start + chunk_size, L)
#         chunks.append(text[start:end])
#         if end >= L:
#             break
#         start = end - overlap
#     return chunks

# # -------------------------
# # PDF extraction (pypdf -> OCR)
# # -------------------------
# def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
#     try:
#         reader = PdfReader(io.BytesIO(pdf_bytes))
#         pages = [p.extract_text() or "" for p in reader.pages]
#         text = "\n\n".join(pages).strip()
#         if text:
#             st.info("Extracted text via pypdf.")
#             return text
#         st.info("pypdf returned no text (likely image-only PDF).")
#     except Exception as e:
#         st.info(f"pypdf extraction error: {e}")
#     if OCR_AVAILABLE:
#         try:
#             images = convert_from_bytes(pdf_bytes, dpi=300)
#             page_texts = []
#             for im in images:
#                 page_texts.append(pytesseract.image_to_string(im))
#             aggregated = "\n\n".join(page_texts).strip()
#             if aggregated:
#                 st.info("Extracted text via OCR (pytesseract).")
#                 return aggregated
#             st.info("OCR returned empty text.")
#         except Exception as e:
#             st.info(f"OCR conversion error: {e}")
#     else:
#         st.info("OCR not available (pdf2image/pytesseract missing).")
#     return ""

# # -------------------------
# # Image caption / OCR
# # -------------------------
# def hf_http_image_caption(model: str, image_bytes: bytes) -> Optional[str]:
#     token = get_hf_token()
#     if not token:
#         return None
#     url = f"https://api-inference.huggingface.co/models/{model}"
#     try:
#         r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, files={"image": ("img.jpg", image_bytes, "image/jpeg")}, timeout=HF_TIMEOUT)
#         if r.status_code in (403, 404, 410):
#             return None
#         r.raise_for_status()
#         j = r.json()
#         if isinstance(j, list) and j and isinstance(j[0], dict):
#             return j[0].get("generated_text") or j[0].get("caption") or str(j[0])
#         if isinstance(j, dict):
#             if "generated_text" in j:
#                 return j["generated_text"]
#             if "caption" in j:
#                 return j["caption"]
#         return str(j)
#     except Exception:
#         return None

# def caption_image_resilient(image_bytes: bytes, candidate_models: List[str]) -> str:
#     hf_client = get_hf_inference_client_cached()
#     if hf_client:
#         for model in candidate_models:
#             try:
#                 resp = hf_client(inputs=image_bytes, model=model, timeout=HF_TIMEOUT)
#                 if isinstance(resp, dict) and "generated_text" in resp:
#                     st.info(f"Image caption via HF client (model={model})")
#                     return resp["generated_text"]
#                 if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
#                     st.info(f"Image caption via HF client (model={model})")
#                     return resp[0]["generated_text"]
#             except Exception:
#                 continue
#     for model in candidate_models:
#         out = hf_http_image_caption(model, image_bytes)
#         if out:
#             st.info(f"Image caption via HF HTTP endpoint (model={model})")
#             return out
#     if TRANSFORMERS_AVAILABLE:
#         try:
#             pipe = get_local_caption_pipeline_cached(candidate_models[0])
#             pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             res = pipe(pil)
#             if isinstance(res, list) and res and isinstance(res[0], dict):
#                 txt = res[0].get("generated_text") or res[0].get("caption")
#                 if txt:
#                     st.info("Image caption via local transformers pipeline.")
#                     return txt
#             if isinstance(res, str):
#                 return res
#             if isinstance(res, list) and res and isinstance(res[0], str):
#                 return res[0]
#         except Exception as e:
#             st.info(f"Local transformers caption error: {e}")
#     if OCR_AVAILABLE:
#         try:
#             pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             txt = pytesseract.image_to_string(pil)
#             if txt and txt.strip():
#                 st.info("Image OCR (pytesseract) returned text.")
#                 return txt
#             st.info("Image OCR returned empty text.")
#         except Exception as e:
#             st.info(f"Image OCR error: {e}")
#     raise RuntimeError("Image captioning/OCR failed. Ensure HF token is set or install transformers/pytesseract on the server.")

# # -------------------------
# # Embeddings (robust)
# # -------------------------
# def hf_http_embeddings(model: str, texts: List[str]) -> Optional[List[List[float]]]:
#     token = get_hf_token()
#     if not token:
#         return None
#     url = f"https://api-inference.huggingface.co/embeddings/{model}"
#     try:
#         r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json={"inputs": texts}, timeout=HF_TIMEOUT)
#         if r.status_code in (403, 404, 410):
#             return None
#         r.raise_for_status()
#         j = r.json()
#         if isinstance(j, list) and j and isinstance(j[0], list):
#             return j
#         if isinstance(j, list) and j and isinstance(j[0], dict):
#             out = []
#             for itm in j:
#                 if "embedding" in itm:
#                     out.append(itm["embedding"])
#                 elif "vector" in itm:
#                     out.append(itm["vector"])
#             if out:
#                 return out
#         return None
#     except Exception:
#         return None

# def embed_texts_resilient(texts: List[str], candidate_models: List[str]) -> List[List[float]]:
#     # 1) Local sentence-transformers if installed
#     if LOCAL_S2_AVAILABLE:
#         try:
#             model_name = candidate_models[0] if candidate_models else "all-MiniLM-L6-v2"
#             st.info(f"Using local sentence-transformers model: {model_name}")
#             m = get_local_sentence_transformer_cached(model_name)
#             arr = m.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE, convert_to_numpy=True)
#             return [v.tolist() for v in arr]
#         except Exception as e:
#             st.info(f"Local sentence-transformers failed: {e}")

#     # 2) HuggingFace InferenceClient.embeddings()
#     hf_client = get_hf_inference_client_cached()
#     if hf_client:
#         for m in candidate_models:
#             try:
#                 st.info(f"Trying HF InferenceClient embeddings with model: {m}")
#                 out = hf_client.embeddings(model=m, inputs=texts)
#                 if out and isinstance(out, list):
#                     return out
#             except Exception as e:
#                 st.info(f"HF InferenceClient embeddings failed for {m}: {e}")
#                 continue

#     # 3) HF HTTP embeddings endpoint
#     for m in candidate_models:
#         st.info(f"Trying HF HTTP embeddings endpoint with model: {m}")
#         out = hf_http_embeddings(m, texts)
#         if out:
#             return out

#     # If we arrived here, all backends failed
#     # Provide clear instructions to user in the raised error
#     raise RuntimeError(
#         "All embedding backends failed.\n"
#         "Possible fixes:\n"
#         " - Set a valid HF API token in Streamlit Secrets as HF_API_TOKEN (or env var).\n"
#         " - Or install 'sentence-transformers' locally (heavy) so the app can compute embeddings on the host.\n"
#         " - Check network/firewall and HF token permissions.\n"
#         "See app logs for details."
#     )

# # -------------------------
# # Build Chroma from uploads
# # -------------------------
# def build_chroma_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], candidate_embed_models: List[str], max_chunks: Optional[int] = None):
#     if not uploaded_files:
#         return None
#     fp = fingerprint_files(uploaded_files)
#     persist_dir = TMP_BASE / fp
#     persist_dir.mkdir(parents=True, exist_ok=True)
#     try:
#         client = make_chroma_client(str(persist_dir))
#     except Exception as e:
#         st.error(f"Chroma init error: {e}")
#         return None
#     try:
#         try:
#             col = client.get_collection(name=f"col_{fp}")
#         except Exception:
#             col = client.create_collection(name=f"col_{fp}")
#     except Exception as e:
#         st.error(f"Chroma collection error: {e}")
#         return None

#     try:
#         if col.count() and col.count() > 0:
#             st.success(f"Collection already indexed with {col.count()} items.")
#             return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": col.count(), "snippets": []}
#     except Exception:
#         pass

#     texts = []
#     ids = []
#     metadatas = []
#     snippets = []

#     for uploaded in uploaded_files:
#         try:
#             size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
#         except Exception:
#             size = 0
#         if size > MAX_UPLOAD_MB * 1024 * 1024:
#             st.warning(f"Skipping {uploaded.name} (> {MAX_UPLOAD_MB}MB)")
#             continue
#         data = uploaded.getbuffer().tobytes()
#         file_text = ""
#         if uploaded.name.lower().endswith(".pdf"):
#             file_text = extract_text_from_pdf_bytes(data)
#         else:
#             try:
#                 file_text = data.decode("utf-8", errors="ignore")
#             except Exception:
#                 try:
#                     file_text = data.decode("latin-1", errors="ignore")
#                 except Exception:
#                     file_text = ""
#         if not file_text or not file_text.strip():
#             st.info(f"No text extracted from {uploaded.name}; skipping.")
#             continue
#         chunks = chunk_text(file_text)
#         for i, c in enumerate(chunks):
#             uid = f"{uploaded.name}__{i}"
#             ids.append(uid)
#             texts.append(c)
#             metadatas.append({"source": uploaded.name, "chunk": i})
#             snippets.append({"name": uploaded.name, "preview": c[:200]})

#     if not texts:
#         st.warning("No text extracted to index.")
#         return None

#     if max_chunks and len(texts) > max_chunks:
#         texts = texts[:max_chunks]
#         ids = ids[:max_chunks]
#         metadatas = metadatas[:max_chunks]

#     # compute embeddings
#     try:
#         embeddings = []
#         progress = st.progress(0)
#         total = len(texts)
#         for i in range(0, total, BATCH_SIZE):
#             batch = texts[i : i + BATCH_SIZE]
#             ev = embed_texts_resilient(batch, candidate_embed_models)
#             embeddings.extend(ev)
#             progress.progress(min(100, int(100 * (i + len(batch)) / total)))
#         progress.empty()
#     except Exception as e:
#         st.error(f"Embedding computation failed: {e}")
#         return None

#     try:
#         col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
#         try:
#             client.persist()
#         except Exception:
#             st.info("client.persist() unsupported or failed in this environment (using in-memory/partial persist).")
#         return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": len(texts), "snippets": snippets}
#     except Exception as e:
#         st.error(f"Failed to save to Chroma: {e}")
#         return None

# # -------------------------
# # Query helper
# # -------------------------
# def query_chroma_collection(collection, query_text: str, candidate_embed_models: List[str], k: int = 3):
#     if collection is None:
#         return []
#     try:
#         qv = embed_texts_resilient([query_text], candidate_embed_models)[0]
#     except Exception as e:
#         st.error(f"Query embed error: {e}")
#         return []
#     try:
#         res = collection.query(query_embeddings=[qv], n_results=k, include=["documents","metadatas","distances"])
#     except Exception as e:
#         st.error(f"Chroma query error: {e}")
#         return []
#     docs = []
#     docs_list = res.get("documents", [[]])[0]
#     metas_list = res.get("metadatas", [[]])[0]
#     dists = res.get("distances", [[]])[0]
#     for doc, meta, dist in zip(docs_list, metas_list, dists):
#         docs.append({"document": doc, "metadata": meta, "distance": dist})
#     return docs

# # -------------------------
# # UI
# # -------------------------
# with st.sidebar:
#     st.header("Settings")
#     # st.markdown("HF token: set `HF_API_TOKEN` in Streamlit Secrets or env (recommended).")
#     # st.markdown("GROQ key: set `GROQ_API_KEY` in Secrets/env to use Groq (optional).")
#     uploaded_files = st.file_uploader("Upload PDF / TXT (multiple)", type=["pdf","txt"], accept_multiple_files=True)
#     image_file = st.file_uploader("Upload image (optional)", type=["png","jpg","jpeg"])
#     st.markdown("---")
#     st.subheader("Indexing options")
#     max_chunks = st.number_input("Max chunks to index (0=no limit)", value=0, min_value=0)
#     memory_window = st.slider("Memory window", 1, 8, 4)
#     st.markdown("---")
#     if st.button("Clear conversation & index"):
#         for k in ["messages","chroma_info","last_fp"]:
#             if k in st.session_state:
#                 del st.session_state[k]
#         st.rerun()

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []
# if "chroma_info" not in st.session_state:
#     st.session_state["chroma_info"] = None

# # Show embedding backend availability banner
# colA, colB, colC = st.columns(3)
# with colA:
#     st.write("Local S2:", "Yes" if LOCAL_S2_AVAILABLE else "No")
# with colB:
#     st.write("HF InferenceClient:", "Yes" if HUGGINGFACE_HUB_AVAILABLE else "No")
# with colC:
#     st.write("OCR:", "Yes" if OCR_AVAILABLE else "No")

# st.title("RAG Chat — Final v2 (robust embeddings)")

# if uploaded_files and st.session_state.get("chroma_info") is None:
#     try:
#         new_fp = fingerprint_files(uploaded_files)
#         prev_fp = st.session_state.get("last_fp")
#         if new_fp != prev_fp:
#             st.session_state["last_fp"] = new_fp
#             with st.spinner("Indexing uploaded files..."):
#                 max_chunks_val = None if int(max_chunks) == 0 else int(max_chunks)
#                 info = build_chroma_from_uploads(uploaded_files, HF_EMBED_MODELS, max_chunks=max_chunks_val)
#                 if info:
#                     st.session_state["chroma_info"] = info
#                     st.success(f"Indexed {info.get('indexed_chunks')} chunks.")
#                 else:
#                     st.error("Indexing failed. Check logs or tokens.")
#     except Exception as e:
#         st.error(f"Auto-indexing exception: {e}")

# col1, col2 = st.columns([3,1])

# with col1:
#     for m in st.session_state["messages"]:
#         role = m.get("role","assistant")
#         content = m.get("content","")
#         if role == "user":
#             st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

#     if image_file:
#         st.markdown("### Uploaded Image")
#         st.image(image_file, use_column_width=True)
#         q_img = st.text_input("Ask something about this image (optional):")
#         if st.button("Analyze Image"):
#             img_bytes = image_file.getvalue()
#             with st.spinner("Captioning/reading image..."):
#                 try:
#                     caption = caption_image_resilient(img_bytes, HF_IMAGE_MODELS)
#                 except Exception as e:
#                     caption = f"[Caption/OCR error] {e}"
#             prompt = f"Image caption / OCR:\n{caption}\n\nQuestion: {q_img or 'Describe the image.'}\nAnswer concisely."
#             groq_key = get_groq_key()
#             answer = ""
#             if groq_key:
#                 try:
#                     r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
#                                       json={"model": GROQ_MODEL_DEFAULT, "messages":[{"role":"user","content":prompt}], "max_tokens":512}, timeout=HF_TIMEOUT)
#                     r.raise_for_status()
#                     j = r.json()
#                     choice = j.get("choices",[{}])[0]
#                     text = ""
#                     if isinstance(choice.get("message"), dict):
#                         text = choice["message"].get("content") or ""
#                     else:
#                         text = choice.get("text") or ""
#                     answer = text or str(j)
#                 except Exception as e:
#                     answer = f"[Groq generation error] {e}"
#             else:
#                 hf_token = get_hf_token()
#                 if hf_token:
#                     try:
#                         url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL_FALLBACK}"
#                         r = requests.post(url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
#                         r.raise_for_status()
#                         out = r.json()
#                         if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
#                             answer = out[0]["generated_text"]
#                         else:
#                             answer = str(out)
#                     except Exception as e:
#                         answer = f"[HF text generation error] {e}"
#                 else:
#                     answer = "No HF token or Groq key available for text generation."
#             st.session_state["messages"].append({"role":"assistant","content": f"🖼 Caption/OCR:\n{caption}\n\nAnswer:\n{answer}"})
#             st.rerun()

#     with st.form("chat_form", clear_on_submit=True):
#         user_input = st.text_area("Type your message:", height=140)
#         send = st.form_submit_button("Send")

#     if send and user_input and user_input.strip():
#         st.session_state["messages"].append({"role":"user","content":user_input})
#         with st.spinner("Thinking..."):
#             try:
#                 chroma_info = st.session_state.get("chroma_info")
#                 answer_text = ""
#                 if chroma_info and chroma_info.get("collection"):
#                     collection = chroma_info["collection"]
#                     docs = query_chroma_collection(collection, user_input, HF_EMBED_MODELS, k=3)
#                     retrieved = "\n\n---\n\n".join([d.get("document","")[:1000] for d in docs])
#                     mem = st.session_state.get("messages", [])[-memory_window:]
#                     mem_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in mem])
#                     prompt = f"Memory:\n{mem_text}\n\nContext:\n{retrieved}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
#                 else:
#                     prompt = f"Question:\n{user_input}\nAnswer concisely."
#                 groq_key = get_groq_key()
#                 if groq_key:
#                     try:
#                         r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
#                                           json={"model": GROQ_MODEL_DEFAULT, "messages":[{"role":"user","content":prompt}], "max_tokens":512}, timeout=HF_TIMEOUT)
#                         r.raise_for_status()
#                         j = r.json()
#                         choice = j.get("choices",[{}])[0]
#                         text = ""
#                         if isinstance(choice.get("message"), dict):
#                             text = choice["message"].get("content") or ""
#                         else:
#                             text = choice.get("text") or ""
#                         answer_text = text or str(j)
#                     except Exception as e:
#                         answer_text = f"[Groq generation error] {e}"
#                 else:
#                     hf_token = get_hf_token()
#                     if hf_token:
#                         try:
#                             url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL_FALLBACK}"
#                             r = requests.post(url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
#                             r.raise_for_status()
#                             out = r.json()
#                             if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
#                                 answer_text = out[0]["generated_text"]
#                             else:
#                                 answer_text = str(out)
#                         except Exception as e:
#                             answer_text = f"[HF text generation error] {e}"
#                     else:
#                         answer_text = "No Groq key or HF token available for generation."
#                 st.session_state["messages"].append({"role":"assistant","content": answer_text})
#             except Exception as e:
#                 st.session_state["messages"].append({"role":"assistant","content": f"[Unhandled error] {e}"})
#         st.rerun()

# with col2:
#     st.markdown("### RAG & Status")
#     chroma_info = st.session_state.get("chroma_info")
#     if chroma_info:
#         st.success("Chroma index loaded")
#         st.write({"persist_dir": chroma_info.get("persist_dir"), "indexed_chunks": chroma_info.get("indexed_chunks")})
#         if chroma_info.get("snippets"):
#             st.markdown("**Document previews:**")
#             for s in chroma_info["snippets"][:6]:
#                 st.markdown(f"- **{s['name']}** — `{s['preview'][:160]}...`")
#     else:
#         st.info("No Chroma index found. Upload PDFs/TXT to enable RAG.")

#     st.markdown("---")
#     if uploaded_files and st.button("Index uploaded files (manual)"):
#         with st.spinner("Indexing uploaded files..."):
#             info = build_chroma_from_uploads(uploaded_files, HF_EMBED_MODELS, max_chunks=None if int(max_chunks) == 0 else int(max_chunks))
#             if info:
#                 st.session_state["chroma_info"] = info
#                 st.success(f"Indexed {info.get('indexed_chunks')} chunks.")
#             else:
#                 st.error("Indexing failed.")

#     st.markdown("---")
#     if st.button("Show memory"):
#         mem = st.session_state.get("messages", [])[-memory_window:]
#         st.write(mem)

#     st.markdown("---")
#     # st.caption("Notes: If embedding computation fails, set HF_API_TOKEN in Streamlit Secrets (or install sentence-transformers locally). For scanned PDFs, Poppler + Tesseract required on the host for OCR.")



# app_streamlit_final_v2_with_live_tools.py
"""
Updated Streamlit app (based on your original file) with live-data tool integration.
Features added:
- Intent detection for live queries (time, date, weather, news, stock, currency, web-search)
- Tool implementations that call real-world APIs (OpenWeather, NewsAPI, Alpha Vantage, exchangerate.host)
- Router that runs tools *before* calling the LLM (Groq/HF)
- Helper `get_secret` to read keys from `st.secrets` or environment variables

IMPORTANT:
- Do NOT hardcode API keys in code for production. Put them in Streamlit Secrets or env vars.
- This file intentionally reads keys from `st.secrets`/env; set them before running.

"""

import os
import io
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader
from PIL import Image

# Chroma
import chromadb
from chromadb.config import Settings

# Graceful optional imports
OCR_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LOCAL_S2_AVAILABLE = False
HUGGINGFACE_HUB_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_S2_AVAILABLE = True
except Exception:
    LOCAL_S2_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_HUB_AVAILABLE = True
except Exception:
    HUGGINGFACE_HUB_AVAILABLE = False

# -------------------------
# EMBEDDED API KEYS (user requested)
# -------------------------
# You asked to embed API keys directly into the code. This is included here per your request,
# but be aware that committing these keys to a public repo is insecure. Prefer Streamlit secrets or env vars.

OPENWEATHER_KEY = "cfb90f018bd56eb2030692333f5a4c8e"
NEWSAPI_KEY = "2c8708d41814476da404560666af3b3a"
ALPHA_VANTAGE_KEY = "WW7N5HKIQK2S3AEW"
EXCHANGE_RATE_KEY = "9c6f013c49ece4a2dae3efa983dfd00d"

# Keep HF/Groq fallbacks as None by default (you can set them in st.secrets or env if needed)
HF_KEY_FALLBACK = "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"
GROQ_KEY_FALLBACK = "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"

# For convenience, if get_secret(...) is called for these names, we will let get_secret check the
# module-level constants as valid fallbacks (see get_secret implementation below).
# -------------------------
# CONFIG
# -------------------------
HF_TIMEOUT = 60
TMP_BASE = Path(tempfile.gettempdir()) / "streamlit_rag_chroma_final_v2"
TMP_BASE.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 200
BATCH_SIZE = 16

HF_EMBED_MODELS = ["intfloat/e5-small", "sentence-transformers/all-MiniLM-L6-v2"]
HF_IMAGE_MODELS = ["nlpconnect/vit-gpt2-image-captioning", "Salesforce/blip-image-captioning-base"]
HF_TEXT_MODEL_FALLBACK = "google/flan-t5-large"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"

st.set_page_config(page_title="RAG Chat — Final v2 (+ live tools)", layout="wide")

# -------------------------
# Secrets helper
# -------------------------
def get_secret(name: str) -> Optional[str]:
    """Read secret from st.secrets or environment. If not set there, fall back to module-level constants
    (so embedded keys like OPENWEATHER_KEY, HF_KEY_FALLBACK, etc. will be returned when present).
    Returns None if nothing is found."""
    try:
        # 1) environment
        val = os.environ.get(name)
        if val:
            return val
        # 2) streamlit secrets
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
        # 3) module-level embedded constants (fall back)
        g = globals()
        if name in g and g[name]:
            return g[name]
    except Exception:
        pass
    return None

# Convenience token getters (use get_secret but preserve original functions' names for compatibility)

def get_hf_token() -> Optional[str]:
    token = None
    try:
        token = get_secret("HF_API_TOKEN")
    except Exception:
        token = os.environ.get("HF_API_TOKEN")
    if token and token.strip():
        return token.strip()
    return HF_KEY_FALLBACK


def get_groq_key() -> Optional[str]:
    key = None
    try:
        key = get_secret("GROQ_API_KEY")
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    return GROQ_KEY_FALLBACK

# -------------------------
# cached clients & local models
# -------------------------
@st.cache_resource
def get_hf_inference_client_cached():
    token = get_hf_token()
    if not token:
        return None
    if not HUGGINGFACE_HUB_AVAILABLE:
        return None
    try:
        return InferenceClient(token=token)
    except Exception:
        return None

@st.cache_resource
def get_local_sentence_transformer_cached(model_name="all-MiniLM-L6-v2"):
    if not LOCAL_S2_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed on server.")
    return SentenceTransformer(model_name)

@st.cache_resource
def get_local_caption_pipeline_cached(model_name="nlpconnect/vit-gpt2-image-captioning"):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed on server.")
    return pipeline("image-to-text", model=model_name, device=-1)

# -------------------------
# Chroma client helper
# -------------------------
def make_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    persist_directory = str(persist_directory) if persist_directory else None
    candidates = []
    if persist_directory:
        candidates.append({"persist_directory": persist_directory, "chroma_api_impl": "duckdb+parquet"})
        candidates.append({"persist_directory": persist_directory, "chroma_db_impl": "duckdb+parquet"})
        candidates.append({"persist_directory": persist_directory})
    else:
        candidates.append({})
    last_exc = None
    for cfg in candidates:
        try:
            settings = Settings(**cfg)
            client = chromadb.Client(settings)
            _ = client.list_collections()
            st.info(f"Chroma client initialized with settings: {list(cfg.keys()) or ['default']}")
            return client
        except Exception as e:
            last_exc = e
            continue
    try:
        client = chromadb.Client()
        st.warning("Falling back to in-memory Chroma (no persistence).")
        return client
    except Exception as e:
        raise RuntimeError(f"Could not initialize Chroma client. Last error: {last_exc or e}")

# -------------------------
# Utilities (unchanged)
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


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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

# -------------------------
# PDF extraction (unchanged)
# -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(pages).strip()
        if text:
            st.info("Extracted text via pypdf.")
            return text
        st.info("pypdf returned no text (likely image-only PDF).")
    except Exception as e:
        st.info(f"pypdf extraction error: {e}")
    if OCR_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            page_texts = []
            for im in images:
                page_texts.append(pytesseract.image_to_string(im))
            aggregated = "\n\n".join(page_texts).strip()
            if aggregated:
                st.info("Extracted text via OCR (pytesseract).")
                return aggregated
            st.info("OCR returned empty text.")
        except Exception as e:
            st.info(f"OCR conversion error: {e}")
    else:
        st.info("OCR not available (pdf2image/pytesseract missing).")
    return ""

# -------------------------
# Image caption / OCR (unchanged)
# -------------------------
def hf_http_image_caption(model: str, image_bytes: bytes) -> Optional[str]:
    token = get_hf_token()
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, files={"image": ("img.jpg", image_bytes, "image/jpeg")}, timeout=HF_TIMEOUT)
        if r.status_code in (403, 404, 410):
            return None
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j and isinstance(j[0], dict):
            return j[0].get("generated_text") or j[0].get("caption") or str(j[0])
        if isinstance(j, dict):
            if "generated_text" in j:
                return j["generated_text"]
            if "caption" in j:
                return j["caption"]
        return str(j)
    except Exception:
        return None


def caption_image_resilient(image_bytes: bytes, candidate_models: List[str]) -> str:
    hf_client = get_hf_inference_client_cached()
    if hf_client:
        for model in candidate_models:
            try:
                resp = hf_client(inputs=image_bytes, model=model, timeout=HF_TIMEOUT)
                if isinstance(resp, dict) and "generated_text" in resp:
                    st.info(f"Image caption via HF client (model={model})")
                    return resp["generated_text"]
                if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
                    st.info(f"Image caption via HF client (model={model})")
                    return resp[0]["generated_text"]
            except Exception:
                continue
    for model in candidate_models:
        out = hf_http_image_caption(model, image_bytes)
        if out:
            st.info(f"Image caption via HF HTTP endpoint (model={model})")
            return out
    if TRANSFORMERS_AVAILABLE:
        try:
            pipe = get_local_caption_pipeline_cached(candidate_models[0])
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            res = pipe(pil)
            if isinstance(res, list) and res and isinstance(res[0], dict):
                txt = res[0].get("generated_text") or res[0].get("caption")
                if txt:
                    st.info("Image caption via local transformers pipeline.")
                    return txt
            if isinstance(res, str):
                return res
            if isinstance(res, list) and res and isinstance(res[0], str):
                return res[0]
        except Exception as e:
            st.info(f"Local transformers caption error: {e}")
    if OCR_AVAILABLE:
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            txt = pytesseract.image_to_string(pil)
            if txt and txt.strip():
                st.info("Image OCR (pytesseract) returned text.")
                return txt
            st.info("Image OCR returned empty text.")
        except Exception as e:
            st.info(f"Image OCR error: {e}")
    raise RuntimeError("Image captioning/OCR failed. Ensure HF token is set or install transformers/pytesseract on the server.")

# -------------------------
# Embeddings (unchanged)
# -------------------------
def hf_http_embeddings(model: str, texts: List[str]) -> Optional[List[List[float]]]:
    token = get_hf_token()
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/embeddings/{model}"
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json={"inputs": texts}, timeout=HF_TIMEOUT)
        if r.status_code in (403, 404, 410):
            return None
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j and isinstance(j[0], list):
            return j
        if isinstance(j, list) and j and isinstance(j[0], dict):
            out = []
            for itm in j:
                if "embedding" in itm:
                    out.append(itm["embedding"])
                elif "vector" in itm:
                    out.append(itm["vector"])
            if out:
                return out
        return None
    except Exception:
        return None


def embed_texts_resilient(texts: List[str], candidate_models: List[str]) -> List[List[float]]:
    if LOCAL_S2_AVAILABLE:
        try:
            model_name = candidate_models[0] if candidate_models else "all-MiniLM-L6-v2"
            st.info(f"Using local sentence-transformers model: {model_name}")
            m = get_local_sentence_transformer_cached(model_name)
            arr = m.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE, convert_to_numpy=True)
            return [v.tolist() for v in arr]
        except Exception as e:
            st.info(f"Local sentence-transformers failed: {e}")

    hf_client = get_hf_inference_client_cached()
    if hf_client:
        for m in candidate_models:
            try:
                st.info(f"Trying HF InferenceClient embeddings with model: {m}")
                out = hf_client.embeddings(model=m, inputs=texts)
                if out and isinstance(out, list):
                    return out
            except Exception as e:
                st.info(f"HF InferenceClient embeddings failed for {m}: {e}")
                continue

    for m in candidate_models:
        st.info(f"Trying HF HTTP embeddings endpoint with model: {m}")
        out = hf_http_embeddings(m, texts)
        if out:
            return out

    raise RuntimeError(
        "All embedding backends failed.\n"
        "Possible fixes:\n"
        " - Set a valid HF API token in Streamlit Secrets as HF_API_TOKEN (or env var).\n"
        " - Or install 'sentence-transformers' locally (heavy) so the app can compute embeddings on the host.\n"
        " - Check network/firewall and HF token permissions.\n"
        "See app logs for details."
    )

# -------------------------
# Build Chroma from uploads (unchanged)
# -------------------------
def build_chroma_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], candidate_embed_models: List[str], max_chunks: Optional[int] = None):
    if not uploaded_files:
        return None
    fp = fingerprint_files(uploaded_files)
    persist_dir = TMP_BASE / fp
    persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = make_chroma_client(str(persist_dir))
    except Exception as e:
        st.error(f"Chroma init error: {e}")
        return None
    try:
        try:
            col = client.get_collection(name=f"col_{fp}")
        except Exception:
            col = client.create_collection(name=f"col_{fp}")
    except Exception as e:
        st.error(f"Chroma collection error: {e}")
        return None

    try:
        if col.count() and col.count() > 0:
            st.success(f"Collection already indexed with {col.count()} items.")
            return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": col.count(), "snippets": []}
    except Exception:
        pass

    texts = []
    ids = []
    metadatas = []
    snippets = []

    for uploaded in uploaded_files:
        try:
            size = getattr(uploaded, "size", None) or len(uploaded.getbuffer())
        except Exception:
            size = 0
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipping {uploaded.name} (> {MAX_UPLOAD_MB}MB)")
            continue
        data = uploaded.getbuffer().tobytes()
        file_text = ""
        if uploaded.name.lower().endswith(".pdf"):
            file_text = extract_text_from_pdf_bytes(data)
        else:
            try:
                file_text = data.decode("utf-8", errors="ignore")
            except Exception:
                try:
                    file_text = data.decode("latin-1", errors="ignore")
                except Exception:
                    file_text = ""
        if not file_text or not file_text.strip():
            st.info(f"No text extracted from {uploaded.name}; skipping.")
            continue
        chunks = chunk_text(file_text)
        for i, c in enumerate(chunks):
            uid = f"{uploaded.name}__{i}"
            ids.append(uid)
            texts.append(c)
            metadatas.append({"source": uploaded.name, "chunk": i})
            snippets.append({"name": uploaded.name, "preview": c[:200]})

    if not texts:
        st.warning("No text extracted to index.")
        return None

    if max_chunks and len(texts) > max_chunks:
        texts = texts[:max_chunks]
        ids = ids[:max_chunks]
        metadatas = metadatas[:max_chunks]

    try:
        embeddings = []
        progress = st.progress(0)
        total = len(texts)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            ev = embed_texts_resilient(batch, candidate_embed_models)
            embeddings.extend(ev)
            progress.progress(min(100, int(100 * (i + len(batch)) / total)))
        progress.empty()
    except Exception as e:
        st.error(f"Embedding computation failed: {e}")
        return None

    try:
        col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        try:
            client.persist()
        except Exception:
            st.info("client.persist() unsupported or failed in this environment (using in-memory/partial persist).")
        return {"collection": col, "persist_dir": str(persist_dir), "indexed_chunks": len(texts), "snippets": snippets}
    except Exception as e:
        st.error(f"Failed to save to Chroma: {e}")
        return None

# -------------------------
# Query helper (unchanged)
# -------------------------
def query_chroma_collection(collection, query_text: str, candidate_embed_models: List[str], k: int = 3):
    if collection is None:
        return []
    try:
        qv = embed_texts_resilient([query_text], candidate_embed_models)[0]
    except Exception as e:
        st.error(f"Query embed error: {e}")
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
# ---------- LIVE TOOL HELPERS ----------
# -------------------------
from datetime import datetime
import re

def tool_get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def tool_get_today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def tool_convert_timezone(dt_str: str, from_tz: str, to_tz: str) -> str:
    return "Timezone conversion not implemented in lightweight mode. Install zoneinfo/pytz for production." 

# OpenWeather (requires OPENWEATHER_KEY in st.secrets or env)
def tool_get_weather(city: str) -> str:
    key = get_secret("OPENWEATHER_KEY")
    if not key:
        return "OpenWeather key not set. Add OPENWEATHER_KEY to env or Streamlit secrets."
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(url, params={"q": city, "appid": key, "units": "metric"}, timeout=10)
        j = r.json()
        if r.status_code != 200:
            return f"Weather API error: {j.get('message', r.text)}"
        temp = j["main"]["temp"]
        desc = j["weather"][0]["description"]
        return f"Weather in {city}: {temp}°C, {desc}"
    except Exception as e:
        return f"Weather fetch failed: {e}"

# News (NewsAPI.org)
def tool_get_news(query: str, page_size: int = 3) -> str:
    key = get_secret("NEWSAPI_KEY")
    if not key:
        return "News API key not set. Add NEWSAPI_KEY to env or Streamlit secrets."
    url = "https://newsapi.org/v2/everything"
    try:
        r = requests.get(url, params={"q": query, "pageSize": page_size, "apiKey": key}, timeout=10)
        j = r.json()
        if r.status_code != 200:
            return f"News API error: {j.get('message', r.text)}"
        items = j.get("articles", [])[:page_size]
        if not items:
            return "No news found for that query."
        out = []
        for a in items:
            out.append(f"- {a.get('title')} ({a.get('source',{}).get('name')}) — {a.get('publishedAt')}\n  {a.get('url')}")
        return "\n\n".join(out)
    except Exception as e:
        return f"News fetch failed: {e}"

# Stock price (Alpha Vantage)
def tool_get_stock(ticker: str) -> str:
    key = get_secret("ALPHA_VANTAGE_KEY")
    if not key:
        return "Alpha Vantage key not set. Add ALPHA_VANTAGE_KEY to env or Streamlit secrets."
    url = "https://www.alphavantage.co/query"
    try:
        r = requests.get(url, params={"function":"GLOBAL_QUOTE","symbol":ticker,"apikey":key}, timeout=10)
        j = r.json()
        quote = j.get("Global Quote", {})
        if not quote:
            return f"No stock data for {ticker}."
        price = quote.get("05. price")
        change = quote.get("09. change")
        pct = quote.get("10. change percent")
        return f"{ticker.upper()}: {price} USD (change {change}, {pct})"
    except Exception as e:
        return f"Stock fetch failed: {e}"

# Currency conversion using exchangerate.host (no API key required)
def tool_convert_currency(amount: float, from_ccy: str, to_ccy: str) -> str:
    try:
        url = f"https://api.exchangerate.host/convert"
        r = requests.get(url, params={"from": from_ccy, "to": to_ccy, "amount": amount}, timeout=10)
        j = r.json()
        if not j.get("success", True):
            return "Currency conversion failed."
        result = j.get("result")
        return f"{amount} {from_ccy.upper()} = {result:.4f} {to_ccy.upper()}"
    except Exception as e:
        return f"Currency conversion error: {e}"

# Simple web search fallback using DuckDuckGo HTML (no API required)
def tool_web_search(query: str, max_results: int = 3) -> str:
    try:
        url = "https://html.duckduckgo.com/html/"
        r = requests.post(url, data={"q": query}, timeout=10)
        txt = r.text
        links = re.findall(r'<a rel="nofollow" class="result__a" href="([^\"]+)"', txt)[:max_results]
        if not links:
            return "No quick search results."
        return "\n".join([f"- {u}" for u in links])
    except Exception as e:
        return f"Web search failed: {e}"

# -------------------------
# ---------- INTENT DETECTION & ROUTER ----------
# -------------------------

def detect_live_intent(user_text: str) -> Dict[str, Any]:
    t = user_text.lower().strip()
    if re.search(r"\b(time|current time|what time)\b", t):
        return {"intent":"time"}
    if re.search(r"\b(date|today's date|what date)\b", t):
        return {"intent":"date"}
    if re.search(r"\b(weather|forecast)\b", t):
        m = re.search(r"weather (in|at) ([a-zA-Z\s,]+)", t)
        city = m.group(2).strip() if m else None
        return {"intent":"weather", "city": city}
    if re.search(r"\b(news|latest news|headlines)\b", t):
        m = re.search(r"news (about|on|for) ([a-zA-Z\s]+)", t)
        return {"intent":"news", "query": (m.group(2).strip() if m else t)}
    if re.search(r"\b(stock|share|price|quote)\b", t):
        m = re.search(r"(?:stock|share|price|quote)[: ]*([A-Za-z\.\-]+)", t)
        ticker = m.group(1).strip() if m else None
        return {"intent":"stock", "ticker": ticker}
    if re.search(r"\b(convert|exchange|how many)\b.*\b(usd|inr|eur|gbp|jpy|aud|cad)\b", t):
        m = re.search(r"([0-9,.]+)\s*([A-Za-z]{3})\s*(to|in)\s*([A-Za-z]{3})", t)
        if m:
            return {"intent":"currency", "amount": float(m.group(1).replace(",","")), "from": m.group(2), "to": m.group(4)}
        return {"intent":"currency"}
    if re.search(r"\b(score|match|fixture|flight|arriv|depart|status)\b", t):
        return {"intent":"web_search", "query": t}
    return {"intent": None}

# -------------------------
# UI (mostly unchanged) but with live-tool routing integrated
# -------------------------
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Upload PDF / TXT (multiple)", type=["pdf","txt"], accept_multiple_files=True)
    image_file = st.file_uploader("Upload image (optional)", type=["png","jpg","jpeg"])
    st.markdown("---")
    st.subheader("Indexing options")
    max_chunks = st.number_input("Max chunks to index (0=no limit)", value=0, min_value=0)
    memory_window = st.slider("Memory window", 1, 8, 4)
    st.markdown("---")
    if st.button("Clear conversation & index"):
        for k in ["messages","chroma_info","last_fp"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chroma_info" not in st.session_state:
    st.session_state["chroma_info"] = None

# Show embedding backend availability banner
colA, colB, colC = st.columns(3)
with colA:
    st.write("Local S2:", "Yes" if LOCAL_S2_AVAILABLE else "No")
with colB:
    st.write("HF InferenceClient:", "Yes" if HUGGINGFACE_HUB_AVAILABLE else "No")
with colC:
    st.write("OCR:", "Yes" if OCR_AVAILABLE else "No")

st.title("RAG Chat — Final v2 (robust embeddings + live tools)")

if uploaded_files and st.session_state.get("chroma_info") is None:
    try:
        new_fp = fingerprint_files(uploaded_files)
        prev_fp = st.session_state.get("last_fp")
        if new_fp != prev_fp:
            st.session_state["last_fp"] = new_fp
            with st.spinner("Indexing uploaded files..."):
                max_chunks_val = None if int(max_chunks) == 0 else int(max_chunks)
                info = build_chroma_from_uploads(uploaded_files, HF_EMBED_MODELS, max_chunks=max_chunks_val)
                if info:
                    st.session_state["chroma_info"] = info
                    st.success(f"Indexed {info.get('indexed_chunks')} chunks.")
                else:
                    st.error("Indexing failed. Check logs or tokens.")
    except Exception as e:
        st.error(f"Auto-indexing exception: {e}")

col1, col2 = st.columns([3,1])

with col1:
    for m in st.session_state["messages"]:
        role = m.get("role","assistant")
        content = m.get("content","")
        if role == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end"><div style="background:#DCF8C6;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display:flex;justify-content:flex-start"><div style="background:#fff;padding:10px;border-radius:10px;max-width:80%">{content}</div></div>', unsafe_allow_html=True)

    if image_file:
        st.markdown("### Uploaded Image")
        st.image(image_file, use_column_width=True)
        q_img = st.text_input("Ask something about this image (optional):")
        if st.button("Analyze Image"):
            img_bytes = image_file.getvalue()
            with st.spinner("Captioning/reading image..."):
                try:
                    caption = caption_image_resilient(img_bytes, HF_IMAGE_MODELS)
                except Exception as e:
                    caption = f"[Caption/OCR error] {e}"
            prompt = f"Image caption / OCR:\n{caption}\n\nQuestion: {q_img or 'Describe the image.'}\nAnswer concisely."
            groq_key = get_groq_key()
            answer = ""
            if groq_key:
                try:
                    r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
                                      json={"model": GROQ_MODEL_DEFAULT, "messages":[{"role":"user","content":prompt}], "max_tokens":512}, timeout=HF_TIMEOUT)
                    r.raise_for_status()
                    j = r.json()
                    choice = j.get("choices",[{}])[0]
                    text = ""
                    if isinstance(choice.get("message"), dict):
                        text = choice["message"].get("content") or ""
                    else:
                        text = choice.get("text") or ""
                    answer = text or str(j)
                except Exception as e:
                    answer = f"[Groq generation error] {e}"
            else:
                hf_token = get_hf_token()
                if hf_token:
                    try:
                        url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL_FALLBACK}"
                        r = requests.post(url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                        r.raise_for_status()
                        out = r.json()
                        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                            answer = out[0]["generated_text"]
                        else:
                            answer = str(out)
                    except Exception as e:
                        answer = f"[HF text generation error] {e}"
                else:
                    answer = "No HF token or Groq key available for text generation."
            st.session_state["messages"].append({"role":"assistant","content": f"🖼 Caption/OCR:\n{caption}\n\nAnswer:\n{answer}"})
            st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=140)
        send = st.form_submit_button("Send")

    if send and user_input and user_input.strip():
        st.session_state["messages"].append({"role":"user","content":user_input})

        # ---------- LIVE TOOL HANDLER ----------
        intent = detect_live_intent(user_input)
        handled = False
        live_answer = None

        if intent.get("intent") == "time":
            live_answer = tool_get_current_time()
            handled = True
        elif intent.get("intent") == "date":
            live_answer = tool_get_today_date()
            handled = True
        elif intent.get("intent") == "weather":
            city = intent.get("city") or ""
            if not city:
                live_answer = "Please specify a city, e.g. 'weather in Delhi'."
            else:
                live_answer = tool_get_weather(city)
            handled = True
        elif intent.get("intent") == "news":
            q = intent.get("query") or user_input
            live_answer = tool_get_news(q)
            handled = True
        elif intent.get("intent") == "stock":
            ticker = intent.get("ticker")
            if not ticker:
                live_answer = "Please specify a ticker symbol, e.g. 'stock AAPL'."
            else:
                live_answer = tool_get_stock(ticker)
            handled = True
        elif intent.get("intent") == "currency":
            if intent.get("amount") and intent.get("from") and intent.get("to"):
                live_answer = tool_convert_currency(intent["amount"], intent["from"], intent["to"])
            else:
                live_answer = "Please ask like: 'Convert 100 USD to INR'."
            handled = True
        elif intent.get("intent") == "web_search":
            q = intent.get("query") or user_input
            live_answer = tool_web_search(q)
            handled = True

        if handled:
            st.session_state["messages"].append({"role":"assistant","content": live_answer})
            st.rerun()

        # ---------- FALLBACK TO EXISTING RAG / LLM FLOW ----------
        with st.spinner("Thinking..."):
            try:
                chroma_info = st.session_state.get("chroma_info")
                answer_text = ""
                if chroma_info and chroma_info.get("collection"):
                    collection = chroma_info["collection"]
                    docs = query_chroma_collection(collection, user_input, HF_EMBED_MODELS, k=3)
                    retrieved = "\n\n---\n\n".join([d.get("document","")[:1000] for d in docs])
                    mem = st.session_state.get("messages", [])[-memory_window:]
                    mem_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in mem])
                    prompt = f"Memory:\n{mem_text}\n\nContext:\n{retrieved}\n\nQuestion:\n{user_input}\n\nAnswer concisely."
                else:
                    prompt = f"Question:\n{user_input}\nAnswer concisely."
                groq_key = get_groq_key()
                if groq_key:
                    try:
                        r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"},
                                          json={"model": GROQ_MODEL_DEFAULT, "messages":[{"role":"user","content":prompt}], "max_tokens":512}, timeout=HF_TIMEOUT)
                        r.raise_for_status()
                        j = r.json()
                        choice = j.get("choices",[{}])[0]
                        text = ""
                        if isinstance(choice.get("message"), dict):
                            text = choice["message"].get("content") or ""
                        else:
                            text = choice.get("text") or ""
                        answer_text = text or str(j)
                    except Exception as e:
                        answer_text = f"[Groq generation error] {e}"
                else:
                    hf_token = get_hf_token()
                    if hf_token:
                        try:
                            url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL_FALLBACK}"
                            r = requests.post(url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": prompt}, timeout=HF_TIMEOUT)
                            r.raise_for_status()
                            out = r.json()
                            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                                answer_text = out[0]["generated_text"]
                            else:
                                answer_text = str(out)
                        except Exception as e:
                            answer_text = f"[HF text generation error] {e}"
                    else:
                        answer_text = "No Groq key or HF token available for generation."
                st.session_state["messages"].append({"role":"assistant","content": answer_text})
            except Exception as e:
                st.session_state["messages"].append({"role":"assistant","content": f"[Unhandled error] {e}"})
        st.rerun()

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
        st.info("No Chroma index found. Upload PDFs/TXT to enable RAG.")

    st.markdown("---")
    if uploaded_files and st.button("Index uploaded files (manual)"):
        with st.spinner("Indexing uploaded files..."):
            info = build_chroma_from_uploads(uploaded_files, HF_EMBED_MODELS, max_chunks=None if int(max_chunks) == 0 else int(max_chunks))
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

# End of file
