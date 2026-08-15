import streamlit as st
import tempfile
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="RAG-Powered Document Assistant", layout="wide")
st.title("📄 RAG‑Powered Document Assistant")
st.markdown("Upload a document and ask questions about it – with guardrails!")

# ---------- Load models (cached) ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.to("cpu")
    return tokenizer, model

embedder = load_embedder()
tokenizer, llm = load_llm()

def embed_text(text):
    return embedder.encode([text])[0].astype('float32')

def generate_answer(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer based on the context:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ---------- Guardrails ----------
def is_out_of_scope(query):
    keywords = ['medical', 'health', 'disease', 'cure', 'surgery', 'diagnosis']
    return any(k in query.lower() for k in keywords)

def grounded_score(answer, chunks, threshold=0.3):
    if not chunks:
        return False
    answer_emb = embed_text(answer)
    chunk_embs = [embed_text(c) for c in chunks]
    sims = [np.dot(answer_emb, ce) / (np.linalg.norm(answer_emb) * np.linalg.norm(ce)) for ce in chunk_embs]
    return np.mean(sims) > threshold

# ---------- State for vector store ----------
if 'index' not in st.session_state:
    st.session_state.index = None
if 'doc_store' not in st.session_state:
    st.session_state.doc_store = []

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload a text or PDF file", type=['txt', 'pdf'])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if st.button("Ingest Document"):
        with st.spinner("Indexing document..."):
            try:
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                texts = [chunk.page_content for chunk in chunks]
                embeddings = embedder.encode(texts)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(np.array(embeddings).astype('float32'))
                st.session_state.index = index
                st.session_state.doc_store = [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
                st.success(f"Ingested {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

# ---------- Query ----------
if st.session_state.index is not None:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question:
        if is_out_of_scope(question):
            st.warning("Out-of-scope: medical/health topics are not allowed.")
        else:
            q_emb = embed_text(question)
            distances, indices = st.session_state.index.search(np.array([q_emb]).astype('float32'), 3)
            retrieved = [st.session_state.doc_store[i] for i in indices[0] if i < len(st.session_state.doc_store)]
            if not retrieved:
                st.info("No relevant document sections found.")
            else:
                context = "\n".join([r["text"] for r in retrieved])
                answer = generate_answer(question, context)
                chunks = [r["text"] for r in retrieved]
                is_grounded = grounded_score(answer, chunks)
                sources = [r["metadata"].get("source", "unknown") for r in retrieved]
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Grounded")
                st.write("✅ Yes" if is_grounded else "⚠️ No (maybe hallucination)")
                st.subheader("Sources")
                for i, src in enumerate(sources):
                    st.write(f"{i+1}. {src}")
else:
    st.info("Please upload and ingest a document first.")