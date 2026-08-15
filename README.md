# 📄 RAG‑Powered Document Assistant

A production‑oriented **Retrieval‑Augmented Generation (RAG)** app that ingests documents (TXT or PDF), chunks and embeds them, then answers questions with retrieval‑based grounding. Built with **Streamlit**, **Sentence‑Transformers**, **FAISS**, and **Flan‑T5**.

---

## 🚀 Live Demo

**[👉 Click here to run the live demo](https://rag-powered-document-assistant-lacznrsbtefxju6js7pbeh.streamlit.app/)**  

## ✨ Features

- **Document Ingestion** – Upload TXT or PDF files; the app chunks, embeds, and indexes them with FAISS.
- **Question Answering** – Ask questions about the document; the app retrieves relevant chunks and generates a grounded answer using Flan‑T5‑small.
- **Guardrails**:
  - ✅ **Out‑of‑scope detection** – blocks medical/health queries.
  - ✅ **Hallucination risk** – checks if the answer is semantically grounded in the retrieved text.
- **Source Attribution** – Shows which document sections the answer is based on.
- **Lightweight & Fast** – Uses CPU‑only models suitable for free hosting.

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| Embeddings | [Sentence‑Transformers (all‑MiniLM‑L6‑v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector Store | [FAISS (CPU)](https://github.com/facebookresearch/faiss) |
| LLM (answer generation) | [Flan‑T5‑small](https://huggingface.co/google/flan-t5-small) |
| Text Splitting | `langchain-text-splitters` |
| Deployment | [Streamlit Cloud](https://streamlit.io/cloud) |
| Language | Python 3.9+ |

---

## 📂 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Dhayalramesh/RAG-Powered-Document-Assistant.git
   cd RAG-Powered-Document-Assistant
2.pip install -r requirements.txt
3.streamlit run app.py
