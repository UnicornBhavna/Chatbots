import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import openai
import os
import numpy as np
import time
from openai import OpenAI

# === Streamlit Config ===
st.set_page_config(page_title="Bhavna's Resume Bot", page_icon="📄")

# === Constants ===
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.pkl"
MAX_REQUESTS_PER_HOUR = 5
RATE_LIMIT_KEY = "rate_limit"

# === OpenAI API Key ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Load FAISS index ===
@st.cache_resource(show_spinner=False)
def load_faiss_index():
    print("🔧 Loading FAISS index...")
    t0 = time.time()
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded in {time.time() - t0:.2f} sec")
    return index

# === Load metadata ===
@st.cache_resource(show_spinner=False)
def load_metadata():
    print("📚 Loading metadata...")
    t0 = time.time()
    with open(METADATA_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Metadata loaded in {time.time() - t0:.2f} sec. Entries: {len(data)}")
    return data

# === Load embedding model ===
@st.cache_resource(show_spinner=False)
def load_model():
    print("🧠 Loading embedding model...")
    t0 = time.time()
    try:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        print(f"✅ Model loaded in {time.time() - t0:.2f} sec")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise RuntimeError("Failed to load SentenceTransformer model.")

# === Similarity Search ===
def similarity_search(query: str, k: int = 3):
    print("🔍 Running similarity search...")
    t0 = time.time()
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = faiss_index.search(query_embedding.astype("float32"), k=k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(metadata_store):
            results.append(metadata_store[i])
    print(f"✅ Search done in {time.time() - t0:.2f} sec")
    return results, scores[0], indices[0]

# === Rate Limiting ===
def check_rate_limit():
    now = time.time()
    data = st.session_state.get(RATE_LIMIT_KEY, {"count": 0, "start_time": now})
    if now - data["start_time"] > 3600:
        data = {"count": 0, "start_time": now}
    if data["count"] >= MAX_REQUESTS_PER_HOUR:
        return False
    data["count"] += 1
    st.session_state[RATE_LIMIT_KEY] = data
    return True

# === Load all resources ===
with st.spinner("⏳ Loading FAISS index..."):
    faiss_index = load_faiss_index()

with st.spinner("⏳ Loading metadata..."):
    metadata_store = load_metadata()

with st.spinner("⏳ Loading embedding model (may take time on first run)..."):
    try:
        embedding_model = load_model()
    except Exception as e:
        st.error("❌ Failed to load embedding model. Please try again later.")
        st.stop()

# === UI ===
st.title("🤖 Bhavna's Resume Chatbot")
st.markdown("Ask about Bhavna's experience, education, skills, or leadership roles.")

query = st.text_input("📨 Ask a question about Bhavna's resume:")

if query:
    if not check_rate_limit():
        st.warning(f"⚠️ You’ve hit the limit of {MAX_REQUESTS_PER_HOUR} questions/hour. Please wait and try again later.")
    else:
        with st.spinner("🔍 Searching relevant resume snippets..."):
            matched_chunks, scores, indices = similarity_search(query)
            context = "\n\n---\n\n".join([chunk.get("text", "") for chunk in matched_chunks])

        with st.spinner("✍️ Generating answer..."):
            try:
                t0 = time.time()
                prompt = f"""Using the following resume snippets, answer the question below.

--- Resume Snippets ---
{context}

--- Question ---
{query}

Respond factually based only on the resume content. If the answer is not found in the resume, reply: "This information is not available in the resume."
"""
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant..."},
                        {"role": "user", "content": prompt}
                        ],
                    max_tokens=500,
                    temperature=0.2
                    )
                answer = response.choices[0].message.content.strip()
                print(f"🧠 OpenAI response generated in {time.time() - t0:.2f} sec")
            except Exception as e:
                answer = f"❌ Error generating answer: {e}"
                print(f"Error during OpenAI call: {e}")

        st.markdown("### ✅ Answer:")
        st.write(answer)

        with st.expander("📄 Show Resume Snippets Used"):
            for chunk, score in zip(matched_chunks, scores):
                st.markdown(f"**Score**: {score:.4f}")
                st.code(chunk.get("text", ""))