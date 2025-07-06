import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import openai
import os
import numpy as np
import time

# === Streamlit Config ===
st.set_page_config(page_title="Bhavna's Resume Bot", page_icon="📄")

# === Constants ===
INDEX_PATH = "faiss.index"
METADATA_PATH = "metadata.pkl"
MAX_REQUESTS_PER_HOUR = 5
RATE_LIMIT_KEY = "rate_limit"

# === Load OpenAI API Key securely ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === Load FAISS index locally ===
@st.cache_resource(show_spinner=False)
def load_faiss_index():
    return faiss.read_index(INDEX_PATH)

# === Load metadata locally ===
@st.cache_resource(show_spinner=False)
def load_metadata():
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)

# === Load embedding model ===
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# === Rate limiting ===
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

# === Load assets ===
with st.spinner("⏳ Loading FAISS index..."):
    faiss_index = load_faiss_index()
with st.spinner("⏳ Loading metadata..."):
    metadata_store = load_metadata()
with st.spinner("⏳ Loading embedding model..."):
    embedding_model = load_model()

# === UI ===
st.title("🤖 Bhavna's Resume Chatbot")
st.markdown("Ask about Bhavna's experience, education, skills, or leadership roles.")

query = st.text_input("📨 Ask a question about Bhavna's resume:")

if query:
    if not check_rate_limit():
        st.warning(f"⚠️ You’ve hit the limit of {MAX_REQUESTS_PER_HOUR} questions/hour. Please wait and try again later.")
    else:
        with st.spinner("🔍 Searching resume..."):
            query_embedding = embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            scores, indices = faiss_index.search(query_embedding.astype("float32"), k=3)

            matched_chunks = [
                metadata_store[idx].get("text", "")
                for idx in indices[0]
                if 0 <= idx < len(metadata_store)
            ]
            context = "\n\n---\n\n".join(matched_chunks)

        with st.spinner("✍️ Generating answer..."):
            try:
                prompt = f"""Using the following resume snippets, answer the question below.

--- Resume Snippets ---
{context}

--- Question ---
{query}

Respond factually based only on the resume content. If the answer is not found in the resume, reply: "This information is not available in the resume."
"""
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant answering questions based strictly on Bhavna's resume."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"❌ Error generating answer: {e}"

        st.markdown("### ✅ Answer:")
        st.write(answer)

        with st.expander("📄 Show Resume Snippets Used"):
            for chunk in matched_chunks:
                st.code(chunk)
