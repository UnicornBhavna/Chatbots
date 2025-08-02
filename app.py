import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np
import time
import openai

# === Streamlit Config ===
st.set_page_config(page_title="Bhavna's Resume Bot", page_icon="📄")

# === Constants ===
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.pkl"
MAX_REQUESTS_PER_HOUR = 10
RATE_LIMIT_KEY = "rate_limit"

# === OpenAI API Key ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === General query detection ===
def is_general_query(query):
    general_phrases = ["hi", "hello", "hey", "how are you", "what’s up", "who are you", "tell me about yourself"]
    return query.strip().lower() in general_phrases

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

# === Similarity Search with Education Fallback ===
def similarity_search(query: str, k: int = 5):
    print("🔍 Running similarity search...")
    t0 = time.time()
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    scores, indices = faiss_index.search(query_embedding.astype("float32"), k=k)
    
    retrieved_chunks = []
    retrieved_scores = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata_store):
            retrieved_chunks.append(metadata_store[idx].get("text", ""))
            retrieved_scores.append(score)

    # ✅ If query is education-related, append all university chunks
    if any(word in query.lower() for word in ["education", "university", "degree", "college"]):
        edu_chunks = [m["text"] for m in metadata_store if "university" in m["text"].lower() or "bachelor" in m["text"].lower()]
        for ec in edu_chunks:
            if ec not in retrieved_chunks:
                retrieved_chunks.append(ec)
                retrieved_scores.append(0.0)  # Add dummy score for appended chunks
        print("🎓 Education-related query detected — merged all university chunks.")

    print(f"✅ Retrieved {len(retrieved_chunks)} chunks in {time.time() - t0:.2f}s")
    return retrieved_chunks, retrieved_scores, indices



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
    elif is_general_query(query):
        with st.spinner("💬 Generating a friendly response..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant who introduces Bhavna in a friendly tone."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown("### ✅ Answer:")
        st.write(answer)
        
    else:
        with st.spinner("🔍 Searching relevant resume snippets..."):
            matched_chunks, scores, indices = similarity_search(query)
            context = "\n\n---\n\n".join([chunk.get("text", "") for chunk in matched_chunks])

        with st.spinner("✍️ Generating answer..."):
            try:
                prompt = f"""You are Bhavna's resume assistant.

Use the following resume snippets to answer the question below. If the question is general (e.g., a greeting or summary), you may answer from your own knowledge or provide a friendly response. Prefer the resume content when relevant.

--- Resume Snippets ---
{context}

--- Question ---
{query}

If the answer is not found in the resume and is not general, reply: "This information is not under my scope, please ask me something else or google it yourself :D"
"""
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"❌ Error generating answer: {e}"
                print(f"Error during OpenAI call: {e}")

        st.markdown("### ✅ Answer:")
        st.write(answer)

        with st.expander("📄 Show Resume Snippets Used"):
            for chunk, score in zip(matched_chunks, scores):
                st.markdown(f"**Score**: {score:.4f}")
                st.code(chunk.get("text", ""))
