import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np
import time
import openai
from io import BytesIO
import requests

# === Streamlit Config ===
st.set_page_config(page_title="BhavBot - Bhavna's Resume Bot", page_icon="🤖")

# === Constants ===
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.pkl"
MAX_REQUESTS_PER_HOUR = 10
RATE_LIMIT_KEY = "rate_limit"

# === OpenAI API Key ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === Load API Key ===
if not openai.api_key:
    st.error("OpenAI API key not found. Please add it in Streamlit Secrets or environment variables.")
    st.stop()


# === General query detection ===
def is_general_query(query):
    general_phrases = ["hi", "hello", "hey", "how are you", "what’s up", "who are you", "tell me about yourself", "what is your name", "name", "owner", "who is your owner", "who created you", "who made you"]
    return query.strip().lower() in general_phrases

def is_internship_query(query: str) -> bool:
    keywords = ["intern", "internship", "interned", "trainee"]
    return any(word in query.lower() for word in keywords)

def is_linkedin_query(query: str) -> bool:
    keywords = ["linkedin", "linked in", "profile"]
    return any(k in query.lower() for k in keywords)

# === Load FAISS index ===
#@st.cache_resource(show_spinner=False)
def load_faiss_index():
    print("🔧 Loading FAISS index...")
    t0 = time.time()
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded in {time.time() - t0:.2f} sec")
    return index

# === Load metadata ===
#@st.cache_resource(show_spinner=False)
def load_metadata():
    print("📚 Loading metadata...")
    t0 = time.time()
    with open(METADATA_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Metadata loaded in {time.time() - t0:.2f} sec. Entries: {len(data)}")
    return data


# === Load embedding model ===
#@st.cache_resource(show_spinner=False)
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
def similarity_search(query: str, k: int = 10):
    print("🔍 Running similarity search...")
    t0 = time.time()
    query_embedding = embedding_model.encode([query])
   # query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    scores, indices = faiss_index.search(query_embedding.astype("float32"), k=k)

    retrieved_chunks = []
    retrieved_scores = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata_store):
            text = metadata_store[idx].get("text")
            if isinstance(text, str) and text.strip():
                retrieved_chunks.append(text)
                retrieved_scores.append(score)


    # ✅ If query is education-related, append all university chunks
    if any(word in query.lower() for word in ["education", "university", "degree", "college"]):
        edu_chunks = [m["text"] for m in metadata_store if "university" in m["text"].lower() or "bachelor" in m["text"].lower()]
        for ec in edu_chunks:
            if ec not in retrieved_chunks:
                retrieved_chunks.append(ec)
                retrieved_scores.append(0.0)  # Dummy score
        print("🎓 Education-related query detected — merged all university chunks.")

    print(f"✅ Retrieved {len(retrieved_chunks)} chunks in {time.time() - t0:.2f}s")

    return retrieved_chunks, retrieved_scores, indices

def get_forced_internship_chunks():
    priority_companies = ["zurich", "zenatix"]
    chunks = []

    for company in priority_companies:
        for m in metadata_store:
            text = m.get("text", "")
            if isinstance(text, str) and company in text.lower():
                chunks.append(text)

    return chunks


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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://raw.githubusercontent.com/UnicornBhavna/Chatbots/main/pic.jpeg", caption="Bhavna Lal", width=180)

# Inject custom CSS for full-page background color

page_bg = """
<style>
    body {
        background-color: #FFE4E1; /* baby light pink */
    }
    [data-testid="stAppViewContainer"] {
        background-color: #FFE4E1 !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFE4E1 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFE4E1 !important;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


with st.spinner("⏳ Loading FAISS index..."):
    faiss_index = load_faiss_index()

with st.spinner("⏳ Loading metadata..."):
    metadata_store = load_metadata()

# ✅ DEBUG — put it HERE
st.write("FAISS vectors:", faiss_index.ntotal)
st.write("Metadata entries:", len(metadata_store))

with st.spinner("⏳ Loading embedding model (may take time on first run)..."):
    try:
        embedding_model = load_model()
    except Exception as e:
        st.error("❌ Failed to load embedding model. Please try again later.")
        st.stop()

# === UI ===
st.title("👩‍💻 BhavBot - Bhavna's Resume Chatbot")
st.markdown("Ask about Bhavna's experience, education, skills, or leadership roles. 💡 *Tip: I’m BhavBot, your friendly resume assistant!*")

st.info(
    "🔗 **Connect with Bhavna on LinkedIn:** "
    "[linkedin.com/in/bhavna-lal](https://www.linkedin.com/in/bhavna-lal/)"
)


query = st.text_input("📨 Ask a question about Bhavna's resume:")

if query:

    if not check_rate_limit():
        st.warning(f"⚠️ You’ve hit the limit of {MAX_REQUESTS_PER_HOUR} questions/hour. Please wait and try again later.")

    elif is_linkedin_query(query):
        st.markdown("### ✅ Answer:")
        
        st.markdown("🔗 You can connect with Bhavna on LinkedIn here:\n\n"
            "[https://www.linkedin.com/in/bhavna-lal/](https://www.linkedin.com/in/bhavna-lal/)")


    elif is_general_query(query):
        with st.spinner("💬 Generating a friendly response..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are BhavBot, a friendly AI assistant who answers questions about Bhavna's resume and introduces yourself as BhavBot when asked who you are or what your name is. If asked for your owner, reply with 'Bhavna'."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown("### ✅ Answer:")
        st.write(answer)

    else:
        with st.spinner("🔍 Searching relevant resume snippets..."):
            if is_internship_query(query):
                matched_chunks = get_forced_internship_chunks()
                if not matched_chunks:
                    context = ""
                else:
                    context = "\n\n---\n\n".join(matched_chunks)
            else:
                matched_chunks, scores, indices = similarity_search(query)
                context = "\n\n---\n\n".join(matched_chunks)
        with st.spinner("✍️ Generating answer..."):
            try:
                prompt = f"""You are BhavBot, Bhavna's AI resume assistant.

You ONLY answer using the provided resume snippets.

You MUST list ALL internship experiences found in the resume snippets.
Do NOT merge them.
Do NOT omit any internship.
Each internship must be listed separately with company name.

If asked who you are, you introduce yourself as BhavBot, Bhavna's friendly AI assistant.  
Do not make up information.

--- Resume Snippets ---
{context}

--- Question ---
{query}

If the answer is not found in the resume, reply: "This information is not available in the resume. Please contact Bhavna directly for more details."
"""
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are BhavBot, a friendly AI assistant who answers based only on Bhavna's resume."},
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

#        with st.expander("📄 Show Resume Snippets Used"):
#            for chunk, score in zip(matched_chunks, scores):
#                st.markdown(f"**Score**: {score:.4f}")
#                st.code(chunk)



## Download PDF

url = "https://raw.githubusercontent.com/UnicornBhavna/Chatbots/main/Bhavna.pdf"

# Fetch PDF bytes
response = requests.get(url)

if response.status_code == 200:
    pdf_bytes = BytesIO(response.content)

    st.download_button(
        label="📄 Download Resume (PDF)",
        data=pdf_bytes,
        file_name="Bhavna_Resume.pdf",
        mime="application/pdf"
    )
else:
    st.error("⚠️ Could not load the PDF from GitHub. Please check the URL.")
