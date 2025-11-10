import os
import re
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFPlumberLoader

# === Paths ===
PDF_PATH = "Bhavna.pdf"
INDEX_PATH = "faiss.index"
METADATA_PATH = "metadata.pkl"

load_dotenv()

# === Load resume text ===
def load_pdf(pdf_path):
    loader = PDFPlumberLoader(pdf_path)
    pages = loader.load_and_split()
    return "\n".join(p.page_content for p in pages)

# === Contact Info Extraction ===
def extract_contact_info(text):
    contact_info = {}
    name = ""
    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phone_matches = re.findall(r'\+?\d{1,3}[-\s]?\d{7,}', text)

    # Name: first bullet under CONTACT
    match = re.search(r'CONTACT\s*●\s*([A-Za-z ]+)', text)
    if match:
        name = match.group(1).strip()
        
    # Extract Chatbot link (any streamlit.app URL)
    chatbot_match = re.search(r'https?://[^\s]*streamlit\.app[^\s]*', text)
    chatbot_link = chatbot_match.group(0).strip() if chatbot_match else ""
    
    # Extract LinkedIn link
    linkedin_match = re.search(r'https?://[^\s]*linkedin\.com[^\s]*', text)
    linkedin_link = linkedin_match.group(0).strip() if linkedin_match else ""

    contact_info["name"] = name
    contact_info["emails"] = list(set(email_matches))
    contact_info["phones"] = list(set(phone_matches))
    return contact_info

# === Current Job ===
def extract_current_position(text):
    lines = text.split('\n')
    current_job = {}

    for i, line in enumerate(lines):
        if "Present" in line and re.match(r'^\d+\.\s*(.+?),\s*(.+?),\s*(.+?)\s*\(([^)]*Present[^)]*)\)', line):
            match = re.match(r'^\d+\.\s*(.+?),\s*(.+?),\s*(.+?)\s*\(([^)]*Present[^)]*)\)', line)
            if match:
                current_job = {
                    'position': match.group(1).strip(),
                    'company': match.group(2).strip(),
                    'location': match.group(3).strip(),
                    'duration': match.group(4).strip(),
                    'responsibilities': []
                }
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('●'):
                    current_job['responsibilities'].append(lines[j].strip())
                    j += 1
                break
    return current_job

# === Summary Bio ===
def generate_summary():
    return (
        "Bhavna is an analytics professional with a Master in Business Analytics from the National University of Singapore and experience in data-driven problem solving across insurance and consulting. At Zurich Insurance and EXL, I built automated dashboards, streamlined reporting pipelines, and delivered insights that improved efficiency and business decision-making. With a strong foundation in machine learning from my academic work, I’m passionate about applying data, technology, and innovation to create practical business impact."
    )

# === Chunk Creator ===
def create_chunks_from_text(text):
    chunks = []
    metadata = []
    lines = text.split('\n')

    def add_chunk(section, title, content_lines):
        content = "\n".join(content_lines).strip()
        if content:
            chunks.append(content)
            metadata.append({
                "section": section,
                "title": title,
                "type": "section_content",
                "text": content
            })

    # Contact
    contact_info = extract_contact_info(text)
    contact_chunk = ["CONTACT DETAILS:"]
    if contact_info.get("name"):
        contact_chunk.append(f"Name: {contact_info['name']}")
    if contact_info.get("phones"):
        contact_chunk.append(f"Phone(s): {', '.join(contact_info['phones'])}")
    if contact_info.get("emails"):
        contact_chunk.append(f"Email(s): {', '.join(contact_info['emails'])}")
    linkedin_match = re.search(r'https://www\.linkedin\.com/in/[^\s)]+', text)
    if linkedin_match:
        contact_chunk.append(f"LinkedIn: {linkedin_match.group(0)}")
    add_chunk("CONTACT", "Contact Details", contact_chunk)

    # Current job
    job = extract_current_position(text)
    if job:
        job_lines = [
            "CURRENT EMPLOYMENT:",
            f"Position: {job['position']}",
            f"Company: {job['company']}",
            f"Location: {job['location']}",
            f"Duration: {job['duration']}",
            "Responsibilities:",
            *job['responsibilities']
        ]
        add_chunk("EXPERIENCE", "Current Job", job_lines)

    # Past jobs
    past_lines = []
    capture = False
    for line in lines:
        if "1. Business Analyst Intern" in line:
            capture = True
        elif "4. Data Science Intern" in line:
            capture = False
        if capture:
            past_lines.append(line.strip())
    add_chunk("EXPERIENCE", "Past Jobs", past_lines)

    # Education
    edu1, edu2 = [], []
    for i, line in enumerate(lines):
        if "1. Masters in Business Analytics" in line:
            edu1.append(line)
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('●'):
                edu1.append(lines[j].strip())
                j += 1
        if "2. Bachelors of Technology" in line:
            edu2.append(line)
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('●'):
                edu2.append(lines[j].strip())
                j += 1
    add_chunk("EDUCATION", "Current Education", edu1)
    add_chunk("EDUCATION", "Previous Education", edu2)

    # Leadership
    lead = []
    capture = False
    for line in lines:
        if "LEADERSHIP EXPERIENCE" in line.upper():
            capture = True
        elif "SKILLS & ACTIVITIES" in line.upper():
            break
        elif capture:
            lead.append(line.strip())
    add_chunk("LEADERSHIP", "Leadership Roles", lead)

    # Skills
    skills = []
    capture = False
    for line in lines:
        if "SKILLS & ACTIVITIES" in line.upper():
            capture = True
        elif "CERTIFICATIONS" in line.upper():
            break
        elif capture:
            skills.append(line.strip())
    add_chunk("SKILLS", "Skills Summary", skills)

    # Certifications
    certs = []
    for i, line in enumerate(lines):
        if "CERTIFICATIONS" in line.upper():
            j = i + 1
            while j < len(lines) and lines[j].strip():
                certs.append(lines[j].strip())
                j += 1
            break
    add_chunk("CERTIFICATIONS", "Certifications", certs)

    # Summary bio
    summary = generate_summary()
    add_chunk("SUMMARY", "Professional Bio", [summary])

    # Full resume fallback
    add_chunk("FULL", "Full Resume", lines)

    print(f"✅ Created {len(chunks)} chunks.")
    return chunks, metadata

# === FAISS Index Builder ===
def build_and_save_index(chunks, metadata):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index and metadata saved.")

# === Run ===
if __name__ == "__main__":
    print("🚀 Processing resume...")
    text = load_pdf(PDF_PATH)
    chunks, metadata = create_chunks_from_text(text)
    build_and_save_index(chunks, metadata)
    print("🎯 Done.")