import pickle
import os

METADATA_FILE = "metadata.pkl"

def load_metadata(path):
    if not os.path.exists(path):
        print(f"❌ Metadata file not found at {path}")
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded metadata with {len(data)} entries")
    return data

def inspect_metadata(metadata):
    print("\n=== 🔍 Inspecting Metadata ===")
    for i, m in enumerate(metadata):
        text = m.get("text", "")
        print(f"\n[{i}] -----------------------------")
        print(text)

def search_for_universities(metadata):
    print("\n=== 🎓 Searching for University/Education Entries ===")
    edu_chunks = []
    for i, m in enumerate(metadata):
        text = m.get("text", "").lower()
        if "university" in text or "education" in text:
            print(f"\n[{i}] >>> MATCH <<<")
            print(m.get("text", ""))
            edu_chunks.append(m)
    if not edu_chunks:
        print("❌ No university or education entries found in metadata!")
    else:
        print(f"\n✅ Found {len(edu_chunks)} education-related chunks")

if __name__ == "__main__":
    metadata = load_metadata(METADATA_FILE)
    if metadata:
        inspect_metadata(metadata)
        search_for_universities(metadata)
