import requests
import chromadb
from pathlib import Path
from docling.document_converter import DocumentConverter

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Initialize local Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Initialize Docling converter for PDFs, DOCX, etc.
converter = DocumentConverter()

def get_embedding(text):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": text})
    return r.json()["embeddings"][0]

def extract_markdown(file_path: Path) -> str:
    """Reads Markdown directly or converts PDF/DOCX into structured Markdown."""
    suffix = file_path.suffix.lower()
    
    if suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    elif suffix in [".pdf", ".docx"]:
        print(f"   📄 Converting {file_path.name} to Markdown via Docling...")
        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()
    else:
        return ""

def chunk_markdown_by_headers(file_text, file_name):
    """Splits markdown by headers, keeping context glued together."""
    lines = file_text.splitlines()
    chunks = []
    
    current_section = file_name
    current_chunk_lines = []

    for line in lines:
        if line.startswith("#"):
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(f"Source: {file_name} > {current_section}\n\n{chunk_text}")
                current_chunk_lines = []
            
            current_section = line.lstrip("#").strip()
        else:
            current_chunk_lines.append(line)

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(f"Source: {file_name} > {current_section}\n\n{chunk_text}")

    return chunks

def index_documents():
    print("Scanning 'docs/' folder for .md, .pdf, and .docx files...")
    all_chunks = []
    all_ids = []
    all_metadatas = []

    supported_extensions = {".md", ".pdf", ".docx"}
    files = [f for f in Path("docs").iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

    for file in files:
        markdown_text = extract_markdown(file)
        if not markdown_text.strip():
            continue

        file_chunks = chunk_markdown_by_headers(markdown_text, file.name)
        
        for i, chunk in enumerate(file_chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file.name}_chunk_{i}")
            all_metadatas.append({"source": file.name})

    if not all_chunks:
        print("No supported documents found in docs/.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = [get_embedding(c) for c in all_chunks]

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )
    
    print(f"✅ Successfully synced {len(all_chunks)} chunks across {len(files)} files!")
    print(f"Total chunks currently in database: {collection.count()}")

if __name__ == "__main__":
    index_documents()