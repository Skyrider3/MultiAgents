# ingest.py
"""
Process LaTeX source files from arXiv papers organized by topic.
Creates separate FAISS indices for each topic category.
"""
import os, json, re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
TOPIC_TABLE_PATH = Path("/Users/CP/Documents/MathMind/papers/topicTable.txt")
BASE_PAPERS_DIR = Path("/Users/CP/Documents/MathMind/papers")

# choose embedding model: local SBERT or OpenAI
USE_OPENAI_EMBED = bool(os.getenv("OPENAI_API_KEY", False))

if USE_OPENAI_EMBED:
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings()
else:
    # smaller, fast SBERT
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    def embed_texts(texts):
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embedder = None

def read_categories(topic_file):
    """Read category codes and names from topicTable.txt"""
    categories = []
    with open(topic_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    code = parts[0].strip()
                    name = parts[1].strip()
                    categories.append((code, name))
    return categories

def extract_latex_symbols(text):
    """Extract LaTeX mathematical symbols and commands from text"""
    patterns = [
        r"\\[A-Za-z]+(?:\{[^}]*\})?",  # LaTeX commands
        r"[A-Za-z]\_\{?[0-9nkp]+\}?",  # subscripts like a_n, x_0
        r"\\pmod\{[^}]+\}",
        r"\bmod\b\s*\d+",
        r"\b[A-Za-z]{1,3}\([nxs]\)"
    ]
    syms = set()
    for p in patterns:
        for m in re.findall(p, text):
            syms.add(m)
    return list(syms)

def load_tex_file(tex_path):
    """Load a .tex file and return its contents"""
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"    Warning: Could not read {tex_path.name}: {e}")
        return None

def load_and_chunk_tex(tex_path):
    """Load a .tex file and chunk it for indexing"""
    content = load_tex_file(tex_path)
    if not content:
        return []
    
    # RecursiveCharacterTextSplitter tries separators in order
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )
    chunks = splitter.split_text(content)
    
    # attach metadata
    md_chunks = []
    for i, c in enumerate(chunks):
        md = {
            "source": str(tex_path),
            "filename": tex_path.name,
            "chunk_id": f"{tex_path.stem}__{i}",
            "file_type": "tex"
        }
        md_chunks.append((c, md))
    return md_chunks

def find_tex_files(topic_dir):
    """Find all .tex files in the latex_sources subdirectories"""
    latex_sources_dir = topic_dir / "latex_sources"
    if not latex_sources_dir.exists():
        return []
    
    # Find all .tex files recursively
    tex_files = list(latex_sources_dir.rglob("*.tex"))
    return tex_files

def build_topic_index(topic_code, topic_name, topic_dir):
    """Build FAISS index for a specific topic"""
    
    print(f"\n{'='*70}")
    print(f"Processing Topic: {topic_name} ({topic_code})")
    print(f"{'='*70}")
    
    # Find all .tex files
    tex_files = find_tex_files(topic_dir)
    
    if not tex_files:
        print(f"  ⚠ No .tex files found in {topic_dir}")
        return False
    
    print(f"  Found {len(tex_files)} .tex file(s)")
    
    texts, metadatas = [], []
    symbol_index = {}  # symbol -> list of chunk_ids
    
    # Process each .tex file
    processed_files = 0
    for tex_file in tex_files:
        chunks = load_and_chunk_tex(tex_file)
        if chunks:
            processed_files += 1
            for chunk, md in chunks:
                texts.append(chunk)
                metadatas.append(md)
                # extract symbols
                syms = extract_latex_symbols(chunk)
                for s in syms:
                    symbol_index.setdefault(s, []).append(md["chunk_id"])
    
    if not texts:
        print(f"  ⚠ No content extracted from .tex files")
        return False
    
    print(f"  Processed {processed_files} files")
    print(f"  Created {len(texts)} chunks")
    
    # Generate embeddings
    print(f"  Generating embeddings...")
    if USE_OPENAI_EMBED:
        embeddings = embedder.embed_documents(texts)
        embeddings = np.array(embeddings)
    else:
        embeddings = embed_texts(texts)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    # Create index directory within the topic folder
    index_dir = topic_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Save everything
    faiss.write_index(index, str(index_dir / "faiss.index"))
    
    with open(index_dir / "meta.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    
    with open(index_dir / "symbol_index.json", "w") as f:
        json.dump(symbol_index, f, indent=2)
    
    print(f"  ✓ Successfully built index in {index_dir}")
    print(f"    - FAISS index: {len(texts)} vectors, dimension {dim}")
    print(f"    - Symbol index: {len(symbol_index)} unique symbols")
    
    return True

def main():
    """Main function to process all topics"""
    
    print("="*70)
    print("LaTeX Multi-Topic Indexer")
    print("="*70)
    
    # Read categories from topicTable.txt
    print(f"\n📖 Reading topics from: {TOPIC_TABLE_PATH}")
    
    if not TOPIC_TABLE_PATH.exists():
        print(f"  ✗ Error: Topic table not found at {TOPIC_TABLE_PATH}")
        return False
    
    categories = read_categories(TOPIC_TABLE_PATH)
    print(f"   Found {len(categories)} topics")
    
    # Process each topic
    success_count = 0
    failed_count = 0
    
    for code, name in categories:
        topic_dir = BASE_PAPERS_DIR / name
        
        if not topic_dir.exists():
            print(f"\n⚠ Warning: Directory not found for {name} ({code})")
            print(f"  Expected: {topic_dir}")
            failed_count += 1
            continue
        
        # Build index for this topic
        success = build_topic_index(code, name, topic_dir)
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 INDEXING COMPLETED!")
    print("="*70)
    print(f"Topics processed: {len(categories)}")
    print(f"Successful: {success_count}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Base directory: {BASE_PAPERS_DIR}")
    print("="*70)
    
    return success_count > 0

if __name__ == "__main__":
    main()
