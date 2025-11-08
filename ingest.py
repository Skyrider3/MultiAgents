# ingest.py
import os, json, re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# Removed unused imports - using SentenceTransformer directly for local embeddings
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PAPERS_DIR = Path("papers")
OUT_DIR = Path("index")
OUT_DIR.mkdir(exist_ok=True)

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

def extract_latex_symbols(text):
    # crude regex for things like \tau(n), a_n, \zeta(s) and congruences like \pmod{p}
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

def load_and_chunk(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    doc = loader.load()
    # try to keep math blocks together by splitting on two linebreaks and page boundaries
    raw_text = "\n\n".join([d.page_content for d in doc])
    # RecursiveCharacterTextSplitter tries separators in order: \n\n, \n, " ", then character-by-character
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )
    chunks = splitter.split_text(raw_text)
    # attach metadata
    md_chunks = []
    for i,c in enumerate(chunks):
        md = {"source": str(pdf_path), "chunk_id": f"{pdf_path.stem}__{i}"}
        md_chunks.append((c, md))
    return md_chunks

def build_indices(pdf_dir=PAPERS_DIR, out_dir=OUT_DIR):
    # Ensure the papers directory exists
    pdf_dir.mkdir(exist_ok=True)
    
    texts, metadatas = [], []
    symbol_index = {}  # symbol -> list of chunk_ids
    
    # Check if there are any PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print(f"Please add research papers (PDFs) to the '{pdf_dir}' directory and run this script again.")
        return False
    
    print(f"Processing {len(pdf_files)} PDF file(s)...")
    for pdf in pdf_files:
        print(f"  Loading {pdf.name}...")
        for chunk, md in load_and_chunk(pdf):
            texts.append(chunk)
            metadatas.append(md)
            # extract symbols
            syms = extract_latex_symbols(chunk)
            for s in syms:
                symbol_index.setdefault(s, []).append(md["chunk_id"])
    
    print(f"Created {len(texts)} chunks from {len(pdf_files)} paper(s)")
    
    # embeddings
    if USE_OPENAI_EMBED:
        embeddings = embedder.embed_documents(texts)
        embeddings = np.array(embeddings)  # Convert to numpy array for FAISS
    else:
        embeddings = embed_texts(texts)
    
    # build faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))  # FAISS requires float32
    
    # save everything
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    with open(out_dir / "symbol_index.json", "w") as f:
        json.dump(symbol_index, f, indent=2)
    
    print(f"Successfully built indices in {out_dir}")
    print(f"  - FAISS index: {len(texts)} vectors, dimension {dim}")
    print(f"  - Symbol index: {len(symbol_index)} unique symbols")
    return True

if __name__ == "__main__":
    build_indices()
