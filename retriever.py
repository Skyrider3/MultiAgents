# agents.py
import os, json, pickle, faiss, numpy as np, random
from typing import List, Dict
from langchain_openai import OpenAI as LCOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load faiss / metadata
INDEX_DIR = "index"
index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
with open(f"{INDEX_DIR}/meta.pkl","rb") as f:
    meta = pickle.load(f)
texts = meta["texts"]
metadatas = meta["metadatas"]
with open(f"{INDEX_DIR}/symbol_index.json","r") as f:
    symbol_index = json.load(f)

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", False))
if USE_OPENAI:
    llm = LCOpenAI(temperature=0.2, model_name="gpt-4o-mini")  # change as needed
else:
    # placeholder: you can swap other LLMs
    from langchain_community.llms.fake import FakeListLLM
    llm = FakeListLLM(responses=["Fake response — configure OPENAI_API_KEY to use real LLM"])

# simple semantic retriever
def semantic_retrieve(query, k=6, embed_fn=None):
    if embed_fn is None:
        # use OpenAI if available
        if USE_OPENAI:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings()
            v = emb.embed_query(query)
        else:
            # fallback to SBERT model reused from ingest - not packaged here; assume embed function exists
            from sentence_transformers import SentenceTransformer
            s_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            v = s_model.encode([query])[0]
    else:
        v = embed_fn(query)
    v = np.array(v).astype("float32")[None, :]
    D, I = index.search(v, k)
    hits = []
    for idx in I[0]:
        hits.append({"text": texts[idx], "meta": metadatas[idx]})
    return hits

# symbol retriever
def symbol_retrieve(symbols: List[str], max_chunks_per_symbol=6):
    hits = []
    for s in symbols:
        if s in symbol_index:
            ids = symbol_index[s][:max_chunks_per_symbol]
            for cid in ids:
                # find text by cid
                for i,md in enumerate(metadatas):
                    if md["chunk_id"] == cid:
                        hits.append({"text": texts[i], "meta": md})
                        break
    return hits