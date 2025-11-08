# agents.py
import os, json, pickle, faiss, numpy as np, random
from typing import List, Dict
from langchain import OpenAI  # if using OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LCOpenAI

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
    from langchain.llms.fake import FakeListLLM
    llm = FakeListLLM(responses=["Fake response — configure OPENAI_API_KEY to use real LLM"])

# simple semantic retriever
def semantic_retrieve(query, k=6, embed_fn=None):
    if embed_fn is None:
        # use OpenAI if available
        if USE_OPENAI:
            from langchain.embeddings.openai import OpenAIEmbeddings
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

# prompt templates
NUMBER_THEORIST_PROMPT = PromptTemplate(
    input_variables=["query","evidence"],
    template="""
You are the Number Theorist Agent.

Task: Given the user query and the retrieved evidence below, propose up to 5 concise conjectures in LaTeX. For each conjecture provide:
1) statement_tex (LaTeX),
2) one-sentence intuition,
3) a small numeric check summary (range tested),
4) related references (list chunk ids or source files),
5) suggested proof approaches.

Evidence (chunks):
{evidence}

Begin. Be precise and conservative; do not fabricate theorems or references. If you lack evidence, say NO_EVIDENCE.
"""
)

EXPERIMENTER_PROMPT = PromptTemplate(
    input_variables=["statement","evidence"],
    template="""
You are the Experimenter Agent.

Task: Given the proposed conjecture (LaTeX) and evidence, produce:
1) reproducible code (Sage/PARI/Python) to test it,
2) expected runtime and resource estimate,
3) prioritized test plan (small N checks, randomized sampling, modulus checks),
4) if quick tests find counterexamples, show one.

Evidence:
{evidence}
Proposed statement:
{statement}
"""
)

SYMBOLIC_PROMPT = PromptTemplate(
    input_variables=["statement","evidence"],
    template="""
You are the Symbolic Agent.

Task: Given a conjectural statement in LaTeX and evidence, attempt symbolic simplification or reduction to known lemmas. Output:
1) a short 'reduction' explaining which known theorems/lemmas may apply (cite chunk ids),
2) algorithmic steps to attempt a proof in Sage/Lean or PARI,
3) confidence and which steps can be formalized automatically.

Evidence:
{evidence}
Statement:
{statement}
"""
)

COORDINATOR_PROMPT = PromptTemplate(
    input_variables=["proposals"],
    template="""
You are the Coordinator.

Task: Aggregate the following proposal objects (JSON lines), score them for novelty and interest, propose which to prioritize for experiments, and produce a single recommended candidate with a plan.

Proposals:
{proposals}
"""
)

# Agent runners
def run_number_theorist(query, evidence_chunks):
    evidence_text = "\n\n".join([c["text"][:800] for c in evidence_chunks])
    prompt = NUMBER_THEORIST_PROMPT.format(query=query, evidence=evidence_text)
    out = llm(prompt)
    return out

def run_experimenter(statement, evidence_chunks):
    ev = "\n\n".join([c["text"][:500] for c in evidence_chunks])
    prompt = EXPERIMENTER_PROMPT.format(statement=statement, evidence=ev)
    out = llm(prompt)
    return out

def run_symbolic(statement, evidence_chunks):
    ev = "\n\n".join([c["text"][:500] for c in evidence_chunks])
    prompt = SYMBOLIC_PROMPT.format(statement=statement, evidence=ev)
    out = llm(prompt)
    return out

def run_coordinator(proposals_json):
    prompt = COORDINATOR_PROMPT.format(proposals=proposals_json)
    out = llm(prompt)
    return out
