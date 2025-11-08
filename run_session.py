# run_session.py
import json, sys, time
from agents import semantic_retrieve, symbol_retrieve, run_number_theorist, run_experimenter, run_symbolic, run_coordinator
from langchain_openai import OpenAIEmbeddings


def expand_query_for_number_theory(q):
    # Add keyword heuristics
    extras = ["multiplicative", "Dirichlet", "mod p", "L-function", "elliptic", "congruence", "density"]
    return q + " " + " ".join(extras)

def session(query):
    print("Query:", query)
    qexp = expand_query_for_number_theory(query)
    sem_hits = semantic_retrieve(qexp, k=10)
    # symbol heuristics: look for latex-like tokens in query
    # crude extract
    syms = []
    import re
    syms += re.findall(r"\\[A-Za-z]+\{[^}]*\}", query)
    syms += re.findall(r"[a-zA-Z]_\{?[0-9nkp]+\}?", query)
    sym_hits = symbol_retrieve(syms)
    # merge evidence but keep provenance
    evidence = sem_hits + sym_hits
    print(f"Found {len(evidence)} evidence chunks.")
    # Number Theorist proposals
    proposals_text = run_number_theorist(query, evidence)
    print("=== Number Theorist output ===\n", proposals_text)
    # naive parse: split proposals by "statement_tex"
    # For demonstration, we'll create a single statement by extracting text between $$ or \( \)
    import re
    statements = re.findall(r"\$(.*?)\$", proposals_text, flags=re.S)
    if not statements:
        statements = [proposals_text.strip().split("\n")[0][:200]]
    results = []
    for st in statements[:3]:
        exp_out = run_experimenter(st, evidence)
        sym_out = run_symbolic(st, evidence)
        results.append({
            "statement": st,
            "experimenter": exp_out,
            "symbolic": sym_out
        })
    # coordinator
    coord = run_coordinator(json.dumps(results))
    out = {
        "proposals": results,
        "coordinator": coord
    }
    print("=== COORDINATOR ===\n", coord)
    # write to file
    ts = int(time.time())
    with open(f"session_{ts}.json","w") as f:
        json.dump(out, f, indent=2)
    print("saved to session file.")
    return out

def main():
    """Entry point for the run-session script."""
    q = "distribution of zeros of Ramanujan tau(n) modulo small primes"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    session(q)

if __name__ == "__main__":
    main()
