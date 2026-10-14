import numpy as np

def mean_score(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None

def retrieval_precision_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    top = retrieved_ids[:k]
    return sum(x in relevant for x in top) / max(k, 1)

def simple_advisory_score(advisory, required_terms):
    text = advisory.lower()
    hits = sum(1 for t in required_terms if t.lower() in text)
    return hits / max(len(required_terms), 1)
