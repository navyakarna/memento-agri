"""Semantic case retrieval with outcome-aware reranking.

Embeddings: sentence-transformers (config.EMBEDDING_MODEL) when installed,
otherwise a stateless scikit-learn HashingVectorizer, so the pipeline still
runs without PyTorch.
Index: FAISS inner-product index when installed, otherwise a NumPy matrix.
The backend used to build the memory is stored in the metadata file and
reused at load time so queries and cases are always embedded the same way.
"""

import json
from pathlib import Path

import numpy as np

# FAISS must be imported before PyTorch (pulled in by sentence-transformers):
# on macOS the reverse order loads two OpenMP runtimes and index.search()
# segfaults.
try:
    import faiss
except ImportError:
    faiss = None

from config import EMBEDDING_MODEL, TOP_K
from memory.case_bank import case_text

# final_score = text similarity + rainfall-anomaly match + observed outcome.
# The outcome term is deliberately small: it breaks ties between similar
# situations rather than overriding climate similarity.
SIMILARITY_WEIGHT = 0.50
CLIMATE_WEIGHT = 0.35
OUTCOME_WEIGHT = 0.15
EXPERIENCE_WEIGHT = 0.1
CLIMATE_SCALE = 0.20  # a 20-point rainfall-anomaly gap -> match of e^-1
CANDIDATE_MULTIPLIER = 25


# ------------------------------------------------------------
# Embedding backends
# ------------------------------------------------------------

class _SentenceEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        vectors = self.model.encode(
            texts, batch_size=256, show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype="float32")


class _HashingEmbedder:
    name = "hashing"

    def __init__(self, model_name=None):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model_name = "hashing-vectorizer"
        self.vectorizer = HashingVectorizer(
            n_features=2 ** 12, ngram_range=(1, 2), alternate_sign=False, norm="l2"
        )

    def encode(self, texts):
        return self.vectorizer.transform(texts).toarray().astype("float32")


def _make_embedder(backend=None, model_name=EMBEDDING_MODEL):
    if backend in (None, _SentenceEmbedder.name):
        try:
            return _SentenceEmbedder(model_name)
        except Exception as e:
            if backend == _SentenceEmbedder.name:
                raise RuntimeError(
                    "Memory was built with sentence-transformers, which cannot be "
                    "loaded now. Install it or rebuild with `python main.py build-memory`."
                ) from e
            print(f"sentence-transformers unavailable ({e}); using hashing embeddings.")
    return _HashingEmbedder()


# ------------------------------------------------------------
# Vector index backends
# ------------------------------------------------------------

def _save_index(vectors, index_path):
    if faiss is not None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        return "faiss"
    # Write through a file handle so NumPy keeps the configured file name.
    with open(index_path, "wb") as f:
        np.save(f, vectors)
    return "numpy"


class _NumpyIndex:
    def __init__(self, vectors):
        self.vectors = vectors
        self.ntotal = len(vectors)

    def search(self, queries, k):
        scores = queries @ self.vectors.T
        k = min(k, self.ntotal)
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        return np.take_along_axis(scores, idx, axis=1), idx


def _load_index(index_path, kind):
    if kind == "faiss":
        if faiss is None:
            raise RuntimeError(
                "Memory index was built with FAISS, which is not installed. "
                "Install faiss-cpu or rebuild with `python main.py build-memory`."
            )
        return faiss.read_index(str(index_path))
    with open(index_path, "rb") as f:
        return _NumpyIndex(np.load(f))


# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

def build_index(cases, index_path, metadata_path):
    """Embed every case and save the vector index plus case metadata."""
    if not cases:
        raise ValueError("No cases to index.")
    embedder = _make_embedder()
    texts = [c.get("text") or case_text(c.get("context", {})) for c in cases]
    print(f"Embedding {len(texts)} cases with {embedder.model_name} ...")
    vectors = embedder.encode(texts)
    index_kind = _save_index(vectors, index_path)
    metadata = {
        "embedding_backend": embedder.name,
        "embedding_model": embedder.model_name,
        "index_backend": index_kind,
        "cases": cases,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    return len(cases)


# ------------------------------------------------------------
# Retrieve
# ------------------------------------------------------------

def _outcome_score(case):
    """Map the observed yield anomaly to 0-1 (0.5 = yield at baseline)."""
    value = case.get("outcome", {}).get("yield_anomaly")
    if value is None:
        return 0.5
    return (max(-1.0, min(1.0, float(value))) + 1.0) / 2.0


def _climate_match(query_context, case):
    """1.0 for identical rainfall anomaly, decaying with the gap; 0.5 if unknown."""
    q = query_context.get("rainfall_anomaly")
    c = case.get("context", {}).get("rainfall_anomaly")
    try:
        return float(np.exp(-abs(float(q) - float(c)) / CLIMATE_SCALE))
    except (TypeError, ValueError):
        return 0.5


class CaseRetriever:
    def __init__(self, index_path, metadata_path, top_k=TOP_K):
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Memento memory not found. Run `python main.py build-memory` first."
            )
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.cases = metadata["cases"]
        self.top_k = top_k
        self.index = _load_index(index_path, metadata.get("index_backend", "faiss"))
        self.embedder = _make_embedder(
            metadata.get("embedding_backend"),
            metadata.get("embedding_model", EMBEDDING_MODEL),
        )

    @classmethod
    def load(cls, index_path, metadata_path, top_k=TOP_K):
        return cls(index_path=index_path, metadata_path=metadata_path, top_k=top_k)

    @staticmethod
    def _query_parts(query):
        """Accept the agent's {context, risk, ...} query or a flat dataset row."""
        if isinstance(query, dict) and isinstance(query.get("context"), dict):
            context = query["context"]
            risk = query.get("risk") or {}
            return context, risk.get("category"), query.get("case_id")
        query = dict(query)
        return query, query.get("risk_category"), query.get("case_id")

    def retrieve(self, query, k=None):
        k = k or self.top_k
        context, risk_category, query_id = self._query_parts(query)
        vector = self.embedder.encode([case_text(context, risk_category)])
        n_candidates = min(k * CANDIDATE_MULTIPLIER, self.index.ntotal)
        scores, ids = self.index.search(vector, n_candidates)

        results = []
        for similarity, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            case = self.cases[i]
            # Never return the query's own record as "memory".
            if query_id is not None and case["case_id"] == str(query_id):
                continue
            outcome = _outcome_score(case)
            climate = _climate_match(context, case)
            final = (
                SIMILARITY_WEIGHT * float(similarity)
                + CLIMATE_WEIGHT * climate
                + OUTCOME_WEIGHT * outcome
            )
            experience = case.get("experience") or {}
            if experience.get("verified") and experience.get("reward") is not None:
                final += EXPERIENCE_WEIGHT * float(experience["reward"])
            result = {key: value for key, value in case.items() if key != "text"}
            result["similarity"] = float(similarity)
            result["climate_match"] = climate
            result["outcome_score"] = outcome
            result["final_score"] = final
            results.append(result)

        results.sort(key=lambda c: c["final_score"], reverse=True)
        return results[:k]
