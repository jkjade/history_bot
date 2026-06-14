import json
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi

RESULTS_DIR = Path("data/results")
INDEX_PATH = RESULTS_DIR / "faiss.index"
META_PATH = RESULTS_DIR / "faiss_meta.json"

_index = None
_meta = None
_bm25 = None


def build_and_save():
    vectors = np.load(RESULTS_DIR / "3_embeddings.npy").astype(np.float32)
    chunks = json.loads((RESULTS_DIR / "2_chunks.json").read_text())

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))

    meta = [{"source": c["source"], "heading": c["heading"], "text": c["text"]} for c in chunks]
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"저장 완료: {index.ntotal}개 벡터 → {INDEX_PATH}")


def _tokenize(text: str) -> list[str]:
    return text.split()


def load():
    global _index, _meta, _bm25
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = json.loads(Path(META_PATH).read_text())
        corpus = [_tokenize(m["text"]) for m in _meta]
        _bm25 = BM25Okapi(corpus)
    return _index, _meta, _bm25


def search(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    index, meta, _ = load()

    vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, top_k)
    return [
        {"score": float(scores[0][i]), **meta[indices[0][i]]}
        for i in range(top_k)
    ]


def hybrid_search(query: str, query_embedding: np.ndarray, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
    """FAISS + BM25 결과를 RRF(Reciprocal Rank Fusion)로 결합.
    RRF score = Σ 1/(rrf_k + rank). 점수 정규화 없이 순위만 사용."""
    index, meta, bm25 = load()
    candidate_k = min(top_k * 10, len(meta))

    # FAISS 검색 → 순위 목록
    vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    _, vec_indices = index.search(vec, candidate_k)
    vec_ranks = {int(idx): rank for rank, idx in enumerate(vec_indices[0])}

    # BM25 검색 → 순위 목록
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_ranked = np.argsort(bm25_scores)[::-1][:candidate_k]
    bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_ranked)}

    # RRF 점수 합산
    all_indices = set(vec_ranks) | set(bm25_ranks)
    rrf_scores = {
        idx: (1 / (rrf_k + vec_ranks[idx]) if idx in vec_ranks else 0)
             + (1 / (rrf_k + bm25_ranks[idx]) if idx in bm25_ranks else 0)
        for idx in all_indices
    }

    top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    return [{"score": rrf_scores[idx], **meta[idx]} for idx in top_indices]


if __name__ == "__main__":
    build_and_save()
