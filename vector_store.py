import json
import numpy as np
import faiss
from pathlib import Path
from embedding import embed_all_pdfs

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.json"


def build_and_save(data_dir: str = "data"):
    results = embed_all_pdfs(data_dir)

    vectors = np.array([r["embedding"] for r in results], dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    meta = [{"source": r["source"], "heading": r["heading"], "text": r["text"]} for r in results]
    Path(META_PATH).write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"저장 완료: {index.ntotal}개 벡터 → {INDEX_PATH}, {META_PATH}")


def load():
    index = faiss.read_index(INDEX_PATH)
    meta = json.loads(Path(META_PATH).read_text())
    return index, meta


def search(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    index, meta = load()

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
