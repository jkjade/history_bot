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


if __name__ == "__main__":
    build_and_save()
