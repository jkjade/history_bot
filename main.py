import json
import numpy as np
from pathlib import Path

from extract_pdf import extract_pdf
from chunking import chunk_items
from embedding import _get_model

RESULTS_DIR = Path("data/results")


def _serialize_items(items: list[dict], source: str) -> list[dict]:
    serialized = []
    for item in items:
        if item["type"] in ("heading", "text"):
            serialized.append({"source": source, "type": item["type"], "content": item["content"]})
        elif item["type"] == "table":
            serialized.append({"source": source, "type": "table", "content": item["content"].to_string(index=False)})
        elif item["type"] == "image":
            serialized.append({"source": source, "type": "image", "caption": item.get("caption", "")})
    return serialized


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(Path("data/original").glob("*.pdf"))

    all_extracted = []
    all_chunks = []
    all_vectors = []

    for pdf_path in pdf_paths:
        print(f"\n[{pdf_path.name}]")
        source = pdf_path.name

        # 1. Extract
        print("  [1/3] 추출 중...")
        items = extract_pdf(str(pdf_path))
        all_extracted.extend(_serialize_items(items, source))

        # 2. Chunk
        print("  [2/3] 청킹 중...")
        chunks = chunk_items(items)
        for chunk in chunks:
            chunk["source"] = source
        all_chunks.extend(chunks)

        # 3. Embed
        print("  [3/3] 임베딩 중...")
        if chunks:
            model = _get_model()
            vectors = model.encode([c["text"] for c in chunks], batch_size=1, show_progress_bar=False)
            all_vectors.append(np.atleast_2d(vectors))
        print(f"  → {len(chunks)}개 청크")

    # 저장
    print("\n결과 저장 중...")
    (RESULTS_DIR / "1_extracted.json").write_text(
        json.dumps(all_extracted, ensure_ascii=False, indent=2)
    )
    (RESULTS_DIR / "2_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2)
    )

    vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)
    np.save(RESULTS_DIR / "3_embeddings.npy", vectors)

    # 4. FAISS (docling 작업 완료 후 임포트)
    print("[4/4] FAISS 인덱스 빌드 중...")
    import faiss
    
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(RESULTS_DIR / "faiss.index"))

    meta = [{"source": c["source"], "heading": c["heading"], "text": c["text"]} for c in all_chunks]
    (RESULTS_DIR / "faiss_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    print(f"\n완료: {len(all_chunks)}개 청크, {index.ntotal}개 벡터")
    print(f"저장 위치: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
