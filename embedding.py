from pathlib import Path
from sentence_transformers import SentenceTransformer
from chunking import chunk_pdf

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("nlpai-lab/KURE-v1")
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, batch_size=1, show_progress_bar=True)
    return [
        {
            "source": chunk["source"],
            "heading": chunk["heading"],
            "text": chunk["text"],
            "embedding": vectors[i],
        }
        for i, chunk in enumerate(chunks)
    ]


def embed_all_pdfs(data_dir: str = "data") -> list[dict]:
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    all_chunks = []

    for pdf_path in pdf_paths:
        print(f"청킹 중: {pdf_path.name}")
        chunks = chunk_pdf(str(pdf_path))
        for chunk in chunks:
            chunk["source"] = pdf_path.name
        all_chunks.extend(chunks)

    print(f"\n총 {len(all_chunks)}개 청크 임베딩 중...")
    return embed_chunks(all_chunks)
