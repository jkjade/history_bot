import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from embedding import _get_model
from vector_store import hybrid_search

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GPT_API")
        if not api_key:
            raise ValueError("GPT_API 환경변수가 설정되지 않았습니다.")
        _client = OpenAI(api_key=api_key)
    return _client


def _embed_query(query: str) -> np.ndarray:
    model = _get_model()
    return model.encode(query, batch_size=1, show_progress_bar=False)


def _build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['source']} — {r['heading']}" if r["heading"] else f"[{i}] {r['source']}"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n".join(parts)


RAG_SYSTEM_PROMPT = """당신은 역사 교육 전문 AI 어시스턴트입니다.
주어진 참고 자료를 바탕으로 질문에 답하세요.
참고 자료에 없는 내용은 추측하지 말고, 모른다고 답하세요.
답변은 한국어로 작성하세요."""

PLAIN_SYSTEM_PROMPT = """당신은 역사 교육 전문 AI 어시스턴트입니다.
질문에 대해 알고 있는 내용을 바탕으로 답하세요.
답변은 한국어로 작성하세요."""


def _chat(system: str, user: str) -> str:
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer(query: str, top_k: int = 5) -> dict:
    query_vec = _embed_query(query)
    results = hybrid_search(query, query_vec, top_k=top_k)
    context = _build_context(results)

    rag_answer = _chat(
        system=RAG_SYSTEM_PROMPT,
        user=f"참고 자료:\n{context}\n\n질문: {query}",
    )
    plain_answer = _chat(
        system=PLAIN_SYSTEM_PROMPT,
        user=query,
    )

    return {
        "rag_answer": rag_answer,
        "plain_answer": plain_answer,
        "sources": results,
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("질문: ")
    result = answer(query)

    print("\n" + "=" * 60)
    print("[RAG 답변] (PDF 기반)")
    print("=" * 60)
    print(result["rag_answer"])

    print("\n[참고 출처]")
    for r in result["sources"]:
        label = f"{r['source']} — {r['heading']}" if r["heading"] else r["source"]
        print(f"  · {label} (유사도: {r['score']:.3f})")

    print("\n" + "=" * 60)
    print("[LLM 답변] (RAG 미적용, LLM 자체 지식)")
    print("=" * 60)
    print(result["plain_answer"])
