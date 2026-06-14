# History Bot

교양 "한국문화유산탐사" PDF 강의자료를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 개요

PDF에서 텍스트·표를 추출하고, 청크 단위로 분할하여 FAISS 벡터 인덱스를 구축합니다. 질의 시 FAISS(의미 검색)와 BM25(키워드 검색)를 RRF 알고리즘으로 결합한 하이브리드 검색을 수행하고, GPT-4o-mini로 최종 답변을 생성합니다.

## 파이프라인

```
PDF 파일
  ↓ extract_pdf.py   — docling으로 텍스트·표·이미지 추출
  ↓ chunking.py      — 섹션 헤딩 기준 청크 분할 + 노이즈 제거
  ↓ embedding.py     — KURE-v1 모델로 임베딩 생성
  ↓ vector_store.py  — FAISS 인덱스 빌드 및 하이브리드 검색 (FAISS + BM25 → RRF)
  ↓ answering.py     — GPT-4o-mini RAG 답변 생성
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install docling sentence-transformers faiss-cpu rank-bm25 openai python-dotenv
```

## 환경 변수

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다.

```
GPT_API=your_openai_api_key
```

## 사용법

### 1. 인덱스 빌드

`data/original/` 폴더에 PDF 파일을 넣고 실행합니다.

```bash
python main.py
```

`data/results/`에 다음 파일이 생성됩니다.

| 파일 | 설명 |
|------|------|
| `1_extracted.json` | 추출된 원본 항목 |
| `2_chunks.json` | 청크 분할 결과 |
| `3_embeddings.npy` | 임베딩 벡터 |
| `faiss.index` | FAISS 인덱스 |
| `faiss_meta.json` | 청크 메타데이터 |

### 2. 질문하기

```bash
python answering.py 창경궁이 창경원으로 격하된 이유는 무엇인가요?
```

또는 인자 없이 실행하면 프롬프트로 입력할 수 있습니다.

```bash
python answering.py
```

출력 예시:

```
============================================================
[RAG 답변] (PDF 기반)
============================================================
창경궁이 창경원으로 격하된 이유는 ...

[참고 출처]
  · 9_창경궁    창경궁(2025년).pdf — 조선 후기 창경궁의 중건과 일제 강점기 창경궁의 격하 (유사도: 0.842)

============================================================
[LLM 답변] (RAG 미적용, LLM 자체 지식)
============================================================
창경궁은 ...
```

## 프로젝트 구조

```
history_bot/
├── data/
│   ├── original/        # 원본 PDF 파일 (gitignore)
│   └── results/         # 빌드 결과물 (gitignore)
├── extract_pdf.py       # PDF 추출
├── chunking.py          # 청크 분할
├── embedding.py         # 임베딩
├── vector_store.py      # FAISS + BM25 하이브리드 검색
├── answering.py         # RAG 답변 생성
└── main.py              # 전체 파이프라인 실행
```
