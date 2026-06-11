import re
from extract_pdf import extract_pdf


_HAS_CONTENT = re.compile(r'[가-힣a-zA-Z0-9一-鿿]')
_STRIP_NON_CONTENT = re.compile(r'[^가-힣a-zA-Z0-9一-鿿\s]')


def _is_noise(text: str) -> bool:
    return not bool(_HAS_CONTENT.search(text))


def _clean_text(text: str) -> str:
    cleaned = _STRIP_NON_CONTENT.sub('', text)
    return re.sub(r' +', ' ', cleaned).strip()


def _table_to_text(df) -> str:
    return df.to_string(index=False)


def _item_to_text(item: dict) -> str:
    if item["type"] in ("heading", "text"):
        return _clean_text(item["content"])
    elif item["type"] == "table":
        return _clean_text(_table_to_text(item["content"]))
    elif item["type"] == "image":
        caption = item.get("caption", "").strip()
        return f"[이미지: {caption}]" if caption else "[이미지]"
    return ""


def chunk_pdf(pdf_path: str) -> list[dict]:
    items = extract_pdf(pdf_path)

    chunks = []
    current_heading = ""
    current_body = []

    def flush():
        if not current_body:
            return
        text_parts = []
        if current_heading:
            text_parts.append(current_heading)
        for item in current_body:
            t = _item_to_text(item)
            if t:
                text_parts.append(t)
        chunks.append({
            "heading": current_heading,
            "items": current_body,
            "text": "\n".join(text_parts),
        })

    for item in items:
        if item["type"] == "heading":
            flush()
            current_heading = _clean_text(item["content"])
            current_body = []
        elif item["type"] == "text" and _is_noise(item["content"]):
            continue
        else:
            current_body.append(item)

    flush()
    return chunks


if __name__ == "__main__":
    chunks = chunk_pdf("data/1_음양오행으로보는우리문화.pdf")
    for i, chunk in enumerate(chunks):
        print(f"=== Chunk {i} | heading: {repr(chunk['heading'])} ===")
        print(chunk["text"])
        print()
