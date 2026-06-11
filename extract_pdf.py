from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.document import PictureItem


def extract_pdf(pdf_path: str) -> dict:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    items = []

    for item, _ in doc.iterate_items():
        label = item.label.value if hasattr(item.label, "value") else str(item.label)

        if label in ("text", "paragraph", "section_header", "title", "list_item"):
            items.append({"type": "text", "content": item.text})

        elif label == "table":
            df = item.export_to_dataframe(doc)
            items.append({"type": "table", "content": df})

        elif label == "picture":
            img = item.get_image(doc)
            if img is not None:
                caption = item.caption_text(doc) if callable(getattr(item, "caption_text", None)) else ""
                items.append({"type": "image", "content": img, "caption": caption})

    return items

