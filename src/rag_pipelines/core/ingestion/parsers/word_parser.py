from io import BytesIO
from docx import Document
from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.parsers.base import BaseParser


class WordParser(BaseParser):
    model_config = {"arbitrary_types_allowed": True}
    def parse(self, file_stream: BytesIO, filename: str) -> list[ParsedChunk]:
        chunks = []
        doc = Document(file_stream)

        current_heading = None
        current_body: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if para.style.name.startswith("Heading"):
                if current_body:
                    chunks.append(ParsedChunk(
                        content={"heading": current_heading, "body": " ".join(current_body)},
                        source=filename,
                        heading=current_heading,
                    ))
                current_heading = text
                current_body = []
            else:
                current_body.append(text)

        # flush last section
        if current_body:
            chunks.append(ParsedChunk(
                content={"heading": current_heading, "body": " ".join(current_body)},
                source=filename,
                heading=current_heading,
            ))

        return chunks
