import pdfplumber
from io import BytesIO
from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.parsers.base import BaseParser


class PdfParser(BaseParser):
    model_config = {"arbitrary_types_allowed": True}
    def parse(self, file_stream: BytesIO, filename: str) -> list[ParsedChunk]:
        chunks = []
        with pdfplumber.open(file_stream) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(
                    x_tolerance=1,
                    y_tolerance=3,
                    ) or ""

                if text.strip():
                    chunks.append(ParsedChunk(
                        content={"text": text.strip()},
                        source=filename,
                        page=i,
                    ))
        return chunks
