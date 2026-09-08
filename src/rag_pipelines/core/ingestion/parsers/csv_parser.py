import csv
from io import BytesIO
from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.parsers.base import BaseParser


class CsvParser(BaseParser):
    model_config = {"arbitrary_types_allowed": True}
    def parse(self, file_stream: BytesIO, filename: str) -> list[ParsedChunk]:
        chunks = []
        reader = csv.DictReader(file_stream.read().decode("utf-8").splitlines())

        for row in reader:
            if not any(row.values()):
                continue
            chunks.append(ParsedChunk(
                content=dict(row),
                source=filename,
            ))

        return chunks
