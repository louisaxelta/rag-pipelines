from io import BytesIO
from pydantic import BaseModel
from rag_pipelines.core.models import ParsedChunk


class BaseParser(BaseModel):
    def parse(self, file_stream: BytesIO, filename: str) -> list[ParsedChunk]:
        raise NotImplementedError
