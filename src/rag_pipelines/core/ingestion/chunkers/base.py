from pydantic import BaseModel
from rag_pipelines.core.models import ParsedChunk, ChunkerConfig


class BaseChunker(BaseModel):
    config: ChunkerConfig

    def chunk(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        raise NotImplementedError
