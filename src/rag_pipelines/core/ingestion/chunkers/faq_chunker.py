from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.chunkers.base import BaseChunker
from rag_pipelines.core.logging import logger


class FaqChunker(BaseChunker):
    """
    Strict FAQ template: each row must have a question + answer field.
    No splitting needed — each row is already one chunk.
    Field mapping is pre-determined: question → embed, answer → retrieve.
    """

    def chunk(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        results = []

        for pc in parsed_chunks:
            if self.config.group_by not in pc.content:
                logger.warning(f"Skipping row missing '{self.config.group_by}' field: {pc.content}")
                continue
            if "answer" not in pc.content:
                logger.warning(f"Skipping row missing 'answer' field: {pc.content}")
                continue

            results.append({
                **pc.content,
                "_source": pc.source,
                "_sheet": pc.sheet,
            })

        return results
