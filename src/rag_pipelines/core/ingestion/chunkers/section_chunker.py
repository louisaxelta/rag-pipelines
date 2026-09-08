from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.chunkers.base import BaseChunker


class SectionChunker(BaseChunker):
    """
    Structural chunker: each section/slide = 1 chunk.
    Falls back to splitting body text if it exceeds chunk_size.
    """

    def chunk(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        results = []

        for pc in parsed_chunks:
            body = pc.content.get("body", "") or ""

            # if body fits within chunk_size, keep as-is
            if len(body) <= self.config.chunk_size:
                results.append({
                    **pc.content,
                    "_source": pc.source,
                    "_heading": pc.heading,
                    "_slide": pc.slide,
                })
            else:
                # split oversized body with overlap
                for chunk_text in self._split(body):
                    results.append({
                        **pc.content,
                        "body": chunk_text,
                        "_source": pc.source,
                        "_heading": pc.heading,
                        "_slide": pc.slide,
                    })

        return results

    def _split(self, text: str) -> list[str]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks
