import re
from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.chunkers.base import BaseChunker


class TextChunker(BaseChunker):
    """
    Unstructured text chunker for PDF/TXT.
    Supports: fixed, recursive, semantic, sliding_window.
    """

    def chunk(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        full_text = " ".join(
            pc.content.get("text", "") for pc in parsed_chunks
        )
        source = parsed_chunks[0].source if parsed_chunks else ""

        strategy = self.config.strategy
        if strategy == "fixed":
            splits = self._fixed(full_text)
        elif strategy == "recursive":
            splits = self._recursive(full_text)
        elif strategy == "semantic":
            splits = self._semantic(full_text)
        elif strategy == "sliding_window":
            splits = self._sliding_window(full_text)
        else:
            splits = self._fixed(full_text)

        return [{"text": s, "_source": source} for s in splits if s.strip()]

    # --- strategies ---

    def _fixed(self, text: str) -> list[str]:
        size = self.config.chunk_size
        return [text[i: i + size] for i in range(0, len(text), size)]

    def _sliding_window(self, text: str) -> list[str]:
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start: start + size])
            start += size - overlap
        return chunks

    def _recursive(self, text: str, separators: list[str] | None = None) -> list[str]:
        separators = separators or ["\n\n", "\n", ". ", " "]
        size = self.config.chunk_size

        if len(text) <= size:
            return [text]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                chunks = []
                current = ""
                for part in parts:
                    if len(current) + len(part) + len(sep) <= size:
                        current += part + sep
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = part + sep
                if current:
                    chunks.append(current.strip())
                return chunks

        return self._fixed(text)

    def _semantic(self, text: str) -> list[str]:
        # split by sentence boundaries, then group up to chunk_size
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.config.chunk_size:
                current += " " + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks
