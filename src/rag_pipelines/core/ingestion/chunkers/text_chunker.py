import re
from rag_pipelines.core.llm.client import invoke_structured
from rag_pipelines.core.llm.schemas import SemanticChunkGroups
from rag_pipelines.core.logging import logger
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
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        logger.info(
            "[SemanticChunker] sentences=%s chunk_size=%s",
            len(sentences),
            self.config.chunk_size,
        )

        numbered_sentences = "\n".join(
            f"[{index}] {sentence}"
            for index, sentence in enumerate(sentences)
        )

        prompt = f"""Group these sentences into semantically related chunks.
Keep sentence order and aim for at most {self.config.chunk_size} characters per group.
Return every sentence index exactly once.

{numbered_sentences}
"""
        result = invoke_structured(
            prompt,
            SemanticChunkGroups,
            max_tokens=16384,
        )

        logger.info("[SemanticChunker] groups=%s", len(result.groups))
        logger.debug("[SemanticChunker] result=%s", result.model_dump())

        invalid_indices = [
            index
            for group in result.groups
            for index in group.indices
            if not 0 <= index < len(sentences)
        ]
        if invalid_indices:
            logger.error(
                "[SemanticChunker] invalid_indices=%s valid_range=0-%s",
                invalid_indices,
                len(sentences) - 1,
            )

        return [
            " ".join(sentences[index] for index in group.indices)
            for group in result.groups
        ]
