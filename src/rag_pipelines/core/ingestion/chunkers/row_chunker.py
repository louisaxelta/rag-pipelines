from rag_pipelines.core.models import ParsedChunk
from rag_pipelines.core.ingestion.chunkers.base import BaseChunker


class RowChunker(BaseChunker):
    """
    Tabular chunker: each row = 1 chunk, or group every N rows.
    Optionally group by a column value (group_by).
    """

    def chunk(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        if self.config.group_by:
            return self._chunk_by_column(parsed_chunks)
        if self.config.chunk_size > 1:
            return self._chunk_by_n(parsed_chunks)
        return self._chunk_by_row(parsed_chunks)

    def _chunk_by_row(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        return [{**pc.content, "_source": pc.source, "_sheet": pc.sheet} for pc in parsed_chunks]

    def _chunk_by_n(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        results = []
        for i in range(0, len(parsed_chunks), self.config.chunk_size):
            group = parsed_chunks[i: i + self.config.chunk_size]
            merged = {}
            for pc in group:
                for k, v in pc.content.items():
                    merged.setdefault(k, []).append(v)
            results.append({
                **{k: " | ".join(str(v) for v in vals) for k, vals in merged.items()},
                "_source": group[0].source,
                "_sheet": group[0].sheet,
            })
        return results

    def _chunk_by_column(self, parsed_chunks: list[ParsedChunk]) -> list[dict]:
        groups: dict[str, list[ParsedChunk]] = {}
        for pc in parsed_chunks:
            key = str(pc.content.get(self.config.group_by, "unknown"))
            groups.setdefault(key, []).append(pc)

        results = []
        for group_key, group in groups.items():
            merged = {}
            for pc in group:
                for k, v in pc.content.items():
                    merged.setdefault(k, []).append(v)
            results.append({
                **{k: " | ".join(str(v) for v in vals) for k, vals in merged.items()},
                "_group": group_key,
                "_source": group[0].source,
                "_sheet": group[0].sheet,
            })
        return results
