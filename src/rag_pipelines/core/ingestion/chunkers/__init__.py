from rag_pipelines.core.models import ChunkerConfig
from rag_pipelines.core.ingestion.chunkers.faq_chunker import FaqChunker
from rag_pipelines.core.ingestion.chunkers.row_chunker import RowChunker
from rag_pipelines.core.ingestion.chunkers.section_chunker import SectionChunker
from rag_pipelines.core.ingestion.chunkers.text_chunker import TextChunker

_CHUNKER_FAMILY = {
    "tabular":    RowChunker,
    "structural": SectionChunker,
    "text":       TextChunker,
}

_CHUNKER_STRATEGY = {
    "faq":            FaqChunker,
    "row":            RowChunker,
    "section":        SectionChunker,
    "fixed":          TextChunker,
    "recursive":      TextChunker,
    "semantic":       TextChunker,
    "sliding_window": TextChunker,
}


def get_chunker(chunker_family: str, config: ChunkerConfig):
    # if user explicitly picked a strategy, use that
    if config.strategy in _CHUNKER_STRATEGY:
        return _CHUNKER_STRATEGY[config.strategy](config=config)
    # otherwise fall back to family default
    if chunker_family not in _CHUNKER_FAMILY:
        raise ValueError(f"Unknown chunker family: {chunker_family}")
    return _CHUNKER_FAMILY[chunker_family](config=config)
