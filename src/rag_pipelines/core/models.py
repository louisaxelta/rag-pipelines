from typing import Any, Literal
from pydantic import BaseModel


# --- Parser output ---

class ParsedChunk(BaseModel):
    content: dict[str, Any]       # raw extracted content (row, section, slide, page text)
    source: str                   # filename
    page: int | None = None       # for pdf
    sheet: str | None = None      # for excel
    slide: int | None = None      # for pptx
    heading: str | None = None    # for docx


# --- Chunker config ---

ChunkerStrategy = Literal["faq", "row", "section", "fixed", "recursive", "semantic", "sliding_window"]

class ChunkerConfig(BaseModel):
    strategy: ChunkerStrategy
    chunk_size: int = 512
    chunk_overlap: int = 50
    group_by: str | None = None   # for row_chunker: group rows by column value


# --- Field mapping ---

class FieldMapping(BaseModel):
    question: str                 # field name that gets embedded
    answer: str                   # field name returned on retrieval
    metadata: list[str]           # field names stored as metadata


# --- Chunk (after chunking + field mapping applied) ---

class Chunk(BaseModel):
    knowledge_id: str
    question: str
    answer: str
    metadata: dict[str, Any]
    source: str
