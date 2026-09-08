from uuid import uuid4
from rag_pipelines.core.models import Chunk, FieldMapping


def apply_field_mapping(
    raw_chunks: list[dict],
    field_mapping: FieldMapping,
    source: str,
) -> list[Chunk]:
    chunks = []

    for raw in raw_chunks:
        question = raw.get(field_mapping.question)
        answer = raw.get(field_mapping.answer)

        if not question or not answer:
            continue

        metadata = {k: raw[k] for k in field_mapping.metadata if k in raw}

        chunks.append(Chunk(
            knowledge_id=str(uuid4()),
            question=str(question),
            answer=str(answer),
            metadata=metadata,
            source=source,
        ))

    return chunks


def preview_fields(raw_chunks: list[dict]) -> list[str]:
    """Return available field names from the first chunk for UI field selection."""
    if not raw_chunks:
        return []
    return [k for k in raw_chunks[0].keys() if not k.startswith("_")]
