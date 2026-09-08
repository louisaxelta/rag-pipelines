from rag_pipelines.core.ingestion.detector import detect
from rag_pipelines.core.ingestion.parsers import get_parser
from rag_pipelines.core.ingestion.chunkers import get_chunker
from rag_pipelines.core.schema.field_mapper import apply_field_mapping
from rag_pipelines.core.aws.bedrock.bedrock import BedrockClient
from rag_pipelines.core.aws.s3.s3 import S3Client
from rag_pipelines.core.aws.s3.s3_vector import S3VectorClient
from rag_pipelines.core.config import get_config
from rag_pipelines.core.models import Chunk, ChunkerConfig, FieldMapping

CONFIG = get_config()


async def create_knowledge(
    bucket: str,
    key: str,
    chunker_config: ChunkerConfig,
    field_mapping: FieldMapping,
) -> list[str]:
    """
    Full ingestion flow:
      S3 download → detect → parse → chunk → map fields → embed → store vector

    Returns list of knowledge_ids stored.
    """
    filename = key.split("/")[-1]

    s3 = S3Client()
    bedrock = BedrockClient()
    s3_vector = S3VectorClient()

    file_stream = await s3.read_file(bucket, key)

    detection = detect(filename)

    parser = get_parser(detection.parser)
    parsed_chunks = parser.parse(file_stream, filename)

    chunker = get_chunker(detection.chunker_family, chunker_config)
    chunks_raw = chunker.chunk(parsed_chunks)

    chunks: list[Chunk] = apply_field_mapping(chunks_raw, field_mapping, source=filename)

    knowledge_ids: list[str] = []

    for chunk in chunks:
        vector = bedrock.embed(chunk.question)

        await s3_vector.put(
            bucket=CONFIG.S3_VECTOR_BUCKET,
            index=CONFIG.S3_VECTOR_INDEX,
            key=chunk.knowledge_id,
            vector=vector,
            metadata={
                **chunk.metadata,
                "answer": chunk.answer,
                "source": chunk.source,
                "s3_uri": f"s3://{bucket}/{key}",
            },
        )

        knowledge_ids.append(chunk.knowledge_id)

    return knowledge_ids
