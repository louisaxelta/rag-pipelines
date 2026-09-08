from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile

from rag_pipelines.core.config import SUPPORTED_FORMATS
from rag_pipelines.core.ingestion.chunkers import get_chunker
from rag_pipelines.core.ingestion.detector import detect
from rag_pipelines.core.ingestion.parsers import get_parser
from rag_pipelines.core.models import ChunkerConfig
from rag_pipelines.core.schema.field_mapper import preview_fields

router = APIRouter(prefix="/documents", tags=["documents"])

_DEFAULT_STRATEGY = {
    "tabular": "row",
    "structural": "section",
    "text": "fixed",
}


@router.get("/formats")
async def list_formats():
    return {"formats": sorted(SUPPORTED_FORMATS.keys())}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept any supported document format, detect type, parse, and return a preview."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    try:
        detection = detect(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    file_stream = BytesIO(content)

    parser = get_parser(detection.parser)
    parsed_chunks = parser.parse(file_stream, file.filename)

    chunker_config = ChunkerConfig(strategy=_DEFAULT_STRATEGY[detection.chunker_family])
    chunker = get_chunker(detection.chunker_family, chunker_config)
    raw_chunks = chunker.chunk(parsed_chunks)

    return {
        "filename": file.filename,
        "detection": detection.model_dump(),
        "parsed_count": len(parsed_chunks),
        "chunk_count": len(raw_chunks),
        "fields": preview_fields(raw_chunks),
    }
