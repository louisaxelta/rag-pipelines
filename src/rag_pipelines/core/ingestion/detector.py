from pathlib import Path
from pydantic import BaseModel

from rag_pipelines.core.config import SUPPORTED_FORMATS


class DetectionResult(BaseModel):
    extension: str
    parser: str          # which parser to use
    chunker_family: str  # "tabular" | "structural" | "text"


def detect(filename: str) -> DetectionResult:
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: '{ext}'. Supported: {list(SUPPORTED_FORMATS)}")

    parser, chunker_family = SUPPORTED_FORMATS[ext]

    return DetectionResult(
        extension=ext,
        parser=parser,
        chunker_family=chunker_family,
    )
