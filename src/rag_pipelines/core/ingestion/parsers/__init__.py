from rag_pipelines.core.ingestion.parsers.pdf_parser import PdfParser
from rag_pipelines.core.ingestion.parsers.excel_parser import ExcelParser
from rag_pipelines.core.ingestion.parsers.csv_parser import CsvParser
from rag_pipelines.core.ingestion.parsers.word_parser import WordParser
from rag_pipelines.core.ingestion.parsers.powerpoint_parser import PowerpointParser

_PARSERS = {
    "pdf_parser":         PdfParser,
    "excel_parser":       ExcelParser,
    "csv_parser":         CsvParser,
    "word_parser":        WordParser,
    "powerpoint_parser":  PowerpointParser,
}


def get_parser(parser_name: str):
    if parser_name not in _PARSERS:
        raise ValueError(f"Unknown parser: {parser_name}")
    return _PARSERS[parser_name]()
