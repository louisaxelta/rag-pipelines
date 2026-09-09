from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    APP_NAME: str = "RAG Pipeline Service"
    APP_ENV: str = "development"
    DEBUG: bool = False
    VERBOSE: bool = False

    # AWS
    AWS_REGION: str = "us-west-2"
    AWS_ROLE_ARN: str = ""
    USE_ASSUME_ROLE: bool = False
    MODEL_TEMPERATURE: float = 0.1
    MODEL_MAX_TOKENS: int = 16384

    # LLM Models
    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_EMBEDDING_MODEL: Optional[str] = None
    OPENROUTER_LLM_MODEL: Optional[str] = None

    # AWS Bedrock
    BEDROCK_REGION: str = "us-west-2"
    EMBEDDING_MODEL: str = "amazon.titan-embed-text-v2:0"
    EMBEDDING_DIMENSIONS: int = 512
    LLM_MODEL: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    # S3
    S3_BUCKET: str = ""
    S3_REGION: str = "us-west-2"

    # S3 Vectors
    S3_VECTOR_BUCKET: str = ""
    S3_VECTOR_INDEX: str = ""

    # Langfuse
    LANGFUSE_HOST: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_config() -> Config:
    return Config()


# format → (parser, chunker_family)
SUPPORTED_FORMATS: dict[str, tuple[str, str]] = {
    ".pdf":  ("pdf_parser",         "text"),
    ".txt":  ("pdf_parser",         "text"),
    ".md":  ("markdown_parser",         "text"),
    ".xlsx": ("excel_parser",       "tabular"),
    ".xls":  ("excel_parser",       "tabular"),
    ".csv":  ("csv_parser",         "tabular"),
    ".docx": ("word_parser",        "structural"),
    ".doc":  ("word_parser",        "structural"),
    ".pptx": ("powerpoint_parser",  "structural"),
    ".ppt":  ("powerpoint_parser",  "structural"),
}
