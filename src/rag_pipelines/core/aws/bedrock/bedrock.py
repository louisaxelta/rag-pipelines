import json
from typing import Optional

from rag_pipelines.core.config import get_config
from rag_pipelines.core.logging import logger
from rag_pipelines.core.aws.aws_client import _build_bedrock_client

CONFIG = get_config()


class BedrockClient:
    """
    AWS Bedrock client wrapper.
    Handles LLM invocation and text embedding via Bedrock runtime.
    Model-agnostic — caller specifies model_id per call or at init.
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        default_model_id: Optional[str] = None,
    ):
        self.region = region_name or CONFIG.BEDROCK_REGION
        self.default_model_id = default_model_id or CONFIG.LLM_MODEL
        self._client = _build_bedrock_client()
        logger.info(f"BedrockClient initialized region={self.region} default_model={self.default_model_id}")

    def invoke(self, prompt: str, model_id: Optional[str] = None) -> str:
        """Invoke LLM with a prompt, returns text response."""
        model = model_id or self.default_model_id
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        response = self._client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def embed(self, text: str, model_id: Optional[str] = None) -> list[float]:
        """Embed text, returns float vector."""
        model = model_id or CONFIG.EMBEDDING_MODEL
        body = {"inputText": text, "dimensions": CONFIG.EMBEDDING_DIMENSIONS}
        response = self._client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

