import asyncio
import boto3
from typing import Any
from botocore.config import Config
from botocore.exceptions import ClientError

from rag_pipelines.core.config import get_config
from rag_pipelines.core.logging import logger

CONFIG = get_config()


class S3VectorClient:
    def __init__(self, region: str = CONFIG.AWS_REGION):
        self._client = boto3.client(
            "s3vectors",
            region_name=region,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
        )

    async def put(
        self,
        bucket: str,
        index: str,
        key: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        try:
            await asyncio.to_thread(
                self._client.put_vectors,
                vectorBucketName=bucket,
                indexName=index,
                vectors=[{
                    "key": key,
                    "data": {"float32": vector},
                    "metadata": metadata,
                }]
            )
            logger.info(f"Vector stored: bucket={bucket} index={index} key={key}")
        except ClientError as e:
            logger.error(f"Failed to put vector key={key}: {e}")
            raise

    async def query(
        self,
        bucket: str,
        index: str,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        try:
            kwargs = dict(
                vectorBucketName=bucket,
                indexName=index,
                queryVector={"float32": vector},
                topK=top_k,
                returnMetadata=True,
            )
            if filter:
                kwargs["filter"] = filter

            response = await asyncio.to_thread(
                self._client.query_vectors, **kwargs
            )
            logger.info(f"Vector query: bucket={bucket} index={index} top_k={top_k} results={len(response.get('vectors', []))}")
            return response.get("vectors", [])
        except ClientError as e:
            logger.error(f"Failed to query vectors: {e}")
            raise

    async def delete(self, bucket: str, index: str, key: str) -> None:
        try:
            await asyncio.to_thread(
                self._client.delete_vectors,
                vectorBucketName=bucket,
                indexName=index,
                keys=[key],
            )
            logger.info(f"Vector deleted: bucket={bucket} index={index} key={key}")
        except ClientError as e:
            logger.error(f"Failed to delete vector key={key}: {e}")
            raise

    async def list(self, bucket: str, index: str) -> list[dict]:
        try:
            response = await asyncio.to_thread(
                self._client.list_vectors,
                vectorBucketName=bucket,
                indexName=index,
                returnMetadata=True,
            )
            return response.get("vectors", [])
        except ClientError as e:
            logger.error(f"Failed to list vectors: {e}")
            raise
