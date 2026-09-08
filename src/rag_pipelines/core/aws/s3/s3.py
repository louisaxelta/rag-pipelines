import asyncio
import boto3
from io import BytesIO
from botocore.config import Config
from botocore.exceptions import ClientError

from rag_pipelines.core.config import get_config
from rag_pipelines.core.logging import logger

CONFIG = get_config()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


class S3Client:
    def __init__(self, region: str = CONFIG.S3_REGION):
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=10,
            )
        )

    def _parse_bucket(self, bucket: str, key: str) -> tuple:
        parts = bucket.split("/", 1)
        if len(parts) > 1:
            return parts[0], f"{parts[1]}/{key}"
        return bucket, key

    async def read_file(self, bucket: str, key: str) -> BytesIO:
        bucket, key = self._parse_bucket(bucket, key)
        try:
            def _sync_get():
                response = self._client.get_object(Bucket=bucket, Key=key)
                return BytesIO(response["Body"].read())
            return await asyncio.to_thread(_sync_get)
        except ClientError as e:
            logger.error(f"Failed to read file s3://{bucket}/{key}: {e}")
            raise

    async def put_object(self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> str:
        bucket, key = self._parse_bucket(bucket, key)
        try:
            await asyncio.to_thread(
                self._client.put_object, Bucket=bucket, Key=key, Body=body, ContentType=content_type
            )
            return f"s3://{bucket}/{key}"
        except ClientError as e:
            logger.error(f"Failed to put object s3://{bucket}/{key}: {e}")
            raise

    def create_presigned_post(self, bucket: str, key: str, content_type: str, expiration: int = 600) -> dict:
        """Generate S3 Presigned POST for frontend direct upload. File size must be <= 50 MB."""
        try:
            response = self._client.generate_presigned_post(
                Bucket=bucket,
                Key=key,
                Fields={"Content-Type": content_type},
                Conditions=[
                    {"Content-Type": content_type},
                    ["content-length-range", 1, MAX_UPLOAD_SIZE],
                ],
                ExpiresIn=expiration,
            )
            logger.info(f"[S3 Presigned POST] Created for s3://{bucket}/{key}")
            return response
        except ClientError as e:
            logger.error(f"Failed to create presigned POST s3://{bucket}/{key}: {e}")
            raise
