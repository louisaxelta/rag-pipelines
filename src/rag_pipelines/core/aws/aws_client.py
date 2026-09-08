import boto3
from botocore.config import Config
from rag_pipelines.core.config import get_config
from rag_pipelines.core.logging import logger

CONFIG = get_config()


def _build_bedrock_client():
    if CONFIG.USE_ASSUME_ROLE:
        from botocore.credentials import RefreshableCredentials
        from botocore.session import get_botocore_session

        def _fetch_creds() -> dict:
            sts = boto3.client("sts", region_name=CONFIG.BEDROCK_REGION)
            assumed = sts.assume_role(
                RoleArn=CONFIG.AWS_ROLE_ARN,
                RoleSessionName="rag-pipelines-bedrock-session",
            )
            creds = assumed["Credentials"]
            expiry = creds["Expiration"]
            logger.info(f"STS credentials refreshed — valid until {expiry.isoformat()}")
            return {
                "access_key":  creds["AccessKeyId"],
                "secret_key":  creds["SecretAccessKey"],
                "token":       creds["SessionToken"],
                "expiry_time": expiry.isoformat(),
            }

        refreshable_creds = RefreshableCredentials.create_from_metadata(
            metadata=_fetch_creds(),
            refresh_using=_fetch_creds,
            method="sts-assume-role",
        )
        botocore_sess = get_botocore_session()
        botocore_sess._credentials = refreshable_creds
        session = boto3.Session(botocore_session=botocore_sess)
        logger.info("Bedrock session initialized (assume role)")

        return session.client(
            "bedrock-runtime",
            region_name=CONFIG.BEDROCK_REGION,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
        )

    logger.info("Bedrock session initialized (direct credentials)")
    return boto3.client(
        "bedrock-runtime",
        region_name=CONFIG.BEDROCK_REGION,
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
    )

