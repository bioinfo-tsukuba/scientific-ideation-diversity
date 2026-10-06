"""Retry error constants shared by AWS-based embedding backends."""

from __future__ import annotations

# boto3 ClientError codes that warrant a retry.
BEDROCK_RETRYABLE_CLIENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
    }
)

# botocore low-level transport error names that warrant a retry.
BOTOCORE_RETRYABLE_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ReadTimeoutError",
        "ResponseStreamingError",
    }
)
