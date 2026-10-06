"""Pydantic validation status and metadata schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PydanticValidationStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    PASSED = "passed"
    FAILED = "failed"


class ValidationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pydantic_validation_status: PydanticValidationStatus
    pydantic_validation_error: Optional[str]
