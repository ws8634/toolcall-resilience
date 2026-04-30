from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field, field_validator

from toolcall_resilience.errors import ToolcallError


class Status(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


@dataclass
class ParameterSchema:
    name: str
    type_: ParameterType
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    enum: Optional[list[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    items: Optional["ParameterSchema"] = None
    properties: Optional[dict[str, "ParameterSchema"]] = None
    additional_properties: bool = True
    auto_convert: bool = True

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.type_.value,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.description:
            result["description"] = self.description
        if self.enum:
            result["enum"] = self.enum
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.min_length is not None:
            result["min_length"] = self.min_length
        if self.max_length is not None:
            result["max_length"] = self.max_length
        if self.pattern:
            result["pattern"] = self.pattern
        return result


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ParameterSchema] = field(default_factory=list)
    return_type: ParameterType = ParameterType.OBJECT
    strict: bool = False

    def get_parameter(self, name: str) -> Optional[ParameterSchema]:
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type.value,
            "strict": self.strict,
        }


class ToolRequest(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    id_: Optional[str] = Field(default=None, alias="id")

    model_config = {"populate_by_name": True}

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tool_name cannot be empty")
        return v.strip()


class AttemptRecord(BaseModel):
    attempt_number: int
    status: Status
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float = 0.0
    parameters: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    raw_response: Optional[Any] = None
    normalized_response: Optional[Any] = None

    @classmethod
    def from_exception(
        cls,
        attempt_number: int,
        exception: Exception,
        parameters: Optional[dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> "AttemptRecord":
        error_dict: dict[str, Any]
        if isinstance(exception, ToolcallError):
            error_dict = exception.to_dict()
        else:
            error_dict = {
                "category": "unknown",
                "code": "unexpected_exception",
                "message": str(exception),
                "details": {"exception_type": type(exception).__name__},
                "is_retryable": False,
            }

        return cls(
            attempt_number=attempt_number,
            status=Status.FAILED,
            duration_ms=duration_ms,
            parameters=parameters,
            error=error_dict,
        )

    @classmethod
    def from_success(
        cls,
        attempt_number: int,
        response: Any,
        normalized_response: Any,
        parameters: Optional[dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> "AttemptRecord":
        return cls(
            attempt_number=attempt_number,
            status=Status.SUCCESS,
            duration_ms=duration_ms,
            parameters=parameters,
            raw_response=response,
            normalized_response=normalized_response,
        )


class ToolResponse(BaseModel):
    success: bool
    tool_name: str
    request_id: Optional[str] = None
    final_parameters: dict[str, Any] = Field(default_factory=dict)
    attempt_count: int = 0
    attempts: list[AttemptRecord] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    final_result: Optional[Any] = None
    final_error: Optional[dict[str, Any]] = None

    def get_last_error(self) -> Optional[dict[str, Any]]:
        for attempt in reversed(self.attempts):
            if attempt.error:
                return attempt.error
        return self.final_error

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "request_id": self.request_id,
            "attempt_count": self.attempt_count,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "final_result": self.final_result,
            "final_error": self.final_error,
        }

    def to_detailed_dict(self) -> dict[str, Any]:
        return {
            **self.to_summary_dict(),
            "final_parameters": self.final_parameters,
            "attempts": [a.model_dump() for a in self.attempts],
        }


@dataclass
class RegisteredTool:
    name: str
    func: Callable[..., Any]
    schema: ToolSchema
    timeout: Optional[float] = None
    is_async: bool = False
