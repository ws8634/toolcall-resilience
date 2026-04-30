from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ErrorCategory(Enum):
    TOOL_EXECUTION = "tool_execution"
    TIMEOUT = "timeout"
    PARAMETER = "parameter"
    PARSE = "parse"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class ToolcallError(Exception):
    category: ErrorCategory = ErrorCategory.UNKNOWN
    is_retryable: bool = False
    code: str = "unknown_error"

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "is_retryable": self.is_retryable,
        }


class RetryableError(ToolcallError):
    is_retryable: bool = True


class NonRetryableError(ToolcallError):
    is_retryable: bool = False


class ToolExecutionError(RetryableError):
    category: ErrorCategory = ErrorCategory.TOOL_EXECUTION
    code: str = "tool_execution_error"


class TimeoutError(RetryableError):
    category: ErrorCategory = ErrorCategory.TIMEOUT
    code: str = "timeout_error"

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        details: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, details, original_exception)
        self.timeout_seconds = timeout_seconds
        self.details["timeout_seconds"] = timeout_seconds


class ParameterError(NonRetryableError):
    category: ErrorCategory = ErrorCategory.PARAMETER
    code: str = "parameter_error"


class MissingParameterError(ParameterError):
    code: str = "missing_parameter"

    def __init__(
        self,
        parameter_name: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        msg = message or f"Missing required parameter: {parameter_name}"
        super().__init__(msg, details)
        self.parameter_name = parameter_name
        self.details["parameter_name"] = parameter_name


class ExtraParameterError(ParameterError):
    code: str = "extra_parameter"

    def __init__(
        self,
        parameter_name: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        msg = message or f"Unexpected extra parameter: {parameter_name}"
        super().__init__(msg, details)
        self.parameter_name = parameter_name
        self.details["parameter_name"] = parameter_name


class ParameterTypeError(ParameterError):
    code: str = "parameter_type_error"

    def __init__(
        self,
        parameter_name: str,
        expected_type: str,
        actual_type: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        msg = message or f"Parameter '{parameter_name}' expects type {expected_type}, got {actual_type}"
        super().__init__(msg, details)
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.details.update({
            "parameter_name": parameter_name,
            "expected_type": expected_type,
            "actual_type": actual_type,
        })


class ParameterValueError(ParameterError):
    code: str = "parameter_value_error"

    def __init__(
        self,
        parameter_name: str,
        value: Any,
        constraint: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        msg = message or f"Parameter '{parameter_name}' value {value!r} violates constraint: {constraint}"
        super().__init__(msg, details)
        self.parameter_name = parameter_name
        self.value = value
        self.constraint = constraint
        self.details.update({
            "parameter_name": parameter_name,
            "value": value,
            "constraint": constraint,
        })


class ParseError(NonRetryableError):
    category: ErrorCategory = ErrorCategory.PARSE
    code: str = "parse_error"


class JsonParseError(ParseError):
    code: str = "json_parse_error"

    def __init__(
        self,
        raw_input: str,
        parse_position: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        msg = message or f"Failed to parse JSON input"
        super().__init__(msg, details, original_exception)
        self.raw_input = raw_input
        self.parse_position = parse_position
        self.details.update({
            "raw_input_preview": raw_input[:100] if len(raw_input) > 100 else raw_input,
            "input_length": len(raw_input),
        })
        if parse_position is not None:
            self.details["parse_position"] = parse_position


class InvalidRequestError(ParseError):
    code: str = "invalid_request"

    def __init__(
        self,
        missing_fields: Optional[list[str]] = None,
        invalid_fields: Optional[dict[str, str]] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        parts = []
        if missing_fields:
            parts.append(f"Missing fields: {missing_fields}")
        if invalid_fields:
            parts.append(f"Invalid fields: {invalid_fields}")
        msg = message or f"Invalid tool request. {'; '.join(parts)}"
        super().__init__(msg, details)
        self.missing_fields = missing_fields or []
        self.invalid_fields = invalid_fields or {}
        self.details.update({
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
        })


class InvalidResponseError(NonRetryableError):
    category: ErrorCategory = ErrorCategory.INVALID_RESPONSE
    code: str = "invalid_response_error"


class ResponseSchemaError(InvalidResponseError):
    code: str = "response_schema_error"

    def __init__(
        self,
        expected_schema: str,
        actual_response: Any,
        validation_errors: Optional[list[str]] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        msg = message or f"Tool response does not match expected schema"
        super().__init__(msg, details)
        self.expected_schema = expected_schema
        self.actual_response = actual_response
        self.validation_errors = validation_errors or []
        self.details.update({
            "expected_schema": expected_schema,
            "validation_errors": self.validation_errors,
        })
