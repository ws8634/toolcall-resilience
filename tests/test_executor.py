from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pytest

from toolcall_resilience.errors import (
    InvalidRequestError,
    JsonParseError,
    ParameterError,
    ResponseSchemaError,
    TimeoutError,
    ToolExecutionError,
)
from toolcall_resilience.executor import ToolExecutor
from toolcall_resilience.models import (
    AttemptRecord,
    ParameterSchema,
    ParameterType,
    Status,
    ToolRequest,
    ToolResponse,
    ToolSchema,
)
from toolcall_resilience.registry import ToolRegistry
from toolcall_resilience.retry import BackoffStrategy, RetryConfig


class CountingTool:
    def __init__(self, fail_count: int = 0, delay: float = 0.0) -> None:
        self.call_count = 0
        self.fail_count = fail_count
        self.delay = delay
        self.received_params: list[dict[str, Any]] = []

    def __call__(self, value: int = 0) -> dict[str, Any]:
        self.call_count += 1
        self.received_params.append({"value": value})

        if self.delay > 0:
            time.sleep(self.delay)

        if self.call_count <= self.fail_count:
            raise ToolExecutionError(
                message=f"Intentional failure #{self.call_count}",
                details={"attempt": self.call_count},
            )

        return {
            "success": True,
            "call_count": self.call_count,
            "value": value,
        }


class BadFormatTool:
    def __call__(self, return_bad: bool = True) -> Any:
        if return_bad:
            return "this is a string, not an object"
        return {"success": True}


class TestToolExecutor:
    def setup_method(self) -> None:
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(tool_registry=self.registry)

    def register_counting_tool(
        self,
        name: str = "counting_tool",
        fail_count: int = 0,
        delay: float = 0.0,
    ) -> CountingTool:
        tool = CountingTool(fail_count=fail_count, delay=delay)
        schema = ToolSchema(
            name=name,
            description="A counting tool for testing",
            parameters=[
                ParameterSchema(
                    name="value",
                    type_=ParameterType.INTEGER,
                    required=False,
                    default=0,
                ),
            ],
        )
        self.registry.register(name, tool, schema)
        return tool

    def register_bad_format_tool(self, name: str = "bad_format_tool") -> BadFormatTool:
        tool = BadFormatTool()
        schema = ToolSchema(
            name=name,
            description="A tool that can return bad formats",
            parameters=[
                ParameterSchema(
                    name="return_bad",
                    type_=ParameterType.BOOLEAN,
                    required=False,
                    default=True,
                ),
            ],
        )
        self.registry.register(name, tool, schema)
        return tool

    def test_successful_execution(self) -> None:
        tool = self.register_counting_tool("success_tool")

        response = self.executor.execute(
            {"tool_name": "success_tool", "parameters": {"value": 42}}
        )

        assert response.success is True
        assert response.tool_name == "success_tool"
        assert response.attempt_count == 1
        assert response.final_result == {
            "success": True,
            "call_count": 1,
            "value": 42,
        }
        assert tool.call_count == 1

    def test_parameter_auto_conversion(self) -> None:
        tool = self.register_counting_tool("convert_tool")

        response = self.executor.execute(
            {"tool_name": "convert_tool", "parameters": {"value": "100"}}
        )

        assert response.success is True
        assert response.final_result["value"] == 100
        assert tool.call_count == 1
        assert tool.received_params[0]["value"] == 100

    def test_missing_required_parameter(self) -> None:
        schema = ToolSchema(
            name="required_tool",
            description="Tool with required param",
            parameters=[
                ParameterSchema(
                    name="required_param",
                    type_=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

        def dummy_func(required_param: str) -> dict[str, Any]:
            return {"param": required_param}

        self.registry.register("required_tool", dummy_func, schema)

        response = self.executor.execute(
            {"tool_name": "required_tool", "parameters": {}}
        )

        assert response.success is False
        assert response.attempt_count == 0
        assert response.final_error is not None
        assert response.final_error["category"] == "parameter"
        assert response.final_error["code"] == "missing_parameter"

    def test_invalid_parameter_type_no_auto_fix(self) -> None:
        tool = self.register_counting_tool("invalid_type_tool")

        response = self.executor.execute(
            {"tool_name": "invalid_type_tool", "parameters": {"value": "not_a_number"}},
            auto_fix_parameters=False,
        )

        assert response.success is False
        assert response.attempt_count == 0
        assert response.final_error is not None
        assert response.final_error["category"] == "parameter"

    def test_json_parse_error(self) -> None:
        response = self.executor.execute("this is not valid json at all")

        assert response.success is False
        assert response.final_error is not None
        assert response.final_error["category"] == "parse"

    def test_invalid_request_missing_tool_name(self) -> None:
        response = self.executor.execute({"parameters": {"a": 1}})

        assert response.success is False
        assert response.final_error is not None
        assert response.final_error["code"] == "invalid_request"

    def test_tool_not_registered(self) -> None:
        response = self.executor.execute(
            {"tool_name": "non_existent_tool", "parameters": {}}
        )

        assert response.success is False
        assert response.final_error is not None
        assert response.final_error["category"] == "parameter"

    def test_retry_on_tool_execution_error(self) -> None:
        tool = self.register_counting_tool("flaky_tool", fail_count=2)

        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=0,
            jitter=False,
        )

        response = self.executor.execute(
            {"tool_name": "flaky_tool", "parameters": {"value": 99}},
            retry_config=retry_config,
        )

        assert response.success is True
        assert response.attempt_count == 3
        assert tool.call_count == 3
        assert len(response.attempts) == 3

        assert response.attempts[0].status == Status.FAILED
        assert response.attempts[1].status == Status.FAILED
        assert response.attempts[2].status == Status.SUCCESS

    def test_retry_exhausted_eventually_fails(self) -> None:
        tool = self.register_counting_tool("always_fail_tool", fail_count=5)

        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=0,
            jitter=False,
        )

        response = self.executor.execute(
            {"tool_name": "always_fail_tool", "parameters": {}},
            retry_config=retry_config,
        )

        assert response.success is False
        assert response.attempt_count == 3
        assert tool.call_count == 3
        assert all(a.status == Status.FAILED for a in response.attempts)

    def test_no_retry_on_parameter_error(self) -> None:
        schema = ToolSchema(
            name="param_tool",
            description="",
            parameters=[
                ParameterSchema(
                    name="required",
                    type_=ParameterType.INTEGER,
                    required=True,
                ),
            ],
        )

        def dummy(required: int) -> dict[str, Any]:
            return {"value": required}

        self.registry.register("param_tool", dummy, schema)

        retry_config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=0,
        )

        response = self.executor.execute(
            {"tool_name": "param_tool", "parameters": {}},
            retry_config=retry_config,
        )

        assert response.success is False
        assert response.attempt_count == 0

    def test_response_schema_validation_fails(self) -> None:
        self.register_bad_format_tool()

        response = self.executor.execute(
            {"tool_name": "bad_format_tool", "parameters": {"return_bad": True}},
            validate_response=True,
        )

        assert response.success is False
        assert response.attempt_count == 1
        assert response.final_error is not None
        assert response.final_error["category"] == "invalid_response"

    def test_response_schema_validation_skipped(self) -> None:
        self.register_bad_format_tool()

        response = self.executor.execute(
            {"tool_name": "bad_format_tool", "parameters": {"return_bad": True}},
            validate_response=False,
        )

        assert response.success is True
        assert response.final_result == "this is a string, not an object"

    def test_attempt_records_have_complete_fields(self) -> None:
        tool = self.register_counting_tool("tracking_tool", fail_count=1)

        retry_config = RetryConfig(
            max_attempts=2,
            initial_delay_ms=0,
            jitter=False,
        )

        response = self.executor.execute(
            {"tool_name": "tracking_tool", "parameters": {"value": 100}},
            retry_config=retry_config,
        )

        assert response.success is True
        assert len(response.attempts) == 2

        for i, attempt in enumerate(response.attempts, 1):
            assert attempt.attempt_number == i
            assert attempt.timestamp is not None
            assert attempt.duration_ms >= 0.0
            assert attempt.parameters is not None
            assert "value" in attempt.parameters

        failed_attempt = response.attempts[0]
        assert failed_attempt.status == Status.FAILED
        assert failed_attempt.error is not None
        assert failed_attempt.error["category"] == "tool_execution"

        success_attempt = response.attempts[1]
        assert success_attempt.status == Status.SUCCESS
        assert success_attempt.normalized_response is not None
        assert success_attempt.raw_response is not None

    def test_final_parameters_in_response(self) -> None:
        schema = ToolSchema(
            name="default_tool",
            description="",
            parameters=[
                ParameterSchema(
                    name="required",
                    type_=ParameterType.INTEGER,
                    required=True,
                ),
                ParameterSchema(
                    name="optional",
                    type_=ParameterType.STRING,
                    required=False,
                    default="default_value",
                ),
            ],
        )

        def dummy(required: int, optional: str = "default") -> dict[str, Any]:
            return {"required": required, "optional": optional}

        self.registry.register("default_tool", dummy, schema)

        response = self.executor.execute(
            {"tool_name": "default_tool", "parameters": {"required": 42}}
        )

        assert response.success is True
        assert "required" in response.final_parameters
        assert "optional" in response.final_parameters
        assert response.final_parameters["optional"] == "default_value"

    def test_execution_with_tool_request_object(self) -> None:
        tool = self.register_counting_tool("request_object_tool")

        request = ToolRequest(tool_name="request_object_tool", parameters={"value": 123})

        response = self.executor.execute(request)

        assert response.success is True
        assert response.final_result["value"] == 123

    def test_retryable_error_categories(self) -> None:
        schema = ToolSchema(
            name="timeout_tool",
            description="",
            parameters=[],
        )

        def raise_timeout() -> dict[str, Any]:
            raise TimeoutError(
                message="Timeout occurred",
                timeout_seconds=5.0,
            )

        self.registry.register("timeout_tool", raise_timeout, schema)

        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=0,
            jitter=False,
            retryable_error_categories={"timeout"},
        )

        response = self.executor.execute(
            {"tool_name": "timeout_tool", "parameters": {}},
            retry_config=retry_config,
        )

        assert response.success is False
        assert response.attempt_count == 3

    def test_single_quote_json_repaired(self) -> None:
        tool = self.register_counting_tool("json_repair_tool")

        response = self.executor.execute(
            "{'tool_name': 'json_repair_tool', 'parameters': {'value': 55}}"
        )

        assert response.success is True
        assert response.final_result["value"] == 55

    def test_attempt_duration_recorded(self) -> None:
        tool = self.register_counting_tool("duration_tool", delay=0.01)

        response = self.executor.execute(
            {"tool_name": "duration_tool", "parameters": {}}
        )

        assert response.success is True
        assert len(response.attempts) == 1
        assert response.attempts[0].duration_ms >= 5.0
        assert response.total_duration_ms >= 5.0

    def test_to_summary_dict(self) -> None:
        tool = self.register_counting_tool("summary_tool")

        response = self.executor.execute(
            {"tool_name": "summary_tool", "parameters": {"value": 10}}
        )

        summary = response.to_summary_dict()

        assert summary["success"] is True
        assert summary["tool_name"] == "summary_tool"
        assert summary["attempt_count"] == 1
        assert "final_result" in summary
        assert "attempts" not in summary

    def test_to_detailed_dict(self) -> None:
        tool = self.register_counting_tool("detailed_tool", fail_count=1)

        retry_config = RetryConfig(
            max_attempts=2,
            initial_delay_ms=0,
            jitter=False,
        )

        response = self.executor.execute(
            {"tool_name": "detailed_tool", "parameters": {}},
            retry_config=retry_config,
        )

        detailed = response.to_detailed_dict()

        assert detailed["success"] is True
        assert len(detailed["attempts"]) == 2
        assert "final_parameters" in detailed
