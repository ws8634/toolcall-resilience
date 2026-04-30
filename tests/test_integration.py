from __future__ import annotations

import pytest

from toolcall_resilience.executor import ToolExecutor
from toolcall_resilience.models import ParameterSchema, ParameterType, ToolSchema
from toolcall_resilience.registry import ToolRegistry
from toolcall_resilience.retry import RetryConfig
from toolcall_resilience.tools.builtin_tools import (
    DeterministicRandom,
    register_builtin_tools,
)


class TestBuiltinTools:
    def setup_method(self) -> None:
        register_builtin_tools()
        from toolcall_resilience.registry import registry

        self.registry = registry
        self.executor = ToolExecutor(tool_registry=registry)

    def teardown_method(self) -> None:
        self.registry.clear()

    def test_stable_success_tool(self) -> None:
        response = self.executor.execute(
            {"tool_name": "stable_success", "parameters": {"input_value": "test", "multiplier": 3}}
        )

        assert response.success is True
        assert response.final_result["original_value"] == "test"
        assert response.final_result["multiplied_value"] is None

    def test_sensitive_divide_tool(self) -> None:
        response = self.executor.execute(
            {"tool_name": "sensitive_divide", "parameters": {"numerator": 10, "denominator": 2}}
        )

        assert response.success is True
        assert response.final_result["result"] == 5.0

    def test_sensitive_divide_by_zero(self) -> None:
        no_retry_config = RetryConfig.no_retry()
        response = self.executor.execute(
            {"tool_name": "sensitive_divide", "parameters": {"numerator": 10, "denominator": 0}},
            retry_config=no_retry_config,
        )

        assert response.success is False
        assert response.attempt_count == 1
        assert response.final_error is not None
        assert response.final_error["category"] == "tool_execution"

    def test_get_field_tool(self) -> None:
        response = self.executor.execute(
            {
                "tool_name": "get_field",
                "parameters": {"data": {"name": "Alice", "age": 30}, "field": "name"},
            }
        )

        assert response.success is True
        assert response.final_result["value"] == "Alice"

    def test_get_field_with_default(self) -> None:
        response = self.executor.execute(
            {
                "tool_name": "get_field",
                "parameters": {"data": {"name": "Alice"}, "field": "missing", "default": "fallback"},
            }
        )

        assert response.success is True
        assert response.final_result["value"] == "fallback"
        assert response.final_result["found"] is False

    def test_flaky_tool_deterministic(self) -> None:
        DeterministicRandom.set_seed(42)

        retry_config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=0,
            jitter=False,
        )

        response = self.executor.execute(
            {"tool_name": "flaky_tool", "parameters": {"fail_probability": 0.6, "value": 99}},
            retry_config=retry_config,
        )

        assert response.success is True
        assert response.final_result["value"] == 99


class TestRegistry:
    def setup_method(self) -> None:
        self.registry = ToolRegistry()

    def teardown_method(self) -> None:
        self.registry.clear()

    def test_register_and_get(self) -> None:
        def my_tool(x: int) -> dict[str, Any]:
            return {"result": x * 2}

        schema = ToolSchema(
            name="my_tool",
            description="Doubles a number",
            parameters=[
                ParameterSchema(name="x", type_=ParameterType.INTEGER, required=True),
            ],
        )

        self.registry.register("my_tool", my_tool, schema)

        registered = self.registry.get("my_tool")
        assert registered is not None
        assert registered.name == "my_tool"
        assert registered.func == my_tool
        assert registered.schema.description == "Doubles a number"

    def test_get_non_existent_raises(self) -> None:
        from toolcall_resilience.errors import ParameterError

        with pytest.raises(ParameterError):
            self.registry.get_or_raise("non_existent")

    def test_list_tools(self) -> None:
        def tool1() -> dict[str, Any]:
            return {}

        def tool2() -> dict[str, Any]:
            return {}

        schema1 = ToolSchema(name="tool1", description="", parameters=[])
        schema2 = ToolSchema(name="tool2", description="", parameters=[])

        self.registry.register("tool1", tool1, schema1)
        self.registry.register("tool2", tool2, schema2)

        assert set(self.registry.list_tools()) == {"tool1", "tool2"}

    def test_decorator_registration(self) -> None:
        @self.registry.decorator(
            name="decorated_tool",
            description="A decorated tool",
            parameters=[
                ParameterSchema(name="value", type_=ParameterType.STRING, required=True),
            ],
        )
        def my_decorated_tool(value: str) -> dict[str, Any]:
            return {"value": value.upper()}

        registered = self.registry.get("decorated_tool")
        assert registered is not None
        assert registered.name == "decorated_tool"
        assert registered.schema.description == "A decorated tool"

        result = registered.func(value="hello")
        assert result == {"value": "HELLO"}
