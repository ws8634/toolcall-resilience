from __future__ import annotations

import random
import time
from typing import Any, Optional

from toolcall_resilience.errors import ToolExecutionError
from toolcall_resilience.models import ParameterSchema, ParameterType, ToolSchema
from toolcall_resilience.registry import registry


class DeterministicRandom:
    _instance: Optional["DeterministicRandom"] = None
    _seed: Optional[int] = None

    @classmethod
    def get_instance(cls) -> "DeterministicRandom":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_seed(cls, seed: int) -> None:
        cls._seed = seed
        random.seed(seed)

    @classmethod
    def reset(cls) -> None:
        cls._seed = None
        random.seed()

    def random(self) -> float:
        return random.random()

    def randint(self, a: int, b: int) -> int:
        return random.randint(a, b)

    def choice(self, seq: list[Any]) -> Any:
        return random.choice(seq)


_rand = DeterministicRandom.get_instance()


def stable_success(input_value: Any, multiplier: int = 1) -> dict[str, Any]:
    return {
        "success": True,
        "original_value": input_value,
        "multiplied_value": input_value * multiplier if isinstance(input_value, (int, float)) else None,
        "message": "Operation completed successfully",
    }


def flaky_tool(
    fail_probability: float = 0.7,
    value: int = 42,
) -> dict[str, Any]:
    if _rand.random() < fail_probability:
        raise ToolExecutionError(
            message=f"Flaky tool failed intentionally (probability: {fail_probability})",
            details={"fail_probability": fail_probability},
        )
    return {
        "success": True,
        "value": value,
        "message": "Flaky tool succeeded this time",
    }


def slow_tool(sleep_seconds: float = 2.0, value: str = "slow_result") -> dict[str, Any]:
    time.sleep(sleep_seconds)
    return {
        "success": True,
        "slept_seconds": sleep_seconds,
        "value": value,
    }


def bad_format_tool(return_bad_format: bool = True) -> Any:
    if return_bad_format:
        return "this_is_a_string_not_an_object"
    return {"success": True, "message": "Good format response"}


def sensitive_divide(numerator: float, denominator: float) -> dict[str, Any]:
    if denominator == 0:
        raise ToolExecutionError(
            message="Division by zero",
            details={"numerator": numerator, "denominator": denominator},
        )
    return {
        "success": True,
        "numerator": numerator,
        "denominator": denominator,
        "result": numerator / denominator,
    }


def get_field(data: dict[str, Any], field: str, default: Optional[Any] = None) -> dict[str, Any]:
    if field in data:
        return {
            "success": True,
            "field": field,
            "value": data[field],
            "found": True,
        }
    if default is not None:
        return {
            "success": True,
            "field": field,
            "value": default,
            "found": False,
            "used_default": True,
        }
    raise ToolExecutionError(
        message=f"Field '{field}' not found in data and no default provided",
        details={"available_fields": list(data.keys()), "requested_field": field},
    )


def register_builtin_tools() -> None:
    registry.register(
        name="stable_success",
        func=stable_success,
        schema=ToolSchema(
            name="stable_success",
            description="A tool that always succeeds with a predictable response",
            parameters=[
                ParameterSchema(
                    name="input_value",
                    type_=ParameterType.STRING,
                    required=True,
                    description="The input value to process",
                ),
                ParameterSchema(
                    name="multiplier",
                    type_=ParameterType.INTEGER,
                    required=False,
                    default=1,
                    description="Multiplier for numeric values",
                    minimum=0,
                ),
            ],
        ),
    )

    registry.register(
        name="flaky_tool",
        func=flaky_tool,
        schema=ToolSchema(
            name="flaky_tool",
            description="A tool that randomly fails with configurable probability",
            parameters=[
                ParameterSchema(
                    name="fail_probability",
                    type_=ParameterType.NUMBER,
                    required=False,
                    default=0.7,
                    description="Probability of failure (0.0 to 1.0)",
                    minimum=0.0,
                    maximum=1.0,
                ),
                ParameterSchema(
                    name="value",
                    type_=ParameterType.INTEGER,
                    required=False,
                    default=42,
                    description="Value to return on success",
                ),
            ],
        ),
    )

    registry.register(
        name="slow_tool",
        func=slow_tool,
        schema=ToolSchema(
            name="slow_tool",
            description="A tool that sleeps for a configurable duration before returning",
            parameters=[
                ParameterSchema(
                    name="sleep_seconds",
                    type_=ParameterType.NUMBER,
                    required=False,
                    default=2.0,
                    description="Number of seconds to sleep",
                    minimum=0.0,
                ),
                ParameterSchema(
                    name="value",
                    type_=ParameterType.STRING,
                    required=False,
                    default="slow_result",
                    description="Value to return after sleeping",
                ),
            ],
        ),
        timeout=30.0,
    )

    registry.register(
        name="bad_format_tool",
        func=bad_format_tool,
        schema=ToolSchema(
            name="bad_format_tool",
            description="A tool that can return invalid response formats",
            parameters=[
                ParameterSchema(
                    name="return_bad_format",
                    type_=ParameterType.BOOLEAN,
                    required=False,
                    default=True,
                    description="If True, returns a string instead of object",
                ),
            ],
        ),
    )

    registry.register(
        name="sensitive_divide",
        func=sensitive_divide,
        schema=ToolSchema(
            name="sensitive_divide",
            description="A division tool that is sensitive to parameter values",
            parameters=[
                ParameterSchema(
                    name="numerator",
                    type_=ParameterType.NUMBER,
                    required=True,
                    description="The numerator (dividend)",
                ),
                ParameterSchema(
                    name="denominator",
                    type_=ParameterType.NUMBER,
                    required=True,
                    description="The denominator (divisor) - cannot be zero",
                ),
            ],
        ),
    )

    registry.register(
        name="get_field",
        func=get_field,
        schema=ToolSchema(
            name="get_field",
            description="Extract a field from a dictionary with optional default",
            parameters=[
                ParameterSchema(
                    name="data",
                    type_=ParameterType.OBJECT,
                    required=True,
                    description="The dictionary to extract from",
                ),
                ParameterSchema(
                    name="field",
                    type_=ParameterType.STRING,
                    required=True,
                    description="The field name to extract",
                ),
                ParameterSchema(
                    name="default",
                    type_=ParameterType.STRING,
                    required=False,
                    default=None,
                    description="Default value if field not found",
                ),
            ],
        ),
    )
