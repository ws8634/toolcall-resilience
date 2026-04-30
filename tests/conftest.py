from __future__ import annotations

from typing import Any

import pytest

from toolcall_resilience.executor import ToolExecutor
from toolcall_resilience.models import ParameterSchema, ParameterType, ToolSchema
from toolcall_resilience.registry import ToolRegistry
from toolcall_resilience.tools.builtin_tools import DeterministicRandom, register_builtin_tools


@pytest.fixture
def clean_registry() -> ToolRegistry:
    registry = ToolRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def builtin_registry() -> ToolRegistry:
    register_builtin_tools()
    from toolcall_resilience.registry import registry

    yield registry
    registry.clear()


@pytest.fixture
def executor(builtin_registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(tool_registry=builtin_registry)


@pytest.fixture
def deterministic_random() -> None:
    DeterministicRandom.set_seed(42)
    yield
    DeterministicRandom.reset()


@pytest.fixture
def simple_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="simple_tool",
        description="A simple test tool",
        parameters=[
            ParameterSchema(
                name="value",
                type_=ParameterType.INTEGER,
                required=True,
                description="An integer value",
                minimum=0,
                maximum=100,
            ),
            ParameterSchema(
                name="name",
                type_=ParameterType.STRING,
                required=False,
                default="default",
                description="A string name",
            ),
        ],
    )


class MockTime:
    def __init__(self) -> None:
        self._current_time = 0.0
        self._sleep_calls: list[float] = []

    def time(self) -> float:
        return self._current_time

    def sleep(self, seconds: float) -> None:
        self._sleep_calls.append(seconds)
        self._current_time += seconds

    @property
    def sleep_calls(self) -> list[float]:
        return self._sleep_calls

    def advance(self, seconds: float) -> None:
        self._current_time += seconds


@pytest.fixture
def mock_time() -> MockTime:
    return MockTime()
