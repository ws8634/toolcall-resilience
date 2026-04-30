from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, TypeVar

from toolcall_resilience.errors import ParameterError
from toolcall_resilience.models import ParameterSchema, RegisteredTool, ToolSchema

F = TypeVar("F", bound=Callable[..., Any])


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        schema: ToolSchema,
        timeout: Optional[float] = None,
    ) -> None:
        is_async = inspect.iscoroutinefunction(func)
        registered_tool = RegisteredTool(
            name=name,
            func=func,
            schema=schema,
            timeout=timeout,
            is_async=is_async,
        )
        self._tools[name] = registered_tool

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> RegisteredTool:
        tool = self.get(name)
        if tool is None:
            raise ParameterError(f"Tool not registered: {name}")
        return tool

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> dict[str, ToolSchema]:
        return {name: tool.schema for name, tool in self._tools.items()}

    def clear(self) -> None:
        self._tools.clear()

    def decorator(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[list[ParameterSchema]] = None,
        timeout: Optional[float] = None,
        strict: bool = False,
    ) -> Callable[[F], F]:
        def wrapper(func: F) -> F:
            tool_name = name or func.__name__
            schema = ToolSchema(
                name=tool_name,
                description=description,
                parameters=parameters or [],
                strict=strict,
            )
            self.register(tool_name, func, schema, timeout=timeout)
            return func

        return wrapper


registry = ToolRegistry()
