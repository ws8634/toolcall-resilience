from __future__ import annotations

import asyncio
import concurrent.futures
import time
import traceback
from typing import Any, Callable, Optional, TypeVar

from toolcall_resilience.errors import (
    InvalidResponseError,
    ParameterError,
    ParseError,
    TimeoutError,
    ToolExecutionError,
    ToolcallError,
)
from toolcall_resilience.models import (
    AttemptRecord,
    Status,
    ToolRequest,
    ToolResponse,
)
from toolcall_resilience.parsing import RequestParser, ResponseValidator
from toolcall_resilience.registry import ToolRegistry, registry
from toolcall_resilience.retry import RetryConfig, RetryPolicy
from toolcall_resilience.validation import ParameterValidator

T = TypeVar("T")


class ToolExecutor:
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        retry_config: Optional[RetryConfig] = None,
        parameter_validator: Optional[ParameterValidator] = None,
        request_parser: Optional[RequestParser] = None,
        response_validator: Optional[ResponseValidator] = None,
    ) -> None:
        self.registry = tool_registry or registry
        self.retry_config = retry_config or RetryConfig.default()
        self.parameter_validator = parameter_validator or ParameterValidator()
        self.request_parser = request_parser or RequestParser()
        self.response_validator = response_validator or ResponseValidator()
        self._time_func: Callable[[], float] = time.perf_counter

    def set_time_func(self, time_func: Callable[[], float]) -> None:
        self._time_func = time_func

    def execute(
        self,
        request: str | dict[str, Any] | ToolRequest,
        retry_config: Optional[RetryConfig] = None,
        auto_fix_parameters: bool = True,
        validate_response: bool = True,
    ) -> ToolResponse:
        total_start_time = self._time_func()
        config = retry_config or self.retry_config
        retry_policy = RetryPolicy(config)

        tool_response = ToolResponse(
            success=False,
            tool_name="",
            request_id=None,
        )

        try:
            parsed_request = self._parse_request(request)
            tool_response.tool_name = parsed_request.tool_name
            tool_response.request_id = parsed_request.id_

            registered_tool = self.registry.get_or_raise(parsed_request.tool_name)

            validation_result = self.parameter_validator.validate(
                parsed_request.parameters,
                registered_tool.schema,
                auto_fix=auto_fix_parameters,
            )

            if not validation_result.is_valid:
                first_error = validation_result.errors[0]
                tool_response.final_parameters = validation_result.validated_parameters
                tool_response.final_error = (
                    first_error.to_dict() if isinstance(first_error, ToolcallError) else {"message": str(first_error)}
                )
                tool_response.total_duration_ms = (self._time_func() - total_start_time) * 1000
                return tool_response

            final_params = validation_result.validated_parameters
            tool_response.final_parameters = final_params

            attempt_number = 1
            last_exception: Optional[Exception] = None

            while attempt_number <= config.max_attempts:
                attempt_start_time = self._time_func()

                try:
                    timeout = config.timeout_seconds or registered_tool.timeout

                    if registered_tool.is_async:
                        raw_response = self._run_async_with_timeout(
                            registered_tool.func,
                            final_params,
                            timeout,
                        )
                    else:
                        raw_response = self._run_sync_with_timeout(
                            registered_tool.func,
                            final_params,
                            timeout,
                        )

                    if validate_response:
                        try:
                            normalized_response = self.response_validator.validate(
                                raw_response,
                                registered_tool.schema,
                            )
                        except InvalidResponseError as e:
                            attempt_duration = (self._time_func() - attempt_start_time) * 1000
                            attempt_record = AttemptRecord.from_exception(
                                attempt_number=attempt_number,
                                exception=e,
                                parameters=final_params,
                                duration_ms=attempt_duration,
                            )
                            tool_response.attempts.append(attempt_record)
                            tool_response.attempt_count += 1
                            last_exception = e

                            if retry_policy.should_retry(e, attempt_number):
                                retry_policy.wait(attempt_number + 1)
                                attempt_number += 1
                                continue
                            break

                    else:
                        normalized_response = raw_response

                    attempt_duration = (self._time_func() - attempt_start_time) * 1000

                    attempt_record = AttemptRecord.from_success(
                        attempt_number=attempt_number,
                        response=raw_response,
                        normalized_response=normalized_response,
                        parameters=final_params,
                        duration_ms=attempt_duration,
                    )
                    tool_response.attempts.append(attempt_record)
                    tool_response.attempt_count += 1

                    tool_response.success = True
                    tool_response.final_result = normalized_response
                    tool_response.total_duration_ms = (self._time_func() - total_start_time) * 1000

                    return tool_response

                except ToolcallError as e:
                    attempt_duration = (self._time_func() - attempt_start_time) * 1000
                    attempt_record = AttemptRecord.from_exception(
                        attempt_number=attempt_number,
                        exception=e,
                        parameters=final_params,
                        duration_ms=attempt_duration,
                    )
                    tool_response.attempts.append(attempt_record)
                    tool_response.attempt_count += 1
                    last_exception = e

                    if retry_policy.should_retry(e, attempt_number):
                        retry_policy.wait(attempt_number + 1)
                        attempt_number += 1
                        continue
                    break

                except Exception as e:
                    attempt_duration = (self._time_func() - attempt_start_time) * 1000
                    wrapped_error = ToolExecutionError(
                        message=str(e),
                        details={"exception_type": type(e).__name__},
                        original_exception=e,
                    )
                    attempt_record = AttemptRecord.from_exception(
                        attempt_number=attempt_number,
                        exception=wrapped_error,
                        parameters=final_params,
                        duration_ms=attempt_duration,
                    )
                    tool_response.attempts.append(attempt_record)
                    tool_response.attempt_count += 1
                    last_exception = wrapped_error

                    if retry_policy.should_retry(wrapped_error, attempt_number):
                        retry_policy.wait(attempt_number + 1)
                        attempt_number += 1
                        continue
                    break

            tool_response.success = False
            if last_exception:
                if isinstance(last_exception, ToolcallError):
                    tool_response.final_error = last_exception.to_dict()
                else:
                    tool_response.final_error = {
                        "category": "unknown",
                        "code": "unexpected_error",
                        "message": str(last_exception),
                        "is_retryable": False,
                    }
            tool_response.total_duration_ms = (self._time_func() - total_start_time) * 1000

            return tool_response

        except (ParseError, ParameterError) as e:
            tool_response.total_duration_ms = (self._time_func() - total_start_time) * 1000
            tool_response.final_error = e.to_dict()
            return tool_response

        except Exception as e:
            tool_response.total_duration_ms = (self._time_func() - total_start_time) * 1000
            tool_response.final_error = {
                "category": "unknown",
                "code": "unexpected_error",
                "message": str(e),
                "details": {"exception_type": type(e).__name__},
                "is_retryable": False,
            }
            return tool_response

    def _parse_request(self, request: str | dict[str, Any] | ToolRequest) -> ToolRequest:
        if isinstance(request, ToolRequest):
            return request
        return self.request_parser.parse(request)

    def _run_sync_with_timeout(
        self,
        func: Callable[..., Any],
        params: dict[str, Any],
        timeout: Optional[float],
    ) -> Any:
        if timeout is None:
            return func(**params)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **params)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    message=f"Tool execution timed out after {timeout} seconds",
                    timeout_seconds=timeout,
                )

    def _run_async_with_timeout(
        self,
        func: Callable[..., Any],
        params: dict[str, Any],
        timeout: Optional[float],
    ) -> Any:
        async def run_with_timeout() -> Any:
            coro = func(**params)
            if timeout is not None:
                try:
                    return await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        message=f"Async tool execution timed out after {timeout} seconds",
                        timeout_seconds=timeout,
                    )
            return await coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError("Cannot run async tools from a running event loop")

        return asyncio.run(run_with_timeout())


executor = ToolExecutor()
