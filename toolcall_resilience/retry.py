from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type

from toolcall_resilience.errors import (
    InvalidResponseError,
    ParameterError,
    ParseError,
    RetryableError,
    TimeoutError,
    ToolExecutionError,
    ToolcallError,
)


class BackoffStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    timeout_seconds: Optional[float] = None
    retryable_error_categories: set[str] = field(
        default_factory=lambda: {
            "tool_execution",
            "timeout",
        }
    )
    retryable_error_codes: set[str] = field(default_factory=set)
    non_retryable_error_codes: set[str] = field(default_factory=set)
    custom_retry_check: Optional[Callable[[Exception, int], bool]] = None

    @classmethod
    def default(cls) -> "RetryConfig":
        return cls()

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        return cls(
            max_attempts=5,
            initial_delay_ms=50.0,
            max_delay_ms=2000.0,
            backoff_strategy=BackoffStrategy.LINEAR,
            backoff_multiplier=1.5,
            jitter=True,
        )

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        return cls(
            max_attempts=1,
            initial_delay_ms=0.0,
            max_delay_ms=0.0,
            retryable_error_categories=set(),
        )


class RetryPolicy:
    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig.default()
        self._sleep_func: Callable[[float], None] = time.sleep

    def set_sleep_func(self, sleep_func: Callable[[float], None]) -> None:
        self._sleep_func = sleep_func

    def should_retry(self, exception: Exception, attempt_number: int) -> bool:
        if attempt_number >= self.config.max_attempts:
            return False

        if self.config.custom_retry_check:
            return self.config.custom_retry_check(exception, attempt_number)

        if isinstance(exception, ToolcallError):
            if not exception.is_retryable:
                return False

            error_category = exception.category.value
            error_code = exception.code

            if error_code in self.config.non_retryable_error_codes:
                return False

            if (
                error_category in self.config.retryable_error_categories
                or error_code in self.config.retryable_error_codes
            ):
                return True

            return exception.is_retryable

        return False

    def calculate_delay(self, attempt_number: int) -> float:
        if attempt_number <= 1:
            return 0.0

        base_delay = self.config.initial_delay_ms

        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = base_delay
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = base_delay * (attempt_number - 1)
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (self.config.backoff_multiplier ** (attempt_number - 2))
        else:
            delay = base_delay

        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        delay = max(0.0, min(delay, self.config.max_delay_ms))

        return delay

    def wait(self, attempt_number: int) -> float:
        delay_ms = self.calculate_delay(attempt_number)
        if delay_ms > 0:
            self._sleep_func(delay_ms / 1000.0)
        return delay_ms
