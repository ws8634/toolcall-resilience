from __future__ import annotations

from typing import Any, Callable

import pytest

from toolcall_resilience.errors import (
    ParameterError,
    TimeoutError,
    ToolExecutionError,
)
from toolcall_resilience.retry import BackoffStrategy, RetryConfig, RetryPolicy


class TestRetryConfig:
    def test_default_config(self) -> None:
        config = RetryConfig.default()

        assert config.max_attempts == 3
        assert config.initial_delay_ms == 100.0
        assert config.max_delay_ms == 5000.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert "tool_execution" in config.retryable_error_categories
        assert "timeout" in config.retryable_error_categories

    def test_aggressive_config(self) -> None:
        config = RetryConfig.aggressive()

        assert config.max_attempts == 5
        assert config.initial_delay_ms == 50.0
        assert config.backoff_strategy == BackoffStrategy.LINEAR

    def test_no_retry_config(self) -> None:
        config = RetryConfig.no_retry()

        assert config.max_attempts == 1
        assert config.retryable_error_categories == set()


class TestRetryPolicy:
    def setup_method(self) -> None:
        self.config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=100,
            jitter=False,
        )
        self.policy = RetryPolicy(self.config)
        self.sleep_calls: list[float] = []
        self.policy.set_sleep_func(lambda s: self.sleep_calls.append(s))

    def test_should_retry_first_attempt(self) -> None:
        error = ToolExecutionError(message="test error")

        assert self.policy.should_retry(error, attempt_number=1) is True

    def test_should_not_retry_after_max_attempts(self) -> None:
        error = ToolExecutionError(message="test error")

        assert self.policy.should_retry(error, attempt_number=3) is False

    def test_should_not_retry_parameter_error(self) -> None:
        error = ParameterError(message="invalid parameter")

        assert self.policy.should_retry(error, attempt_number=1) is False

    def test_should_retry_timeout_error(self) -> None:
        error = TimeoutError(message="timeout", timeout_seconds=5.0)

        assert self.policy.should_retry(error, attempt_number=1) is True

    def test_custom_retry_check(self) -> None:
        def custom_check(exc: Exception, attempt: int) -> bool:
            return "retry" in str(exc).lower() and attempt < 2

        config = RetryConfig(
            max_attempts=3,
            custom_retry_check=custom_check,
        )
        policy = RetryPolicy(config)

        retryable_error = ToolExecutionError(message="Please retry this operation")
        non_retryable_error = ToolExecutionError(message="Fatal error, cannot continue")

        assert policy.should_retry(retryable_error, attempt_number=1) is True
        assert policy.should_retry(non_retryable_error, attempt_number=1) is False
        assert policy.should_retry(retryable_error, attempt_number=2) is False

    def test_fixed_backoff_delay(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=100,
            backoff_strategy=BackoffStrategy.FIXED,
            jitter=False,
        )
        policy = RetryPolicy(config)

        assert policy.calculate_delay(attempt_number=1) == 0.0
        assert policy.calculate_delay(attempt_number=2) == 100.0
        assert policy.calculate_delay(attempt_number=3) == 100.0
        assert policy.calculate_delay(attempt_number=4) == 100.0

    def test_linear_backoff_delay(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=100,
            backoff_strategy=BackoffStrategy.LINEAR,
            backoff_multiplier=1.0,
            jitter=False,
        )
        policy = RetryPolicy(config)

        assert policy.calculate_delay(attempt_number=1) == 0.0
        assert policy.calculate_delay(attempt_number=2) == 100.0
        assert policy.calculate_delay(attempt_number=3) == 200.0
        assert policy.calculate_delay(attempt_number=4) == 300.0

    def test_exponential_backoff_delay(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=100,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=False,
        )
        policy = RetryPolicy(config)

        assert policy.calculate_delay(attempt_number=1) == 0.0
        assert policy.calculate_delay(attempt_number=2) == 100.0
        assert policy.calculate_delay(attempt_number=3) == 200.0
        assert policy.calculate_delay(attempt_number=4) == 400.0
        assert policy.calculate_delay(attempt_number=5) == 800.0

    def test_max_delay_limit(self) -> None:
        config = RetryConfig(
            max_attempts=10,
            initial_delay_ms=1000,
            max_delay_ms=5000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=False,
        )
        policy = RetryPolicy(config)

        assert policy.calculate_delay(attempt_number=2) == 1000.0
        assert policy.calculate_delay(attempt_number=3) == 2000.0
        assert policy.calculate_delay(attempt_number=4) == 4000.0
        assert policy.calculate_delay(attempt_number=5) == 5000.0
        assert policy.calculate_delay(attempt_number=6) == 5000.0

    def test_jitter_applies_variation(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=100,
            backoff_strategy=BackoffStrategy.FIXED,
            jitter=True,
            jitter_factor=0.5,
        )
        policy = RetryPolicy(config)

        delays = [policy.calculate_delay(attempt_number=2) for _ in range(20)]

        assert any(d != 100.0 for d in delays)
        assert all(50.0 <= d <= 150.0 for d in delays)

    def test_wait_calls_sleep(self) -> None:
        config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=200,
            backoff_strategy=BackoffStrategy.FIXED,
            jitter=False,
        )
        sleep_calls: list[float] = []
        policy = RetryPolicy(config)
        policy.set_sleep_func(lambda s: sleep_calls.append(s))

        policy.wait(attempt_number=2)

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 0.2

    def test_wait_no_delay_for_first_attempt(self) -> None:
        sleep_calls: list[float] = []
        self.policy.set_sleep_func(lambda s: sleep_calls.append(s))

        self.policy.wait(attempt_number=1)

        assert len(sleep_calls) == 0

    def test_non_retryable_error_codes(self) -> None:
        config = RetryConfig(
            max_attempts=3,
            non_retryable_error_codes={"specific_fatal_error"},
        )
        policy = RetryPolicy(config)

        error = ToolExecutionError(message="test")
        error.code = "specific_fatal_error"

        assert policy.should_retry(error, attempt_number=1) is False

    def test_retryable_error_codes(self) -> None:
        config = RetryConfig(
            max_attempts=3,
            retryable_error_codes={"specific_retryable_error"},
            retryable_error_categories=set(),
        )
        policy = RetryPolicy(config)

        error = ToolExecutionError(message="test")
        error.code = "specific_retryable_error"

        assert policy.should_retry(error, attempt_number=1) is True
