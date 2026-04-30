from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from toolcall_resilience.executor import ToolExecutor
from toolcall_resilience.models import ToolResponse
from toolcall_resilience.retry import BackoffStrategy, RetryConfig
from toolcall_resilience.tools.builtin_tools import (
    DeterministicRandom,
    register_builtin_tools,
)


@dataclass
class DemoCase:
    name: str
    description: str
    request: str | dict[str, Any]
    retry_config: Optional[RetryConfig] = None
    validate_response: bool = True
    setup: Optional[Callable[[], None]] = None


def print_header(title: str) -> None:
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_separator() -> None:
    print()
    print("-" * 70)
    print()


def format_response(response: ToolResponse, show_attempts: bool = True) -> str:
    lines = []
    lines.append(f"  Success: {response.success}")
    lines.append(f"  Tool: {response.tool_name}")
    lines.append(f"  Attempts: {response.attempt_count}")
    lines.append(f"  Duration: {response.total_duration_ms:.2f}ms")

    if response.final_result:
        lines.append(f"  Final Result: {json.dumps(response.final_result, ensure_ascii=False, default=str)}")

    if response.final_error:
        err = response.final_error
        lines.append(f"  Final Error:")
        lines.append(f"    Category: {err.get('category', 'unknown')}")
        lines.append(f"    Code: {err.get('code', 'unknown')}")
        lines.append(f"    Message: {err.get('message', '')}")
        lines.append(f"    Is Retryable: {err.get('is_retryable', False)}")

    if show_attempts and response.attempts:
        lines.append("")
        lines.append("  Attempt Details:")
        for i, attempt in enumerate(response.attempts, 1):
            status_symbol = "✓" if attempt.status.value == "success" else "✗"
            lines.append(f"    Attempt {i} [{status_symbol}] ({attempt.duration_ms:.2f}ms):")
            if attempt.error:
                err = attempt.error
                lines.append(f"      Error: {err.get('category', 'unknown')} - {err.get('message', '')}")
            if attempt.normalized_response:
                lines.append(f"      Response: {json.dumps(attempt.normalized_response, ensure_ascii=False, default=str)[:100]}")

    return "\n".join(lines)


def run_demo_case(case: DemoCase, executor: ToolExecutor) -> ToolResponse:
    print(f"📋 Case: {case.name}")
    print(f"   Description: {case.description}")
    print(f"   Request: {case.request if isinstance(case.request, str) else json.dumps(case.request, ensure_ascii=False)}")
    print()

    if case.setup:
        case.setup()

    response = executor.execute(
        request=case.request,
        retry_config=case.retry_config,
        validate_response=case.validate_response,
    )

    print(format_response(response))
    return response


def get_demo_cases() -> list[DemoCase]:
    return [
        DemoCase(
            name="1. 成功调用 (Happy Path)",
            description="稳定成功的工具，参数正确，一次成功",
            request={"tool_name": "stable_success", "parameters": {"input_value": "hello", "multiplier": 2}},
        ),
        DemoCase(
            name="2. 参数自动修复后成功",
            description="参数类型自动转换：字符串'5'转为整数，字符串'true'转为布尔值",
            request={"tool_name": "sensitive_divide", "parameters": {"numerator": "10", "denominator": "2"}},
        ),
        DemoCase(
            name="3. 参数无法修复直接失败",
            description="缺少必需参数，无法自动修复",
            request={"tool_name": "sensitive_divide", "parameters": {"numerator": 10}},
        ),
        DemoCase(
            name="4. 格式解析修复成功",
            description="使用单引号的JSON（非标准），能够自动修复并解析",
            request="{'tool_name': 'stable_success', 'parameters': {'input_value': 'fixed_me'}}",
        ),
        DemoCase(
            name="5. 格式解析修复失败",
            description="完全无法解析的输入",
            request="this is not even close to valid json",
        ),
        DemoCase(
            name="6. Flaky工具重试最终成功",
            description="高失败概率的工具，通过重试机制最终成功（使用固定seed保证可复现）",
            request={"tool_name": "flaky_tool", "parameters": {"fail_probability": 0.6, "value": 99}},
            retry_config=RetryConfig(
                max_attempts=5,
                initial_delay_ms=50,
                max_delay_ms=200,
                backoff_strategy=BackoffStrategy.LINEAR,
                backoff_multiplier=1.0,
                jitter=False,
            ),
            setup=lambda: DeterministicRandom.set_seed(42),
        ),
        DemoCase(
            name="7. 超时触发重试后失败",
            description="工具执行超时，重试后仍然失败（配置短超时）",
            request={"tool_name": "slow_tool", "parameters": {"sleep_seconds": 0.5, "value": "timeout_demo"}},
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay_ms=50,
                max_delay_ms=100,
                timeout_seconds=0.1,
                jitter=False,
            ),
        ),
        DemoCase(
            name="8. 坏格式返回被识别",
            description="工具返回不符合schema的格式（字符串而非对象），被识别为结构化错误",
            request={"tool_name": "bad_format_tool", "parameters": {"return_bad_format": True}},
            validate_response=True,
        ),
        DemoCase(
            name="9. 坏格式返回跳过验证",
            description="关闭响应验证，即使格式不对也接受",
            request={"tool_name": "bad_format_tool", "parameters": {"return_bad_format": True}},
            validate_response=False,
        ),
    ]


def run_demo(verbose: bool = False) -> int:
    print_header("🔧 工具调用容错机制 Demo")
    print()
    print("本Demo展示工具调用容错机制的各个关键特性：")
    print("  ✓ 参数自动校验与转换")
    print("  ✓ JSON格式自动修复")
    print("  ✓ 自动重试策略")
    print("  ✓ 超时处理")
    print("  ✓ 响应格式校验")
    print()
    print("所有测试用例使用确定性随机种子，结果可复现。")
    print()

    DeterministicRandom.set_seed(12345)

    print_header("📦 注册内置工具")
    register_builtin_tools()
    from toolcall_resilience.registry import registry

    print(f"已注册工具: {registry.list_tools()}")
    print()

    executor = ToolExecutor()

    demo_cases = get_demo_cases()
    results: list[tuple[str, bool]] = []

    for i, case in enumerate(demo_cases, 1):
        print_header(f"Case {i}/{len(demo_cases)}")
        try:
            response = run_demo_case(case, executor)

            expected_success = True
            if case.name in [
                "3. 参数无法修复直接失败",
                "5. 格式解析修复失败",
                "7. 超时触发重试后失败",
                "8. 坏格式返回被识别",
            ]:
                expected_success = False

            passed = response.success == expected_success
            results.append((case.name, passed))

            print()
            if passed:
                print(f"  ✅ Case PASSED")
            else:
                print(f"  ❌ Case FAILED (expected success={expected_success}, got success={response.success})")

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((case.name, False))

        print_separator()

    print_header("📊 Demo 结果汇总")
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    print()
    print(f"  总计: {total} 个用例")
    print(f"  通过: {passed} 个")
    print(f"  失败: {failed} 个")
    print()

    if failed == 0:
        print("🎉 所有Demo用例通过！")
        return 0
    else:
        print(f"⚠️  {failed} 个用例失败")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="工具调用容错机制 - Demo & 测试入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m toolcall_resilience              # 运行完整Demo
  python -m toolcall_resilience --demo       # 运行完整Demo
  python -m toolcall_resilience --verbose    # 运行Demo并显示详细信息
        """,
    )
    parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        default=True,
        help="运行容错机制Demo (默认)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细输出",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="设置随机种子 (默认: 12345)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        DeterministicRandom.set_seed(args.seed)

    if args.demo:
        sys.exit(run_demo(verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
