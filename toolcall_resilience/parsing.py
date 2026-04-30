from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from toolcall_resilience.errors import (
    InvalidRequestError,
    JsonParseError,
    ResponseSchemaError,
)
from toolcall_resilience.models import ParameterType, ToolRequest, ToolSchema


class JsonRepairResult:
    def __init__(
        self,
        success: bool,
        repaired: Optional[str] = None,
        parsed: Optional[Any] = None,
        original_error: Optional[Exception] = None,
        repair_attempts: int = 0,
    ) -> None:
        self.success = success
        self.repaired = repaired
        self.parsed = parsed
        self.original_error = original_error
        self.repair_attempts = repair_attempts


class JsonParser:
    def __init__(self, allow_repair: bool = True) -> None:
        self.allow_repair = allow_repair

    def parse(self, raw_input: str, try_repair: bool = True) -> Any:
        try:
            return json.loads(raw_input)
        except json.JSONDecodeError as e:
            if try_repair and self.allow_repair:
                result = self.try_repair(raw_input, e)
                if result.success and result.parsed is not None:
                    return result.parsed
            raise JsonParseError(
                raw_input=raw_input,
                parse_position=e.pos if hasattr(e, "pos") else None,
                message=str(e),
                original_exception=e,
            ) from e

    def try_repair(self, raw_input: str, original_error: json.JSONDecodeError) -> JsonRepairResult:
        repair_attempts = 0
        current = raw_input

        repair_rounds = [
            [self._extract_json_from_markdown],
            [self._add_missing_quotes, self._replace_single_quotes],
            [self._fix_trailing_commas],
            [self._fix_unclosed_brackets],
        ]

        for round_strategies in repair_rounds:
            for strategy in round_strategies:
                try:
                    repaired = strategy(current)
                    if repaired != current:
                        repair_attempts += 1
                        try:
                            parsed = json.loads(repaired)
                            return JsonRepairResult(
                                success=True,
                                repaired=repaired,
                                parsed=parsed,
                                original_error=original_error,
                                repair_attempts=repair_attempts,
                            )
                        except (json.JSONDecodeError, ValueError):
                            current = repaired
                            continue
                except (json.JSONDecodeError, ValueError):
                    continue

        try:
            parsed = json.loads(current)
            return JsonRepairResult(
                success=True,
                repaired=current,
                parsed=parsed,
                original_error=original_error,
                repair_attempts=repair_attempts,
            )
        except (json.JSONDecodeError, ValueError):
            pass

        return JsonRepairResult(
            success=False,
            original_error=original_error,
            repair_attempts=repair_attempts,
        )

    def _replace_single_quotes(self, s: str) -> str:
        def replace_quotes(match: re.Match[str]) -> str:
            content = match.group(0)
            if content.startswith("'") and content.endswith("'"):
                inner = content[1:-1].replace('"', '\\"').replace("\\'", "'")
                return f'"{inner}"'
            return content

        pattern = r"'(?:[^'\\]|\\.)*'"
        return re.sub(pattern, replace_quotes, s)

    def _add_missing_quotes(self, s: str) -> str:
        s = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', s)
        return s

    def _fix_trailing_commas(self, s: str) -> str:
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        return s

    def _fix_unclosed_brackets(self, s: str) -> str:
        s = s.strip()

        open_braces = s.count("{")
        close_braces = s.count("}")
        open_brackets = s.count("[")
        close_brackets = s.count("]")

        while close_braces < open_braces:
            s += "}"
            close_braces += 1

        while close_brackets < open_brackets:
            s += "]"
            close_brackets += 1

        return s

    def _extract_json_from_markdown(self, s: str) -> str:
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_block_pattern, s)
        if match:
            return match.group(1).strip()

        json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        match = re.search(json_pattern, s)
        if match:
            return match.group(1).strip()

        return s


class RequestParser:
    def __init__(self, json_parser: Optional[JsonParser] = None) -> None:
        self.json_parser = json_parser or JsonParser()

    def parse(self, raw_input: str | dict[str, Any], try_repair: bool = True) -> ToolRequest:
        if isinstance(raw_input, dict):
            return self._validate_dict(raw_input)

        try:
            parsed = self.json_parser.parse(raw_input, try_repair=try_repair)
        except JsonParseError:
            raise

        if not isinstance(parsed, dict):
            raise InvalidRequestError(
                message=f"Expected object/dict, got {type(parsed).__name__}",
                details={"parsed_type": type(parsed).__name__},
            )

        return self._validate_dict(parsed)

    def _validate_dict(self, data: dict[str, Any]) -> ToolRequest:
        try:
            return ToolRequest.model_validate(data)
        except ValidationError as e:
            missing_fields = []
            invalid_fields = {}

            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                error_type = error["type"]

                if error_type == "missing":
                    missing_fields.append(field)
                else:
                    invalid_fields[field] = error["msg"]

            raise InvalidRequestError(
                missing_fields=missing_fields if missing_fields else None,
                invalid_fields=invalid_fields if invalid_fields else None,
                details={"validation_errors": [err["msg"] for err in e.errors()]},
            ) from e


class ResponseValidator:
    def validate(
        self,
        response: Any,
        expected_schema: Optional[ToolSchema] = None,
        expected_type: Optional[ParameterType] = None,
    ) -> Any:
        target_type = expected_type or (expected_schema.return_type if expected_schema else ParameterType.OBJECT)

        if self._matches_type(response, target_type):
            return response

        actual_type = self._get_response_type(response)

        raise ResponseSchemaError(
            expected_schema=target_type.value,
            actual_response=response,
            validation_errors=[f"Expected {target_type.value}, got {actual_type}"],
        )

    def _matches_type(self, value: Any, expected_type: ParameterType) -> bool:
        if value is None:
            return expected_type == ParameterType.NULL

        type_checks: dict[ParameterType, tuple[type, ...]] = {
            ParameterType.STRING: (str,),
            ParameterType.INTEGER: (int,),
            ParameterType.NUMBER: (int, float),
            ParameterType.BOOLEAN: (bool,),
            ParameterType.OBJECT: (dict,),
            ParameterType.ARRAY: (list,),
        }

        if expected_type in type_checks:
            return isinstance(value, type_checks[expected_type])

        return True

    def _get_response_type(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return type(value).__name__
