from __future__ import annotations

import json
from typing import Any, Optional

from toolcall_resilience.errors import (
    ExtraParameterError,
    MissingParameterError,
    ParameterTypeError,
    ParameterValueError,
)
from toolcall_resilience.models import ParameterSchema, ParameterType, ToolSchema


class ParameterValidationResult:
    def __init__(self) -> None:
        self.validated_parameters: dict[str, Any] = {}
        self.errors: list[Exception] = []
        self.warnings: list[str] = []
        self.fixed_count: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, error: Exception) -> None:
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def add_fixed(self, parameter_name: str, original_value: Any, fixed_value: Any) -> None:
        self.fixed_count += 1
        self.add_warning(f"Parameter '{parameter_name}': fixed value from {original_value!r} to {fixed_value!r}")


class ParameterValidator:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def validate(
        self,
        parameters: dict[str, Any],
        schema: ToolSchema,
        auto_fix: bool = True,
    ) -> ParameterValidationResult:
        result = ParameterValidationResult()

        expected_params = {p.name: p for p in schema.parameters}
        provided_names = set(parameters.keys())

        for param_schema in schema.parameters:
            param_name = param_schema.name
            if param_name not in provided_names:
                if param_schema.required:
                    if param_schema.default is not None:
                        result.validated_parameters[param_name] = param_schema.default
                        result.add_warning(
                            f"Parameter '{param_name}': using default value {param_schema.default!r}"
                        )
                    else:
                        result.add_error(MissingParameterError(param_name))
                else:
                    if param_schema.default is not None:
                        result.validated_parameters[param_name] = param_schema.default
                        result.add_warning(
                            f"Parameter '{param_name}': using default value {param_schema.default!r}"
                        )

        if schema.strict or self.strict:
            for param_name in provided_names:
                if param_name not in expected_params:
                    result.add_error(ExtraParameterError(param_name))
        else:
            for param_name in provided_names:
                if param_name not in expected_params:
                    result.add_warning(f"Ignoring unexpected parameter: {param_name}")

        for param_name, param_value in parameters.items():
            if param_name not in expected_params:
                continue

            param_schema = expected_params[param_name]
            validation_result = self._validate_single_parameter(
                param_name, param_value, param_schema, auto_fix
            )

            if validation_result.errors:
                result.errors.extend(validation_result.errors)
            else:
                result.validated_parameters[param_name] = validation_result.validated_parameters.get(
                    param_name, param_value
                )
                result.warnings.extend(validation_result.warnings)
                result.fixed_count += validation_result.fixed_count

        return result

    def _validate_single_parameter(
        self,
        param_name: str,
        param_value: Any,
        param_schema: ParameterSchema,
        auto_fix: bool,
    ) -> ParameterValidationResult:
        result = ParameterValidationResult()

        actual_type = self._get_value_type(param_value)

        if actual_type == param_schema.type_:
            result.validated_parameters[param_name] = param_value
            self._validate_constraints(result, param_name, param_value, param_schema)
            return result

        if auto_fix and param_schema.auto_convert:
            converted = self._try_convert_type(param_value, param_schema.type_)
            if converted is not None:
                result.validated_parameters[param_name] = converted
                result.add_fixed(param_name, param_value, converted)
                self._validate_constraints(result, param_name, converted, param_schema)
                return result

        if param_schema.required and param_value is None:
            result.add_error(
                ParameterTypeError(
                    param_name,
                    param_schema.type_.value,
                    "null",
                )
            )
            return result

        result.add_error(
            ParameterTypeError(
                param_name,
                param_schema.type_.value,
                actual_type.value,
            )
        )
        return result

    def _get_value_type(self, value: Any) -> ParameterType:
        if value is None:
            return ParameterType.NULL
        if isinstance(value, bool):
            return ParameterType.BOOLEAN
        if isinstance(value, int):
            return ParameterType.INTEGER
        if isinstance(value, float):
            return ParameterType.NUMBER
        if isinstance(value, str):
            return ParameterType.STRING
        if isinstance(value, list):
            return ParameterType.ARRAY
        if isinstance(value, dict):
            return ParameterType.OBJECT
        return ParameterType.NULL

    def _try_convert_type(self, value: Any, target_type: ParameterType) -> Optional[Any]:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()

        try:
            if target_type == ParameterType.STRING:
                return str(value)

            if target_type == ParameterType.INTEGER:
                if isinstance(value, str):
                    if value.lower() in ("true", "false"):
                        return 1 if value.lower() == "true" else 0
                    return int(float(value))
                if isinstance(value, bool):
                    return 1 if value else 0
                if isinstance(value, (int, float)):
                    return int(value)

            if target_type == ParameterType.NUMBER:
                if isinstance(value, str):
                    if value.lower() in ("true", "false"):
                        return 1.0 if value.lower() == "true" else 0.0
                    return float(value)
                if isinstance(value, bool):
                    return 1.0 if value else 0.0
                if isinstance(value, (int, float)):
                    return float(value)

            if target_type == ParameterType.BOOLEAN:
                if isinstance(value, str):
                    lower_val = value.lower()
                    if lower_val in ("true", "1", "yes", "on", "y"):
                        return True
                    if lower_val in ("false", "0", "no", "off", "n"):
                        return False
                if isinstance(value, (int, float)):
                    return value != 0
                if isinstance(value, bool):
                    return value

            if target_type == ParameterType.ARRAY:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                    return [value]

            if target_type == ParameterType.OBJECT:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass

        except (ValueError, TypeError):
            pass

        return None

    def _validate_constraints(
        self,
        result: ParameterValidationResult,
        param_name: str,
        value: Any,
        param_schema: ParameterSchema,
    ) -> None:
        if param_schema.enum and value not in param_schema.enum:
            result.add_error(
                ParameterValueError(
                    param_name,
                    value,
                    f"must be one of {param_schema.enum}",
                )
            )

        value_type = self._get_value_type(value)

        if value_type in (ParameterType.INTEGER, ParameterType.NUMBER):
            if param_schema.minimum is not None and value < param_schema.minimum:
                result.add_error(
                    ParameterValueError(
                        param_name,
                        value,
                        f"must be >= {param_schema.minimum}",
                    )
                )
            if param_schema.maximum is not None and value > param_schema.maximum:
                result.add_error(
                    ParameterValueError(
                        param_name,
                        value,
                        f"must be <= {param_schema.maximum}",
                    )
                )

        if value_type == ParameterType.STRING:
            if param_schema.min_length is not None and len(value) < param_schema.min_length:
                result.add_error(
                    ParameterValueError(
                        param_name,
                        value,
                        f"length must be >= {param_schema.min_length}",
                    )
                )
            if param_schema.max_length is not None and len(value) > param_schema.max_length:
                result.add_error(
                    ParameterValueError(
                        param_name,
                        value,
                        f"length must be <= {param_schema.max_length}",
                    )
                )
