from __future__ import annotations

import pytest

from toolcall_resilience.errors import (
    MissingParameterError,
    ParameterTypeError,
    ParameterValueError,
)
from toolcall_resilience.models import ParameterSchema, ParameterType, ToolSchema
from toolcall_resilience.validation import ParameterValidator


class TestParameterValidation:
    def setup_method(self) -> None:
        self.validator = ParameterValidator()

    def create_schema(self, parameters: list[ParameterSchema]) -> ToolSchema:
        return ToolSchema(
            name="test_tool",
            description="Test tool",
            parameters=parameters,
        )

    def test_valid_parameters(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="a", type_=ParameterType.INTEGER, required=True),
            ParameterSchema(name="b", type_=ParameterType.STRING, required=False, default="test"),
        ])

        result = self.validator.validate({"a": 10, "b": "hello"}, schema)

        assert result.is_valid is True
        assert result.validated_parameters == {"a": 10, "b": "hello"}
        assert result.errors == []

    def test_missing_required_parameter(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="required_param", type_=ParameterType.STRING, required=True),
        ])

        result = self.validator.validate({}, schema)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], MissingParameterError)
        assert result.errors[0].parameter_name == "required_param"

    def test_default_value_for_missing_optional(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="optional", type_=ParameterType.STRING, required=False, default="default_val"),
        ])

        result = self.validator.validate({}, schema)

        assert result.is_valid is True
        assert result.validated_parameters["optional"] == "default_val"
        assert len(result.warnings) == 1

    def test_parameter_type_mismatch(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="number", type_=ParameterType.INTEGER, required=True),
        ])

        result = self.validator.validate({"number": "not_a_number"}, schema, auto_fix=False)

        assert result.is_valid is False
        assert isinstance(result.errors[0], ParameterTypeError)
        assert result.errors[0].parameter_name == "number"

    def test_auto_convert_string_to_int(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.INTEGER, required=True, auto_convert=True),
        ])

        result = self.validator.validate({"value": "42"}, schema, auto_fix=True)

        assert result.is_valid is True
        assert result.validated_parameters["value"] == 42
        assert result.fixed_count == 1

    def test_auto_convert_string_to_float(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.NUMBER, required=True, auto_convert=True),
        ])

        result = self.validator.validate({"value": "3.14"}, schema, auto_fix=True)

        assert result.is_valid is True
        assert result.validated_parameters["value"] == 3.14
        assert result.fixed_count == 1

    def test_auto_convert_string_to_bool_true(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="flag", type_=ParameterType.BOOLEAN, required=True, auto_convert=True),
        ])

        for true_value in ["true", "True", "TRUE", "1", "yes", "on"]:
            result = self.validator.validate({"flag": true_value}, schema, auto_fix=True)
            assert result.is_valid is True, f"Failed for value: {true_value}"
            assert result.validated_parameters["flag"] is True

    def test_auto_convert_string_to_bool_false(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="flag", type_=ParameterType.BOOLEAN, required=True, auto_convert=True),
        ])

        for false_value in ["false", "False", "FALSE", "0", "no", "off"]:
            result = self.validator.validate({"flag": false_value}, schema, auto_fix=True)
            assert result.is_valid is True, f"Failed for value: {false_value}"
            assert result.validated_parameters["flag"] is False

    def test_minimum_constraint_violation(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.INTEGER, required=True, minimum=10),
        ])

        result = self.validator.validate({"value": 5}, schema)

        assert result.is_valid is False
        assert isinstance(result.errors[0], ParameterValueError)

    def test_minimum_constraint_satisfied(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.INTEGER, required=True, minimum=10),
        ])

        result = self.validator.validate({"value": 15}, schema)

        assert result.is_valid is True
        assert result.validated_parameters["value"] == 15

    def test_maximum_constraint_violation(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.NUMBER, required=True, maximum=100.0),
        ])

        result = self.validator.validate({"value": 150.5}, schema)

        assert result.is_valid is False
        assert isinstance(result.errors[0], ParameterValueError)

    def test_enum_constraint(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="color", type_=ParameterType.STRING, required=True, enum=["red", "green", "blue"]),
        ])

        valid_result = self.validator.validate({"color": "red"}, schema)
        assert valid_result.is_valid is True

        invalid_result = self.validator.validate({"color": "yellow"}, schema)
        assert invalid_result.is_valid is False
        assert isinstance(invalid_result.errors[0], ParameterValueError)

    def test_strict_mode_rejects_extra_params(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="a", type_=ParameterType.INTEGER, required=True),
        ])
        schema.strict = True

        result = self.validator.validate({"a": 1, "b": 2}, schema)

        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_non_strict_mode_ignores_extra_params(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="a", type_=ParameterType.INTEGER, required=True),
        ])
        schema.strict = False

        result = self.validator.validate({"a": 1, "extra": "ignored"}, schema)

        assert result.is_valid is True
        assert "a" in result.validated_parameters
        assert "extra" not in result.validated_parameters
        assert len(result.warnings) == 1

    def test_multiple_type_conversions(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="count", type_=ParameterType.INTEGER, required=True),
            ParameterSchema(name="price", type_=ParameterType.NUMBER, required=True),
            ParameterSchema(name="active", type_=ParameterType.BOOLEAN, required=True),
        ])

        result = self.validator.validate(
            {"count": "10", "price": "99.99", "active": "true"},
            schema,
            auto_fix=True,
        )

        assert result.is_valid is True
        assert result.validated_parameters["count"] == 10
        assert result.validated_parameters["price"] == 99.99
        assert result.validated_parameters["active"] is True
        assert result.fixed_count == 3

    def test_int_to_bool_conversion(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="flag", type_=ParameterType.BOOLEAN, required=True, auto_convert=True),
        ])

        true_result = self.validator.validate({"flag": 1}, schema, auto_fix=True)
        assert true_result.is_valid is True
        assert true_result.validated_parameters["flag"] is True

        false_result = self.validator.validate({"flag": 0}, schema, auto_fix=True)
        assert false_result.is_valid is True
        assert false_result.validated_parameters["flag"] is False

    def test_float_to_int_conversion(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="value", type_=ParameterType.INTEGER, required=True, auto_convert=True),
        ])

        result = self.validator.validate({"value": 42.0}, schema, auto_fix=True)

        assert result.is_valid is True
        assert result.validated_parameters["value"] == 42
        assert isinstance(result.validated_parameters["value"], int)

    def test_none_for_required_parameter(self) -> None:
        schema = self.create_schema([
            ParameterSchema(name="required", type_=ParameterType.STRING, required=True),
        ])

        result = self.validator.validate({"required": None}, schema)

        assert result.is_valid is False
        assert isinstance(result.errors[0], ParameterTypeError)
