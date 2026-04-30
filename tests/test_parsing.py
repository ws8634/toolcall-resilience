from __future__ import annotations

import json

import pytest

from toolcall_resilience.errors import JsonParseError
from toolcall_resilience.parsing import JsonParser, RequestParser, ResponseValidator
from toolcall_resilience.models import ParameterType, ToolRequest, ToolSchema


class TestJsonParser:
    def setup_method(self) -> None:
        self.parser = JsonParser(allow_repair=True)

    def test_parse_valid_json(self) -> None:
        result = self.parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_invalid_json_without_repair(self) -> None:
        parser = JsonParser(allow_repair=False)
        with pytest.raises(JsonParseError):
            parser.parse('{key: "value"}', try_repair=False)

    def test_repair_single_quotes(self) -> None:
        result = self.parser.parse("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_repair_missing_quotes(self) -> None:
        result = self.parser.parse("{key: 'value'}")
        assert result == {"key": "value"}

    def test_repair_trailing_comma(self) -> None:
        result = self.parser.parse('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_repair_array_trailing_comma(self) -> None:
        result = self.parser.parse("[1, 2, 3,]")
        assert result == [1, 2, 3]

    def test_extract_json_from_markdown(self) -> None:
        markdown_input = """
Here is some text.

```json
{"name": "test", "value": 42}
```

More text.
"""
        result = self.parser.parse(markdown_input)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_without_markdown_tags(self) -> None:
        mixed_input = "Some text before {\"key\": \"value\"} and some text after"
        result = self.parser.parse(mixed_input)
        assert result == {"key": "value"}

    def test_fix_unclosed_braces(self) -> None:
        result = self.parser.parse('{"name": "test"')
        assert result == {"name": "test"}

    def test_fix_unclosed_brackets(self) -> None:
        result = self.parser.parse('[1, 2, 3')
        assert result == [1, 2, 3]

    def test_completely_invalid_input(self) -> None:
        with pytest.raises(JsonParseError):
            self.parser.parse("this is not json at all")


class TestRequestParser:
    def setup_method(self) -> None:
        self.parser = RequestParser()

    def test_parse_valid_dict(self) -> None:
        request_dict = {"tool_name": "my_tool", "parameters": {"a": 1}}
        result = self.parser.parse(request_dict)

        assert isinstance(result, ToolRequest)
        assert result.tool_name == "my_tool"
        assert result.parameters == {"a": 1}

    def test_parse_valid_json_string(self) -> None:
        json_str = '{"tool_name": "my_tool", "parameters": {"a": 1}}'
        result = self.parser.parse(json_str)

        assert isinstance(result, ToolRequest)
        assert result.tool_name == "my_tool"

    def test_parse_missing_tool_name(self) -> None:
        from toolcall_resilience.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            self.parser.parse({"parameters": {"a": 1}})

        assert "tool_name" in exc_info.value.missing_fields

    def test_parse_invalid_json(self) -> None:
        from toolcall_resilience.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            self.parser.parse({"tool_name": 123, "parameters": "not a dict"})

    def test_parse_repaired_json(self) -> None:
        result = self.parser.parse("{'tool_name': 'test_tool', 'parameters': {'x': 1}}")

        assert isinstance(result, ToolRequest)
        assert result.tool_name == "test_tool"
        assert result.parameters == {"x": 1}

    def test_parse_with_request_id(self) -> None:
        request_dict = {"tool_name": "my_tool", "id": "req-123", "parameters": {}}
        result = self.parser.parse(request_dict)

        assert result.id_ == "req-123"


class TestResponseValidator:
    def setup_method(self) -> None:
        self.validator = ResponseValidator()

    def test_validate_object_response(self) -> None:
        response = {"success": True, "data": "test"}
        result = self.validator.validate(response, expected_type=ParameterType.OBJECT)

        assert result == response

    def test_validate_string_response(self) -> None:
        response = "hello world"
        result = self.validator.validate(response, expected_type=ParameterType.STRING)

        assert result == response

    def test_validate_array_response(self) -> None:
        response = [1, 2, 3]
        result = self.validator.validate(response, expected_type=ParameterType.ARRAY)

        assert result == response

    def test_validate_integer_response(self) -> None:
        response = 42
        result = self.validator.validate(response, expected_type=ParameterType.INTEGER)

        assert result == 42

    def test_validate_number_response(self) -> None:
        response = 3.14
        result = self.validator.validate(response, expected_type=ParameterType.NUMBER)

        assert result == 3.14

    def test_validate_boolean_response(self) -> None:
        response = True
        result = self.validator.validate(response, expected_type=ParameterType.BOOLEAN)

        assert result is True

    def test_type_mismatch_raises_error(self) -> None:
        from toolcall_resilience.errors import ResponseSchemaError

        response = "a string"
        with pytest.raises(ResponseSchemaError) as exc_info:
            self.validator.validate(response, expected_type=ParameterType.INTEGER)

        assert "Expected integer" in exc_info.value.validation_errors[0] or "Expected integer" in str(
            exc_info.value
        )

    def test_validate_with_tool_schema(self) -> None:
        schema = ToolSchema(
            name="test",
            description="test",
            return_type=ParameterType.OBJECT,
        )

        response = {"result": "ok"}
        result = self.validator.validate(response, expected_schema=schema)

        assert result == response

    def test_none_matches_null_type(self) -> None:
        result = self.validator.validate(None, expected_type=ParameterType.NULL)
        assert result is None

    def test_int_is_valid_number(self) -> None:
        result = self.validator.validate(42, expected_type=ParameterType.NUMBER)
        assert result == 42
