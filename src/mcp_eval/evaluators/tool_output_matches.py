"""ToolOutputMatches evaluator for validating tool output against expected patterns."""

import re
from typing import Any, Dict, List, Literal, Pattern
from dataclasses import dataclass

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class ToolOutputMatches(SyncEvaluator):
    """
    Evaluator that validates tool output against expected patterns.

    This evaluator allows you to validate the output of tool calls against expected
    values using various matching strategies. It supports nested field extraction,
    different comparison types, and flexible call targeting.

    Examples:
        ```
        # Exact match on full output
        ToolOutputMatches(tool_name="read_file", expected_output="Hello world")

        # Match substring in output
        ToolOutputMatches(tool_name="search", expected_output="found", match_type="contains")

        # Regex pattern matching
        ToolOutputMatches(tool_name="validate", expected_output=r"\\d+", match_type="regex")

        # Extract nested field and match
        ToolOutputMatches(
            tool_name="api_call",
            expected_output="success",
            field_path="result.status"
        )

        # Partial dictionary matching
        ToolOutputMatches(
            tool_name="get_config",
            expected_output={"debug": True},
            match_type="partial"
        )
        ```
    """

    tool_name: str
    """Name of the tool whose output should be validated."""

    expected_output: Dict[str, Any] | str | Pattern | int | float | List[Any]
    """Expected output value or pattern to match against."""

    field_path: str | None = None
    """Optional path to extract nested field from tool output.
    
    Supports dot notation for nested objects and bracket notation for arrays:
    - "content.text" - Extract text field from content object
    - "items[0].name" - Extract name from first item in items array
    - "result.data[2].value" - Complex nested extraction
    """

    match_type: Literal["exact", "contains", "regex", "partial"] = "exact"
    """Type of matching to perform:
    
    - "exact": Exact equality comparison (default)
    - "contains": Substring/item containment check
    - "regex": Regular expression pattern matching (string outputs only)
    - "partial": Partial matching for dicts/lists (all expected items must be present)
    """

    case_sensitive: bool = True
    """Whether string comparisons should be case sensitive.
    
    Applies to "contains" and "regex" match types when comparing strings.
    """

    call_index: int = -1
    requires_final_metrics: bool = True
    """Which tool call to validate when multiple calls exist:
    
    - -1: Last call (most recent, default)
    - 0: First call
    - 1: Second call
    - etc.
    """

    contains_fallback_for_exact: bool = True
    """If True and match_type is 'exact' for strings, treat containment as pass when
    actual is a longer string that includes expected. Helpful for large tool payloads.
    """

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        """Evaluate tool output against expected patterns."""
        tool_calls = [call for call in ctx.tool_calls if call.name == self.tool_name]

        if not tool_calls:
            return EvaluatorResult(
                passed=False,
                expected=f"Tool '{self.tool_name}' to be called",
                actual="Tool not called",
                error="No matching tool calls found",
            )

        # Get the specified call
        try:
            target_call = tool_calls[self.call_index]
        except IndexError:
            return EvaluatorResult(
                passed=False,
                expected=f"At least {abs(self.call_index) + 1} calls to '{self.tool_name}'",
                actual=f"{len(tool_calls)} calls",
                error="Not enough tool calls",
            )

        # Extract the value to validate
        try:
            actual_value = self._extract_field_value(target_call.result)
        except Exception as e:
            return EvaluatorResult(
                passed=False,
                expected=f"Valid field path: {self.field_path}",
                actual=f"Error extracting field: {str(e)}",
                error=f"Field extraction failed: {str(e)}",
            )

        # Try to normalize common MCP tool result shapes into plain text for string comparisons
        try:
            actual_value = self._auto_normalize_actual(actual_value)
        except Exception:
            # Best-effort normalization; ignore if it fails
            pass

        # Perform validation based on match type
        try:
            passed = self._validate_match(actual_value)
        except Exception as e:
            return EvaluatorResult(
                passed=False,
                expected=self.expected_output,
                actual=actual_value,
                error=f"Validation failed: {str(e)}",
            )

        return EvaluatorResult(
            passed=passed,
            expected=self.expected_output,
            actual=actual_value,
            details={
                "tool_name": self.tool_name,
                "call_index": self.call_index,
                "field_path": self.field_path,
                "match_type": self.match_type,
                "case_sensitive": self.case_sensitive,
            },
        )

    def _extract_field_value(self, result: Any) -> Any:
        """Extract value from result using field_path."""
        if self.field_path is None:
            return result

        current = result
        path_parts = self._parse_field_path(self.field_path)

        for part in path_parts:
            if isinstance(part, int):  # Array index
                if not isinstance(current, (list, tuple)):
                    raise ValueError(f"Cannot index non-list with [{part}]")
                if part >= len(current) or part < -len(current):
                    raise ValueError(
                        f"Index [{part}] out of range for list of length {len(current)}"
                    )
                current = current[part]
            else:  # Dictionary key
                if not isinstance(current, dict):
                    raise ValueError(f"Cannot access key '{part}' on non-dict")
                if part not in current:
                    raise ValueError(f"Key '{part}' not found in result")
                current = current[part]

        return current

    def _parse_field_path(self, path: str) -> List[str | int]:
        """Parse field path into components."""
        parts = []
        current = ""
        i = 0

        while i < len(path):
            char = path[i]
            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != "]":
                    j += 1
                if j >= len(path):
                    raise ValueError(f"Unclosed bracket in field path: {path}")
                index_str = path[i + 1 : j]
                try:
                    parts.append(int(index_str))
                except ValueError:
                    raise ValueError(f"Invalid array index: {index_str}")
                i = j  # Skip the closing bracket
            else:
                current += char
            i += 1

        if current:
            parts.append(current)

        return parts

    # ---------------- helpers: normalization -----------------

    def _auto_normalize_actual(self, value: Any) -> Any:
        """Best-effort normalization of common tool result shapes.

        - If the expected output is a string or regex and the actual value is a structured
          dict/list with content items, flatten to a single text string.
        - Otherwise return the value unchanged.
        """
        expected_is_string_like = isinstance(self.expected_output, (str, int, float))
        if not expected_is_string_like and not hasattr(self.expected_output, "pattern"):
            return value

        # Attempt to extract human-readable text
        flattened = self._extract_text(value)
        return flattened if flattened is not None else value

    def _extract_text(self, obj: Any) -> str | None:
        """Recursively extract text from common MCP content/result shapes.

        Recognized patterns:
        - { "content": [ {"type": "text", "text": "..."}, ... ] }
        - { "text": "..." }
        - ["...", {"text": "..."}, ...]
        - Arbitrary nested dict/list with string leaves -> join with newlines
        """
        try:
            # Direct string
            if isinstance(obj, str):
                return obj

            # List: join extracted text parts
            if isinstance(obj, list):
                parts: List[str] = []
                for item in obj:
                    t = self._extract_text(item)
                    if t:
                        parts.append(t)
                return "\n".join(parts) if parts else None

            # Dict: handle well-known keys first
            if isinstance(obj, dict):
                # Explicit text key
                if isinstance(obj.get("text"), str):
                    return obj.get("text")

                # Content list shape
                content = obj.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        # Common item shape: {type: "text", text: "..."}
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        else:
                            t = self._extract_text(item)
                            if t:
                                parts.append(t)
                    if parts:
                        return "\n".join(parts)
                elif isinstance(content, str):
                    return content

                # Some tools may return {"output": "..."} or nested payloads
                output = obj.get("output")
                if isinstance(output, (str, list, dict)):
                    t = self._extract_text(output)
                    if t:
                        return t

                # Fallback: join all string leaf values
                parts: List[str] = []
                for v in obj.values():
                    t = self._extract_text(v)
                    if t:
                        parts.append(t)
                return "\n".join(parts) if parts else None
        except Exception:
            return None

        return None

    def _validate_match(self, actual_value: Any) -> bool:
        """Validate actual value against expected output based on match type."""
        if self.match_type == "exact":
            if actual_value == self.expected_output:
                return True
            # Best-effort fallback for string comparisons on large payloads
            if self.contains_fallback_for_exact and isinstance(actual_value, str) and isinstance(self.expected_output, str):
                if self.case_sensitive:
                    return self.expected_output in actual_value
                else:
                    return self.expected_output.lower() in actual_value.lower()
            return False

        elif self.match_type == "contains":
            if isinstance(actual_value, str) and isinstance(self.expected_output, str):
                if self.case_sensitive:
                    return self.expected_output in actual_value
                else:
                    return self.expected_output.lower() in actual_value.lower()
            elif isinstance(actual_value, (list, tuple)):
                return self.expected_output in actual_value
            elif isinstance(actual_value, dict) and isinstance(
                self.expected_output, str
            ):
                # Search in dict values
                for value in actual_value.values():
                    if isinstance(value, str):
                        if self.case_sensitive:
                            if self.expected_output in value:
                                return True
                        else:
                            if self.expected_output.lower() in value.lower():
                                return True
                return False
            else:
                return False

        elif self.match_type == "regex":
            if not isinstance(actual_value, str):
                return False

            if isinstance(self.expected_output, Pattern):
                pattern = self.expected_output
            elif isinstance(self.expected_output, str):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                pattern = re.compile(self.expected_output, flags)
            else:
                return False

            return bool(pattern.search(actual_value))

        elif self.match_type == "partial":
            if isinstance(self.expected_output, dict) and isinstance(
                actual_value, dict
            ):
                # Check if all expected keys and values are present
                for key, expected_val in self.expected_output.items():
                    if key not in actual_value:
                        return False
                    if isinstance(expected_val, dict) and isinstance(
                        actual_value[key], dict
                    ):
                        # Recursive partial matching for nested dicts
                        nested_validator = ToolOutputMatches(
                            tool_name=self.tool_name,
                            expected_output=expected_val,
                            match_type="partial",
                            case_sensitive=self.case_sensitive,
                        )
                        if not nested_validator._validate_match(actual_value[key]):
                            return False
                    elif isinstance(expected_val, list) and isinstance(
                        actual_value[key], list
                    ):
                        # Recursive partial matching for lists
                        nested_validator = ToolOutputMatches(
                            tool_name=self.tool_name,
                            expected_output=expected_val,
                            match_type="partial",
                            case_sensitive=self.case_sensitive,
                        )
                        if not nested_validator._validate_match(actual_value[key]):
                            return False
                    else:
                        if actual_value[key] != expected_val:
                            return False
                return True
            elif isinstance(self.expected_output, list) and isinstance(
                actual_value, list
            ):
                # Check if all expected items are present
                for expected_item in self.expected_output:
                    found = False
                    for actual_item in actual_value:
                        # If both items are dicts, use partial matching
                        if isinstance(expected_item, dict) and isinstance(
                            actual_item, dict
                        ):
                            nested_validator = ToolOutputMatches(
                                tool_name=self.tool_name,
                                expected_output=expected_item,
                                match_type="partial",
                                case_sensitive=self.case_sensitive,
                            )
                            if nested_validator._validate_match(actual_item):
                                found = True
                                break
                        # Otherwise check for exact equality
                        elif expected_item == actual_item:
                            found = True
                            break
                    if not found:
                        return False
                return True
            else:
                return False

        else:
            raise ValueError(f"Unknown match_type: {self.match_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evaluator to dict."""
        result = {
            "tool_name": self.tool_name,
            "expected_output": self.expected_output,
            "field_path": self.field_path,
            "match_type": self.match_type,
            "case_sensitive": self.case_sensitive,
            "call_index": self.call_index,
            "contains_fallback_for_exact": self.contains_fallback_for_exact,
        }

        # Handle Pattern objects
        if isinstance(self.expected_output, Pattern):
            result["expected_output"] = self.expected_output.pattern

        return result
