"""PathEfficiency evaluator for checking optimal task completion paths."""

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

from mcp_eval.evaluators.base import SyncEvaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult


@dataclass
class PathEfficiency(SyncEvaluator):
    """Evaluates if agent took the optimal path to complete task."""

    optimal_steps: int | None = None
    """Expected optimal number of tool calls (auto-calculated if None)."""

    expected_tool_sequence: List[str] | None = None
    """Expected sequence of tool calls."""

    allow_extra_steps: int = 0
    """Tolerance for additional tool calls beyond optimal."""

    penalize_backtracking: bool = True
    """Whether to penalize returning to previous tools."""

    penalize_repeated_tools: bool = True
    """Whether to penalize excessive tool repetition."""

    tool_usage_limits: Dict[str, int] | None = None
    """Custom limits per tool (e.g., {"read": 2, "write": 1})."""

    default_tool_limit: int = 1
    """Default limit for tools not in tool_usage_limits."""

    golden_path: List[str] | None = None
    """Golden path support (single path or named key in config)."""

    requires_final_metrics: bool = True

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        actual_steps = len(ctx.metrics.tool_calls)
        tool_sequence = [call.name for call in ctx.tool_calls]

        # Auto-calculate optimal if not provided
        if self.optimal_steps is None:
            # Heuristic: unique tools used
            self.optimal_steps = len(set(tool_sequence))

        # Calculate efficiency score
        efficiency_score = self.optimal_steps / actual_steps if actual_steps > 0 else 0

        # Check for inefficiencies
        inefficiencies = []

        # Golden path comparison
        if self.golden_path:
            gp_ok, gp_issues, gp_eff = self._compare_to_golden(
                tool_sequence, self.golden_path
            )
            if not gp_ok:
                inefficiencies.extend(gp_issues)
            # Blend efficiency score with golden-path efficiency (average)
            efficiency_score = (
                (efficiency_score + gp_eff) / 2 if efficiency_score > 0 else gp_eff
            )

        # Check sequence
        sequence_correct = True
        if self.expected_tool_sequence:
            sequence_correct, seq_issues = self._check_sequence(
                tool_sequence, self.expected_tool_sequence
            )
            inefficiencies.extend(seq_issues)

        # Check for backtracking
        if self.penalize_backtracking:
            backtrack_count = self._count_backtracking(tool_sequence)
            if backtrack_count > 0:
                inefficiencies.append(f"Backtracking detected: {backtrack_count} times")
                efficiency_score *= (
                    1 - 0.1 * backtrack_count
                )  # 10% penalty per backtrack

        # Check for repeated tools
        if self.penalize_repeated_tools:
            repetitions = self._count_repetitions(tool_sequence)
            if repetitions:
                inefficiencies.append(f"Repeated tools: {repetitions}")
                efficiency_score *= 0.9  # 10% penalty for repetitions

        # Check individual failure conditions
        steps_exceeded = actual_steps > self.optimal_steps + self.allow_extra_steps
        has_inefficiencies = len(inefficiencies) > 0

        # Overall pass/fail
        passed = not steps_exceeded and sequence_correct and not has_inefficiencies

        # Build comprehensive expected description
        expected = "Optimal path"
        actual = f"{actual_steps} steps"

        if not sequence_correct:
            expected = f"sequence: {self.expected_tool_sequence}"
            actual = f"sequence: {tool_sequence}"
        elif has_inefficiencies:
            actual = f"inefficiencies: {inefficiencies}"
            expected = "No inefficiencies"
        elif steps_exceeded:
            expected = f"≤{self.optimal_steps + self.allow_extra_steps} steps"
            actual = f"{actual_steps} steps"

        return EvaluatorResult(
            passed=passed,
            expected=expected,
            actual=actual,
            score=efficiency_score,
            details={
                "tool_sequence": tool_sequence,
                "efficiency_score": efficiency_score,
                "sequence_correct": sequence_correct,
                "steps_exceeded": steps_exceeded,
                "has_inefficiencies": has_inefficiencies,
                "inefficiencies": inefficiencies,
                "optimal_path": self.expected_tool_sequence or self.golden_path,
                "actual_steps": actual_steps,
                "max_allowed_steps": self.optimal_steps + self.allow_extra_steps,
            },
        )

    def _check_sequence(
        self, actual: List[str], expected: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if actual sequence matches expected."""
        issues = []

        # Check if expected tools appear in order
        expected_idx = 0
        for tool in actual:
            if expected_idx < len(expected) and tool == expected[expected_idx]:
                expected_idx += 1

        if expected_idx != len(expected):
            missing = expected[expected_idx:]
            issues.append(f"Missing expected tools: {missing}")
            return False, issues

        # Check for unexpected tools between expected ones
        actual_filtered = [t for t in actual if t in expected]
        if actual_filtered != expected:
            issues.append("Expected tools not in correct order")
            return False, issues

        return True, issues

    def _count_backtracking(self, sequence: List[str]) -> int:
        """Count times agent went back to previous tools."""
        if len(sequence) < 2:
            return 0

        backtrack_count = 0
        seen_tools = set()
        tool_last_index = {}

        for i, tool in enumerate(sequence):
            if tool in seen_tools:
                # Check if we're going backwards
                if i - tool_last_index[tool] > 2:  # Allow immediate retry
                    backtrack_count += 1
            seen_tools.add(tool)
            tool_last_index[tool] = i

        return backtrack_count

    def _count_repetitions(self, sequence: List[str]) -> Dict[str, int]:
        """Count unnecessary repetitions of tools."""
        tool_counts = defaultdict(int)
        repetitions = {}

        for tool in sequence:
            tool_counts[tool] += 1

        # Use configurable limits
        tool_limits = self.tool_usage_limits or {}

        for tool, count in tool_counts.items():
            expected = tool_limits.get(tool, self.default_tool_limit)
            if count > expected:
                repetitions[tool] = count - expected

        return repetitions

    def _compare_to_golden(
        self, actual: List[str], golden: List[str]
    ) -> Tuple[bool, List[str], float]:
        """Compare actual path to golden path and compute an efficiency score.

        Returns:
            (matches, issues, efficiency_score)
        """
        issues: List[str] = []
        # Basic subsequence check (golden should appear in order)
        g_idx = 0
        for tool in actual:
            if g_idx < len(golden) and tool == golden[g_idx]:
                g_idx += 1
        if g_idx != len(golden):
            missing = golden[g_idx:]
            issues.append(f"Missing golden steps: {missing}")

        # Efficiency: golden_length / actual_length capped at 1.0
        eff = min(1.0, (len(golden) / len(actual)) if actual else 0.0)
        # Penalty for unexpected tools not in golden
        unexpected = [t for t in actual if t not in golden]
        if unexpected:
            issues.append(f"Unexpected tools: {unexpected}")
            eff *= 0.9

        return (len(issues) == 0, issues, eff)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_steps": self.optimal_steps,
            "expected_tool_sequence": self.expected_tool_sequence,
            "allow_extra_steps": self.allow_extra_steps,
            "penalize_backtracking": self.penalize_backtracking,
            "penalize_repeated_tools": self.penalize_repeated_tools,
            "tool_usage_limits": self.tool_usage_limits,
            "default_tool_limit": self.default_tool_limit,
            "golden_path": self.golden_path,
        }
