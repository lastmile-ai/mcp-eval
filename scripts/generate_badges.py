#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "typer",
# ]
# ///
"""
Generate SVG badges for MCP-Eval CI runs.

Inputs:
  --report <path>   Combined JSON report produced by `mcp-eval run --json ...`
  --outdir <dir>    Output directory for badges (default: mcpeval-reports/badges)
  --label-tests     Label for tests badge (default: mcp-tests)
  --label-cov       Label for coverage badge (default: mcp-cov)

Outputs:
  Writes two SVG badge files: tests.svg and coverage.svg in the output directory.

Coverage heuristic:
  Computes unique tools used vs available across all servers from metrics.tool_coverage
  over all decorator tests and dataset cases present in the combined report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import typer


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_metrics_objects(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Decorator tests
    for result in report.get("decorator_tests", []) or []:
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            yield metrics

    # Dataset cases
    for ds in report.get("dataset_reports", []) or []:
        for case in ds.get("results", []) or []:
            metrics = case.get("metrics")
            if isinstance(metrics, dict):
                yield metrics


def compute_pass_fail(report: Dict[str, Any]) -> Tuple[int, int, float]:
    summary = report.get("summary", {})
    total = int(summary.get("total_decorator_tests", 0)) + int(
        summary.get("total_dataset_cases", 0)
    )
    passed = int(summary.get("passed_decorator_tests", 0)) + int(
        summary.get("passed_dataset_cases", 0)
    )
    rate = (passed / total * 100.0) if total > 0 else 0.0
    return passed, total, rate


def compute_tool_coverage(report: Dict[str, Any]) -> float:
    available: set[str] = set()
    used: set[str] = set()
    for metrics in _collect_metrics_objects(report):
        coverage = metrics.get("tool_coverage") or {}
        if isinstance(coverage, dict):
            for server_name, cov in coverage.items():
                if not isinstance(cov, dict):
                    continue
                for t in cov.get("available_tools", []) or []:
                    available.add(f"{server_name}:{t}")
                for t in cov.get("used_tools", []) or []:
                    used.add(f"{server_name}:{t}")
    if not available:
        return 0.0
    pct = len(used) / len(available) * 100.0
    return pct


def _color_for_percentage(pct: float) -> str:
    # Simple red→yellow→green scale
    if pct >= 90:
        return "#4ec629"  # green
    if pct >= 75:
        return "#97ca00"  # yellow-green
    if pct >= 50:
        return "#dfb317"  # yellow
    if pct >= 25:
        return "#fe7d37"  # orange
    return "#e05d44"  # red


def _measure_text(text: str) -> int:
    # Approximate width calculation for DejaVu Sans 11px; good enough for badges
    # Base char width ~7px, adjust slightly for digits/uppercase
    width = 0
    for ch in text:
        if ch in "0123456789":
            width += 7
        elif ch.isupper():
            width += 7
        else:
            width += 6
    # Padding
    return width + 10


def make_badge(label: str, value: str, color: str) -> str:
    left_w = _measure_text(label)
    right_w = _measure_text(value)
    total_w = left_w + right_w
    # Construct an SVG similar to shields style
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" aria-label="{label}: {value}">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1" />
    <stop offset="1" stop-opacity=".1" />
  </linearGradient>
  <mask id="a">
    <rect width="{total_w}" height="20" rx="3" fill="#fff" />
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h{left_w}v20H0z"/>
    <path fill="{color}" d="M{left_w} 0h{right_w}v20H{left_w}z"/>
    <path fill="url(#b)" d="M0 0h{total_w}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{left_w / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{left_w / 2:.0f}" y="14">{label}</text>
    <text x="{left_w + right_w / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{left_w + right_w / 2:.0f}" y="14">{value}</text>
  </g>
</svg>'''
    return svg


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cli(
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to combined JSON report produced by mcp-eval run --json",
    ),
    outdir: str = typer.Option(
        "mcpeval-reports/badges",
        "--outdir",
        help="Output directory for generated SVG badges",
    ),
    label_tests: str = typer.Option(
        "mcp-tests", "--label-tests", help="Label for tests badge"
    ),
    label_cov: str = typer.Option(
        "mcp-cov", "--label-cov", help="Label for coverage badge"
    ),
):
    report_path = Path(report)
    outdir_path = Path(outdir)

    try:
        report_obj = load_report(report_path)
    except Exception:
        outdir_path.mkdir(parents=True, exist_ok=True)
        write_text(outdir_path / "tests.svg", make_badge(label_tests, "0/0", "#9f9f9f"))
        write_text(outdir_path / "coverage.svg", make_badge(label_cov, "0%", "#9f9f9f"))
        return

    passed, total, rate = compute_pass_fail(report_obj)
    tests_value = f"{passed}/{total}" if total > 0 else "0/0"
    tests_color = _color_for_percentage(rate)

    cov_pct = compute_tool_coverage(report_obj)
    cov_value = f"{int(round(cov_pct))}%"
    cov_color = _color_for_percentage(cov_pct)

    write_text(
        outdir_path / "tests.svg", make_badge(label_tests, tests_value, tests_color)
    )
    write_text(
        outdir_path / "coverage.svg", make_badge(label_cov, cov_value, cov_color)
    )


if __name__ == "__main__":
    typer.run(cli)
