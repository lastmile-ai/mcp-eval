"""HTML report generation for MCP-Eval."""

import json
import base64
from pathlib import Path
from typing import Dict, Any


def generate_combined_html_report(
    report_data: Dict[str, Any], output_path: str
) -> None:
    """Generate a combined HTML report using the index.html template."""
    # Read the template file
    template_path = Path(__file__).parent / "report_template.html"
    with open(template_path, "r") as f:
        html_content = f.read()

    # Convert report_data to JSON and encode as base64
    # Use default=str to handle non-serializable objects like EvaluationRecord
    report_json = json.dumps(report_data, default=str)
    report_base64 = base64.b64encode(report_json.encode()).decode()

    # Append the script tag after the closing </html> tag
    script_tag = (
        f'\n<script>\n  window.mcpevalReportBase64 = "{report_base64}";\n</script>'
    )
    html_content = html_content + script_tag

    # Write the modified HTML to the output file
    with open(output_path, "w") as f:
        f.write(html_content)
