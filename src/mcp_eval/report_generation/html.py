"""HTML report generation for MCP-Eval."""

from typing import Dict, Any
import platform
import sys
import os
from datetime import datetime


def format_config_for_html(config_info: Dict[str, Any]) -> str:
    """Format configuration for HTML display."""
    if not config_info:
        return "No configuration available"

    import yaml

    # Remove the _config_path key for display
    display_config = {k: v for k, v in config_info.items() if k != "_config_path"}
    return yaml.dump(display_config, default_flow_style=False, sort_keys=False)


def generate_combined_html_report(
    report_data: Dict[str, Any], output_path: str
) -> None:
    """Generate a combined HTML report."""
    summary = report_data["summary"]

    total_tests = summary["total_decorator_tests"] + summary["total_dataset_cases"]
    total_passed = summary["passed_decorator_tests"] + summary["passed_dataset_cases"]
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    # Collect environment information
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.getcwd(),
    }

    # Try to load MCP-Eval configuration
    config_info = None
    config_paths = ["mcpeval.yaml", "mcpeval.yml"]
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                import yaml

                with open(config_path, "r") as f:
                    config_info = yaml.safe_load(f)
                    config_info["_config_path"] = config_path
                break
            except Exception:
                continue

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-Eval Combined Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        .collapsible {{
            background-color: #f39c12;
            color: white;
            cursor: pointer;
            padding: 12px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
            border-radius: 4px;
        }}
        .collapsible:hover {{
            background-color: #e67e22;
        }}
        .collapsible.active {{
            background-color: #d35400;
        }}
        .content {{
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .content.show {{
            display: block;
            padding: 18px;
        }}
        .env-info, .config-info {{
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 14px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
        }}
        .summary-item {{
            margin: 10px 0;
            font-size: 16px;
        }}
        .success-rate {{
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }}
        .filter-controls {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #e8f4f8;
            border-radius: 4px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}
        .filter-btn.active {{
            background-color: #3498db;
            color: white;
        }}
        .filter-btn:not(.active) {{
            background-color: #bdc3c7;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .test-row.pass {{
            background-color: #f8fff8;
        }}
        .test-row.fail {{
            background-color: #fff8f8;
        }}
        .failure-message {{
            background-color: #f8f8f8;
            padding: 15px;
            border-left: 4px solid #e17055;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .failure-row {{
            border-top: none !important;
        }}
        .failure-row td {{
            padding-top: 0 !important;
            padding-bottom: 15px !important;
        }}
        .dataset-section {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        .dataset-stats {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }}
        .stat {{
            background-color: white;
            padding: 10px 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
    </style>
    <script>
        function toggleCollapsible(element) {{
            element.classList.toggle("active");
            var content = element.nextElementSibling;
            if (content.style.display === "block") {{
                content.style.display = "none";
            }} else {{
                content.style.display = "block";
            }}
        }}
        
        function filterTests(filter) {{
            const table = document.getElementById('test-table');
            const rows = table.getElementsByTagName('tr');
            const buttons = document.querySelectorAll('.filter-btn');
            
            // Update button states
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Filter rows
            for (let i = 1; i < rows.length; i++) {{ // Skip header row
                const row = rows[i];
                
                // Check if this is a failure message row
                if (row.classList.contains('failure-row')) {{
                    // For failure rows, check the previous row's status
                    const prevRow = rows[i - 1];
                    const statusCell = prevRow.cells[1];
                    const isPass = statusCell.textContent.includes('PASS');
                    
                    if (filter === 'all') {{
                        row.style.display = '';
                    }} else if (filter === 'pass' && isPass) {{
                        row.style.display = '';
                    }} else if (filter === 'fail' && !isPass) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }} else {{
                    // For test rows, check their own status
                    const statusCell = row.cells[1];
                    const isPass = statusCell.textContent.includes('PASS');
                    
                    if (filter === 'all') {{
                        row.style.display = '';
                    }} else if (filter === 'pass' && isPass) {{
                        row.style.display = '';
                    }} else if (filter === 'fail' && !isPass) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(collapsible) {{
                collapsible.addEventListener('click', function() {{
                    toggleCollapsible(this);
                }});
            }});
        }});
    </script>
</head>
<body>
    <div class="container">
        <h1>MCP-Eval Combined Test Report</h1>
        
        <button class="collapsible">Environment Information</button>
        <div class="content">
            <div class="env-info">
                <strong>Python Version:</strong> {env_info["python_version"]}<br>
                <strong>Platform:</strong> {env_info["platform"]}<br>
                <strong>System:</strong> {env_info["system"]}<br>
                <strong>Machine:</strong> {env_info["machine"]}<br>
                <strong>Processor:</strong> {env_info["processor"]}<br>
                <strong>Timestamp:</strong> {env_info["timestamp"]}<br>
                <strong>Working Directory:</strong> {env_info["working_directory"]}
            </div>
        </div>
        
        <button class="collapsible">MCP-Eval Configuration</button>
        <div class="content">
            <div class="config-info">
                {"<strong>Configuration loaded from:</strong> " + config_info.get("_config_path", "N/A") + "<br><br>" if config_info else ""}
                <pre>{"".join(format_config_for_html(config_info)) if config_info else "No configuration file found"}</pre>
            </div>
        </div>
        
        <div class="summary">
            <div class="summary-item">
                <strong>Decorator Tests:</strong> {summary["passed_decorator_tests"]}/{summary["total_decorator_tests"]} passed
            </div>
            <div class="summary-item">
                <strong>Dataset Cases:</strong> {summary["passed_dataset_cases"]}/{summary["total_dataset_cases"]} passed
            </div>
            <div class="summary-item success-rate">
                <strong>Overall Success Rate:</strong> {overall_success_rate:.1f}%
            </div>
        </div>
        
        <h2>Decorator Test Results</h2>
        <div class="filter-controls">
            <button class="filter-btn active" onclick="filterTests('all')">All Tests</button>
            <button class="filter-btn" onclick="filterTests('pass')">Passed Only</button>
            <button class="filter-btn" onclick="filterTests('fail')">Failed Only</button>
        </div>
        <table id="test-table">
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Server</th>
            </tr>
"""

    for test_data in report_data["decorator_tests"]:
        status_class = "pass" if test_data["passed"] else "fail"
        status_text = "✅ PASS" if test_data["passed"] else "❌ FAIL"
        duration = f"{test_data.get('duration_ms', 0):.1f}ms"
        server = test_data.get("server_name", "unknown")

        # Add main test row
        html += f"""            <tr class="test-row {status_class}">
                <td>{test_data["test_name"]}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{duration}</td>
                <td>{server}</td>
            </tr>
"""

        # Add failure message row if test failed
        if not test_data["passed"]:
            failure_msg = test_data.get("error", "")
            if failure_msg:
                # Escape HTML in failure message for proper display in code block
                import html as html_module

                escaped_failure_msg = html_module.escape(failure_msg)
                html += f"""            <tr class="test-row {status_class} failure-row">
                <td colspan="4">
                    <div class="failure-message"><code>{escaped_failure_msg}</code></div>
                </td>
            </tr>
"""

    html += """        </table>
        
        <h2>Dataset Evaluation Results</h2>
"""

    for dataset_data in report_data["dataset_reports"]:
        dataset_summary = dataset_data["summary"]
        html += f"""        <div class="dataset-section">
            <h3>{dataset_data["dataset_name"]}</h3>
            <div class="dataset-stats">
                <div class="stat">
                    <div class="stat-value">{dataset_summary["passed_cases"]}/{dataset_summary["total_cases"]}</div>
                    <div class="stat-label">Cases Passed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{dataset_summary["success_rate"] * 100:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{dataset_summary["average_duration_ms"]:.1f}ms</div>
                    <div class="stat-label">Avg Duration</div>
                </div>
            </div>
        </div>
"""

    html += """    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
