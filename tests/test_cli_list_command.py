import pytest
from typer.testing import CliRunner

from mcp_eval.cli.list_command import app as list_app


@pytest.fixture()
def runner():
    return CliRunner()


def test_list_servers_table_and_verbose(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcpeval.yaml").write_text(
        """
mcp:
  servers:
    demo:
      transport: stdio
      command: uvx
      args: [mcp-server-fetch]
        """.strip()
    )

    res = runner.invoke(list_app, ["servers"])
    assert res.exit_code == 0
    assert "Configured MCP Servers" in res.stdout

    res2 = runner.invoke(list_app, ["servers", "--verbose"])
    assert res2.exit_code == 0
    assert "demo" in res2.stdout
