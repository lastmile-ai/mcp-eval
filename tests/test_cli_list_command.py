import pytest
from typer.testing import CliRunner

from mcp_eval.cli import app as cli_app


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

    res = runner.invoke(cli_app, ["server", "list"])
    assert res.exit_code == 0
    assert "Configured MCP Servers" in res.stdout

    res2 = runner.invoke(cli_app, ["server", "list", "--verbose"])
    assert res2.exit_code == 0
    assert "demo" in res2.stdout
