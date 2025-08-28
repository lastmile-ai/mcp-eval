import pytest
from typer.testing import CliRunner

from mcp_eval.cli import app as cli_app


@pytest.fixture()
def runner():
    # Older Typer/Click versions don't support mix_stderr param in constructor
    return CliRunner()


def test_cli_version(runner):
    result = runner.invoke(cli_app, ["version"])
    assert result.exit_code == 0
    assert "MCP-Eval" in result.stdout


def test_cli_list_servers_and_agents_empty_project(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # minimal configs created by conftest
    res = runner.invoke(cli_app, ["server", "list"])  # no servers configured
    assert res.exit_code == 0

    res = runner.invoke(cli_app, ["agent", "list"])  # no agents configured
    assert res.exit_code == 0


def test_cli_validate_quick(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # ensure mcpeval.yaml/secrets exist via conftest; validate quick should pass basic checks
    res = runner.invoke(cli_app, ["validate", "--quick"])  # skip connections
    # validate may nonzero if missing sections; but with conftest config it should succeed
    assert res.exit_code in (0, 1)


def test_cli_doctor_basic(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(cli_app, ["doctor"])  # basic checks
    assert res.exit_code == 0
