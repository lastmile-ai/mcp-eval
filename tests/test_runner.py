from pathlib import Path


from mcp_eval.runner import (
    discover_tests_and_datasets,
    expand_parametrized_tests,
    group_tests_by_file,
)


def _write_tmp_module(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "mod_test.py"
    p.write_text(content, encoding="utf-8")
    return p


def test_discover_tests_and_datasets(tmp_path, monkeypatch):
    content = """
from mcp_eval.core import task, parametrize
from mcp_eval.datasets import Dataset, Case

@task("desc")
async def t1(agent, session):
    pass

@task("desc")
@parametrize("x", [1,2])
async def t2(agent, session, x):
    pass

ds = Dataset(name="D", cases=[Case(name="c", inputs="i")])
        """
    mod_path = _write_tmp_module(tmp_path, content)

    found = discover_tests_and_datasets(str(mod_path))
    assert len(found["tasks"]) == 2
    assert len(found["datasets"]) == 1

    expanded = expand_parametrized_tests(found["tasks"])
    # t1 (1) + t2 (2) = 3
    assert len(expanded) == 3

    grouped = group_tests_by_file(expanded)
    # single module key present
    assert list(grouped.keys()) == [mod_path]
