from pathlib import Path

from mcp_eval.cli.utils import find_config_files, ensure_mcpeval_yaml, load_yaml, save_yaml


def test_ensure_mcpeval_yaml_and_find_config_files(tmp_path):
    # ensure_mcpeval_yaml should create a minimal config if none exists
    cfg_path = ensure_mcpeval_yaml(tmp_path)
    assert cfg_path.exists()

    paths = find_config_files(tmp_path)
    assert paths.mcpeval_yaml.exists()
    assert paths.mcpeval_secrets.name == "mcpeval.secrets.yaml"

    # roundtrip YAML helpers
    data = load_yaml(cfg_path)
    data["judge"] = {"min_score": 0.5}
    save_yaml(cfg_path, data)
    data2 = load_yaml(cfg_path)
    assert data2["judge"]["min_score"] == 0.5


