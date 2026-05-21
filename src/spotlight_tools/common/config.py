import yaml
from importlib.resources import files

import spotlight_tools
from pathlib import Path


def load_spotlight_tools_config() -> dict:
    assets_dir = files("spotlight_tools", "assets")
    config_path = assets_dir / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file {config_path} does not exist. Make sure the "
            "spotlight-tools package is installed correctly."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

