from importlib.resources import files


def get_assets_dir():
    """Get the path to the assets directory."""
    return files("spotlight_tools") / "assets"
