# src/utils/config_loader.py

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config() -> dict:
    """
    Loads configuration from YAML and .env files.

    - Loads default config from config/config.yaml.
    - Loads environment variables from a .env file.
    - Environment variables can override YAML settings (e.g., API_KEY).

    Returns:
        A dictionary containing the combined configuration.
    """
    # Load .env file for secrets
    load_dotenv()

    # Path to the default configuration file
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Example of overriding config with env vars (optional)
    # This is useful for secrets or environment-specific settings
    if "API_KEY" in os.environ:
        # A good practice is to structure your config to hold these
        if "api" not in config:
            config["api"] = {}
        config["api"]["key"] = os.environ.get("API_KEY")

    if "SECRET_KEY" in os.environ:
        if "api" not in config:
            config["api"] = {}
        config["api"]["secret"] = os.environ.get("SECRET_KEY")

    return config


# Load once and export
config = load_config()
