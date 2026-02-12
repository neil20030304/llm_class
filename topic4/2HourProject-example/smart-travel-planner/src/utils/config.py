"""Configuration and environment variable management."""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_config() -> bool:
    """
    Load environment variables from .env file.
    
    Searches for .env in current directory and parent directories.
    
    Returns:
        True if .env was found and loaded, False otherwise.
    """
    # Try current directory first
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        return True
    
    # Try parent directories (useful when running from subdirectories)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
    
    return False


def get_api_key(key_name: str) -> str | None:
    """
    Get an API key from environment variables.
    
    Args:
        key_name: Name of the environment variable (e.g., 'OPENWEATHER_API_KEY')
    
    Returns:
        The API key value, or None if not set.
    """
    return os.getenv(key_name)


def validate_config() -> dict[str, bool]:
    """
    Validate that all required API keys are configured.
    
    Returns:
        Dictionary mapping key names to their availability status.
    """
    required_keys = ["OPENAI_API_KEY", "OPENWEATHER_API_KEY"]
    return {key: bool(get_api_key(key)) for key in required_keys}
