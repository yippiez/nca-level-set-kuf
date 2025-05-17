"""Path utilities for consistent directory management."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_reports_dir(notebook_name: str) -> Path:
    """Get the reports directory for a specific notebook.
    
    Args:
        notebook_name: Name of the notebook (without .py extension)
        
    Returns:
        Path to the notebook's reports directory
    """
    reports_dir = get_project_root() / "reports" / notebook_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_cache_dir() -> Path:
    """Get the cache directory."""
    cache_dir = get_project_root() / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir
