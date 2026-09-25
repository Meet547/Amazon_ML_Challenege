from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    """
    Ensure the parent directory for a file exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
