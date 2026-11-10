"""Project-local path helpers.

All defaults are resolved relative to the repository, so entry points work from
any current working directory and do not depend on one machine's absolute path.
"""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: PathLike) -> Path:
    """Return an absolute path below the repository root."""
    return PROJECT_ROOT.joinpath(*(Path(part) for part in parts))


def resolve_project_path(path: PathLike) -> Path:
    """Resolve a CLI path, interpreting relative values from the project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()
