"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipeline import register_pipelines as _register


def register_pipelines() -> dict[str, Pipeline]:
    """Register named pipelines only (no __default__)."""
    return _register()
