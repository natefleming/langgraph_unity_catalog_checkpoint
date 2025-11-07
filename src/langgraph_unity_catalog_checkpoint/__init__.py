"""Checkpoint Unity Catalog - LangGraph persistence for Databricks Unity Catalog.

This package provides checkpoint and store implementations for LangGraph using
Databricks Unity Catalog as the backend storage, following the pattern from
langgraph.checkpoint.postgres.

Checkpoint Savers:
- UnityCatalogCheckpointSaver: Synchronous checkpoint saver
- AsyncUnityCatalogCheckpointSaver: Asynchronous checkpoint saver
- ShallowUnityCatalogSaver: Lightweight sync saver (most recent checkpoint only)
- AsyncShallowUnityCatalogSaver: Lightweight async saver (most recent checkpoint only)

Store:
- UnityCatalogStore: Key-value store implementation for Unity Catalog
"""

from langgraph_unity_catalog_checkpoint.checkpoint import (
    AsyncShallowUnityCatalogSaver,
    AsyncUnityCatalogCheckpointSaver,
    BaseUnityCatalogSaver,
    ShallowUnityCatalogSaver,
    UnityCatalogCheckpointSaver,
)
from langgraph_unity_catalog_checkpoint.store import UnityCatalogStore

__version__ = "0.1.0"

__all__ = [
    # Checkpoint Savers
    "UnityCatalogCheckpointSaver",
    "AsyncUnityCatalogCheckpointSaver",
    "ShallowUnityCatalogSaver",
    "AsyncShallowUnityCatalogSaver",
    "BaseUnityCatalogSaver",
    # Store
    "UnityCatalogStore",
    # Version
    "__version__",
]
