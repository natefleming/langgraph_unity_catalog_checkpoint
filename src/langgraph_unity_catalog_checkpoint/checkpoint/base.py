"""Base Unity Catalog checkpoint saver implementation.

This module contains the base class with shared SQL queries and helper methods
for both sync and async Unity Catalog checkpoint savers, following the pattern
from langgraph.checkpoint.postgres.base.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from databricks.sdk import WorkspaceClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

MetadataInput = dict[str, Any] | None


class BaseUnityCatalogSaver(BaseCheckpointSaver[str]):
    """Base class for Unity Catalog checkpoint savers.

    This class contains shared SQL queries, table schemas, and helper methods
    used by both sync and async implementations.
    """

    # Table creation SQL templates
    CREATE_CHECKPOINTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        checkpoint_id STRING NOT NULL,
        parent_checkpoint_id STRING,
        type STRING,
        checkpoint STRING NOT NULL,
        metadata STRING NOT NULL DEFAULT '{{}}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported',
        'description' = 'LangGraph checkpoint storage in Unity Catalog (JSON strings)'
    )
    """

    CREATE_CHECKPOINT_BLOBS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        channel STRING NOT NULL,
        version STRING NOT NULL,
        type STRING NOT NULL,
        blob STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported',
        'description' = 'LangGraph checkpoint blobs storage in Unity Catalog (JSON strings)'
    )
    """

    CREATE_CHECKPOINT_WRITES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        checkpoint_id STRING NOT NULL,
        task_id STRING NOT NULL,
        task_path STRING NOT NULL DEFAULT '',
        idx INT NOT NULL,
        channel STRING NOT NULL,
        type STRING,
        blob STRING NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported',
        'description' = 'LangGraph checkpoint writes storage in Unity Catalog (JSON strings)'
    )
    """

    # Query SQL templates
    SELECT_CHECKPOINTS_SQL = """
    SELECT
        thread_id,
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        type,
        checkpoint,
        metadata
    FROM {table_name}
    """

    SELECT_CHECKPOINT_BLOBS_SQL = """
    SELECT
        channel,
        version,
        type,
        blob
    FROM {table_name}
    WHERE thread_id = ?
        AND checkpoint_ns = ?
        AND channel = ?
        AND version = ?
    """

    SELECT_CHECKPOINT_WRITES_SQL = """
    SELECT
        task_id,
        task_path,
        idx,
        channel,
        type,
        blob
    FROM {table_name}
    WHERE thread_id = ?
        AND checkpoint_ns = ?
        AND checkpoint_id = ?
    ORDER BY task_id, idx
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        checkpoints_table: str = "checkpoints",
        checkpoint_blobs_table: str = "checkpoint_blobs",
        writes_table: str = "checkpoint_writes",
        warehouse_id: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the base Unity Catalog checkpoint saver.

        Args:
            workspace_client: Databricks WorkspaceClient for executing SQL.
            catalog: Unity Catalog name.
            schema: Schema name within the catalog.
            checkpoints_table: Table name for checkpoints (default: "checkpoints").
            checkpoint_blobs_table: Table name for checkpoint blobs (default: "checkpoint_blobs").
            writes_table: Table name for checkpoint writes (default: "checkpoint_writes").
            warehouse_id: SQL warehouse ID for executing statements.
            serde: Serializer for checkpoint data.
        """
        super().__init__(serde=serde)
        self.workspace_client = workspace_client
        self.catalog = catalog
        self.schema = schema
        self.warehouse_id = warehouse_id

        # Full table names
        self.full_checkpoints_table = f"{catalog}.{schema}.{checkpoints_table}"
        self.full_checkpoint_blobs_table = f"{catalog}.{schema}.{checkpoint_blobs_table}"
        self.full_writes_table = f"{catalog}.{schema}.{writes_table}"

    def _escape_string(self, s: str) -> str:
        """Escape a string for use in SQL."""
        return s.replace("'", "''")

    def _serialize_to_json_string(self, data: Any) -> str:
        """Serialize data to JSON string for STRING column."""
        import json
        import base64
        
        # If data is bytes, base64 encode it first
        if isinstance(data, bytes):
            return base64.b64encode(data).decode('utf-8')
        # Otherwise serialize as JSON
        return json.dumps(data)

    def _load_blobs(self, blob_values: list[tuple[str, str, bytes]] | None) -> dict[str, Any]:
        """Load blobs from database rows.

        Args:
            blob_values: List of (channel, type, blob) tuples.

        Returns:
            Dictionary mapping channel names to deserialized values.
        """
        if not blob_values:
            return {}
        return {
            channel: self.serde.loads_typed((type_, blob))
            for channel, type_, blob in blob_values
            if type_ != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        """Dump blobs for storage in database.

        Args:
            thread_id: Thread identifier.
            checkpoint_ns: Checkpoint namespace.
            values: Channel values to store.
            versions: Channel versions.

        Returns:
            List of tuples ready for database insertion.
        """
        if not versions:
            return []

        result = []
        for k, ver in versions.items():
            if k in values:
                type_, blob = self.serde.dumps_typed(values[k])
                result.append((thread_id, checkpoint_ns, k, str(ver), type_, blob))
            else:
                result.append((thread_id, checkpoint_ns, k, str(ver), "empty", None))

        return result

    def _load_writes(
        self, writes: list[tuple[str, str, str, bytes]] | None
    ) -> list[tuple[str, str, Any]]:
        """Load writes from database rows.

        Args:
            writes: List of (task_id, channel, type, blob) tuples.

        Returns:
            List of (task_id, channel, value) tuples with deserialized values.
        """
        if not writes:
            return []

        return [
            (task_id, channel, self.serde.loads_typed((type_, blob)))
            for task_id, channel, type_, blob in writes
        ]

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        """Dump writes for storage in database.

        Args:
            thread_id: Thread identifier.
            checkpoint_ns: Checkpoint namespace.
            checkpoint_id: Checkpoint identifier.
            task_id: Task identifier.
            task_path: Task path.
            writes: List of (channel, value) tuples.

        Returns:
            List of tuples ready for database insertion.
        """
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,  # noqa: A002
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Build WHERE clause for checkpoint queries.

        Args:
            config: Configuration with thread_id and optional checkpoint filters.
            filter: Additional metadata filters.
            before: Only return checkpoints before this checkpoint.

        Returns:
            Tuple of (WHERE clause string, list of parameter values).
        """
        wheres = []
        param_values = []

        # Filter by config
        if config:
            wheres.append("thread_id = ?")
            param_values.append(config["configurable"]["thread_id"])

            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = ?")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = ?")
                param_values.append(checkpoint_id)

        # Filter by metadata (JSON string comparison in Unity Catalog)
        if filter:
            # For Unity Catalog, metadata is stored as JSON string
            # We need to use json_extract or similar for proper filtering
            for key, value in filter.items():
                wheres.append(f"get_json_object(metadata, '$.{key}') = ?")
                param_values.append(str(value))

        # Filter by before checkpoint
        if before is not None:
            wheres.append("checkpoint_id < ?")
            param_values.append(get_checkpoint_id(before))

        where_clause = "WHERE " + " AND ".join(wheres) if wheres else ""
        return where_clause, param_values

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate next version string for a channel.

        Args:
            current: Current version string.
            channel: Channel name (unused, for compatibility).

        Returns:
            Next version string.
        """
        import random

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])

        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
