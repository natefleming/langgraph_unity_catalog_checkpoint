"""Shallow Unity Catalog checkpoint saver implementation.

This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
It is meant to be a light-weight drop-in replacement for the full Unity Catalog checkpoint
saver that supports most of the LangGraph persistence functionality with the exception
of time travel.

Based on: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/shallow.py
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph_unity_catalog_checkpoint.checkpoint.base import BaseUnityCatalogSaver
from langgraph_unity_catalog_checkpoint.logging_config import logger


class ShallowUnityCatalogSaver(BaseUnityCatalogSaver):
    """A checkpoint saver that uses Unity Catalog to store checkpoints.

    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    It is meant to be a light-weight drop-in replacement for the full UnityCatalogCheckpointSaver
    that supports most of the LangGraph persistence functionality with the exception of time travel.
    """

    # Simplified table schemas for shallow storage (no checkpoint_id in primary key)
    CREATE_CHECKPOINTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        type STRING,
        checkpoint BINARY NOT NULL,
        metadata STRING NOT NULL DEFAULT '{{}}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (thread_id, checkpoint_ns)
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported',
        'description' = 'LangGraph shallow checkpoint storage in Unity Catalog'
    )
    """

    CREATE_CHECKPOINT_BLOBS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id STRING NOT NULL,
        checkpoint_ns STRING NOT NULL DEFAULT '',
        channel STRING NOT NULL,
        type STRING NOT NULL,
        blob BINARY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (thread_id, checkpoint_ns, channel)
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.feature.allowColumnDefaults' = 'supported',
        'description' = 'LangGraph shallow checkpoint blobs storage in Unity Catalog'
    )
    """

    lock: threading.Lock

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        checkpoints_table: str = "shallow_checkpoints",
        checkpoint_blobs_table: str = "shallow_checkpoint_blobs",
        writes_table: str = "shallow_checkpoint_writes",
        warehouse_id: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the shallow Unity Catalog checkpoint saver.

        Args:
            workspace_client: Databricks WorkspaceClient for executing SQL.
            catalog: Unity Catalog name.
            schema: Schema name within the catalog.
            checkpoints_table: Table name for checkpoints (default: "shallow_checkpoints").
            checkpoint_blobs_table: Table name for checkpoint blobs (default: "shallow_checkpoint_blobs").
            writes_table: Table name for checkpoint writes (default: "shallow_checkpoint_writes").
            warehouse_id: SQL warehouse ID for executing statements.
            serde: Serializer for checkpoint data.
        """
        super().__init__(
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            checkpoints_table=checkpoints_table,
            checkpoint_blobs_table=checkpoint_blobs_table,
            writes_table=writes_table,
            warehouse_id=warehouse_id,
            serde=serde,
        )
        self.lock = threading.Lock()

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in Unity Catalog if they don't
        already exist. It MUST be called directly by the user the first time
        checkpointer is used.
        """
        try:
            # Create checkpoints table (shallow - no checkpoint_id in PK)
            response1: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=self.CREATE_CHECKPOINTS_TABLE_SQL.format(
                        table_name=self.full_checkpoints_table
                    ),
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            _ = response1.result

            # Create checkpoint blobs table (shallow - no version in PK)
            response2: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=self.CREATE_CHECKPOINT_BLOBS_TABLE_SQL.format(
                        table_name=self.full_checkpoint_blobs_table
                    ),
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            _ = response2.result

            # Create checkpoint writes table (standard)
            response3: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=self.CREATE_CHECKPOINT_WRITES_TABLE_SQL.format(
                        table_name=self.full_writes_table
                    ),
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            _ = response3.result

            logger.info(
                f"Initialized shallow Unity Catalog tables: {self.full_checkpoints_table}, "
                f"{self.full_checkpoint_blobs_table}, {self.full_writes_table}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize shallow tables: {e}")
            raise

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        For ShallowUnityCatalogSaver, this method returns a list with ONLY the
        most recent checkpoint.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An iterator of matching checkpoint tuples (max 1 for shallow).
        """
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Query for the most recent checkpoint
        query = f"""
        SELECT
            thread_id,
            checkpoint_ns,
            checkpoint,
            metadata
        FROM {self.full_checkpoints_table}
        WHERE thread_id = '{self._escape_string(thread_id)}'
            AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
        """

        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )

            if not response.result or not response.result.data_array:
                return

            for row in response.result.data_array:
                # Unity Catalog returns BINARY data as base64
                import base64

                checkpoint_data = self.serde.loads(base64.b64decode(row[2]))  # checkpoint
                metadata = eval(row[3]) if row[3] else {}  # metadata

                # Load channel values from blobs
                channel_values_from_blobs = self._load_channel_values(
                    thread_id, checkpoint_ns, checkpoint_data
                )

                # Load pending writes
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_data.get("id", "")
                )

                # Merge inline channel_values (primitives) with blob channel_values (complex objects)
                # Following PostgreSQL pattern
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_data.get("id", ""),
                        }
                    },
                    checkpoint={
                        **checkpoint_data,
                        "channel_values": {
                            **(checkpoint_data.get("channel_values") or {}),
                            **channel_values_from_blobs,
                        },
                    },
                    metadata=metadata,
                    parent_config=None,  # Shallow doesn't track parents
                    pending_writes=pending_writes,
                )

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return

    def _load_channel_values(
        self, thread_id: str, checkpoint_ns: str, checkpoint_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Load channel values from blobs table."""
        channel_versions = checkpoint_data.get("channel_versions", {})
        if not channel_versions:
            return {}

        # Build query to get all channel values
        channels = list(channel_versions.keys())
        if not channels:
            return {}

        channels_str = ", ".join([f"'{self._escape_string(c)}'" for c in channels])
        query = f"""
        SELECT channel, type, blob
        FROM {self.full_checkpoint_blobs_table}
        WHERE thread_id = '{self._escape_string(thread_id)}'
            AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
            AND channel IN ({channels_str})
        """

        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )

            if not response.result or not response.result.data_array:
                return {}

            result = {}
            for row in response.result.data_array:
                channel, type_, blob_hex = row[0], row[1], row[2]
                if type_ != "empty" and blob_hex:
                    # Unity Catalog returns BINARY data as base64
                    import base64

                    blob = base64.b64decode(blob_hex)
                    result[channel] = self.serde.loads_typed((type_, blob))

            return result
        except Exception as e:
            logger.error(f"Failed to load channel values: {e}")
            return {}

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for a checkpoint."""
        query = f"""
        SELECT task_id, channel, type, blob
        FROM {self.full_writes_table}
        WHERE thread_id = '{self._escape_string(thread_id)}'
            AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
            AND checkpoint_id = '{self._escape_string(checkpoint_id)}'
        ORDER BY task_id, idx
        """

        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )

            if not response.result or not response.result.data_array:
                return []

            # Unity Catalog returns BINARY data as base64
            import base64

            return [
                (
                    row[0],  # task_id
                    row[1],  # channel
                    self.serde.loads_typed((row[2], base64.b64decode(row[3]))),  # type, blob
                )
                for row in response.result.data_array
            ]
        except Exception as e:
            logger.error(f"Failed to load pending writes: {e}")
            return []

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        For ShallowUnityCatalogSaver, this retrieves the most recent checkpoint
        for the given thread.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        for checkpoint_tuple in self.list(config):
            return checkpoint_tuple
        return None

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        For ShallowUnityCatalogSaver, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint.get("id", "")

        try:
            # Delete old writes (keep only current and previous)
            delete_writes_sql = f"""
            DELETE FROM {self.full_writes_table}
            WHERE thread_id = '{self._escape_string(thread_id)}'
                AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
                AND checkpoint_id NOT IN (
                    '{self._escape_string(checkpoint_id)}',
                    '{self._escape_string(config["configurable"].get("checkpoint_id", ""))}'
                )
            """
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=delete_writes_sql,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            _ = response.result

            # Make a copy to avoid mutating the original checkpoint (following PostgreSQL pattern)
            copy = checkpoint.copy()
            copy["channel_values"] = copy["channel_values"].copy()
            
            # Separate inline primitive values from blob values (following PostgreSQL pattern)
            channel_values = copy["channel_values"]
            blob_values = {}
            inline_values = {}
            
            for k, v in channel_values.items():
                if v is None or isinstance(v, (str, int, float, bool)):
                    inline_values[k] = v
                else:
                    blob_values[k] = v
            
            # Upsert blobs
            import base64
            
            for thread_id_, checkpoint_ns_, channel, type_, blob in self._dump_blobs(
                thread_id, checkpoint_ns, blob_values, new_versions
            ):
                if blob:
                    blob_str = base64.b64encode(blob).decode('utf-8')
                    blob_escaped = self._escape_string(blob_str)
                    blob_value = f"'{blob_escaped}'"
                else:
                    blob_value = "NULL"
                    
                upsert_blob_sql = f"""
                MERGE INTO {self.full_checkpoint_blobs_table} AS target
                USING (SELECT
                    '{self._escape_string(thread_id_)}' AS thread_id,
                    '{self._escape_string(checkpoint_ns_)}' AS checkpoint_ns,
                    '{self._escape_string(channel)}' AS channel,
                    '{self._escape_string(type_)}' AS type,
                    {blob_value} AS blob
                ) AS source
                ON target.thread_id = source.thread_id
                    AND target.checkpoint_ns = source.checkpoint_ns
                    AND target.channel = source.channel
                WHEN MATCHED THEN UPDATE SET
                    type = source.type,
                    blob = source.blob
                WHEN NOT MATCHED THEN INSERT (thread_id, checkpoint_ns, channel, type, blob)
                VALUES (source.thread_id, source.checkpoint_ns, source.channel, source.type, source.blob)
                """
                response = self.workspace_client.statement_execution.execute_statement(
                    statement=upsert_blob_sql,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
                _ = response.result

            # Upsert checkpoint with inline primitive values only
            import base64
            import json
            
            copy["channel_values"] = inline_values
            checkpoint_data = self.serde.dumps(copy)
            checkpoint_str = base64.b64encode(checkpoint_data).decode('utf-8')
            checkpoint_escaped = self._escape_string(checkpoint_str)
            
            metadata_json = json.dumps(get_serializable_checkpoint_metadata(config, metadata))
            metadata_str = self._escape_string(metadata_json)

            upsert_checkpoint_sql = f"""
            MERGE INTO {self.full_checkpoints_table} AS target
            USING (SELECT
                '{self._escape_string(thread_id)}' AS thread_id,
                '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
                '{checkpoint_escaped}' AS checkpoint,
                '{metadata_str}' AS metadata
            ) AS source
            ON target.thread_id = source.thread_id
                AND target.checkpoint_ns = source.checkpoint_ns
            WHEN MATCHED THEN UPDATE SET
                checkpoint = source.checkpoint,
                metadata = source.metadata
            WHEN NOT MATCHED THEN INSERT (thread_id, checkpoint_ns, checkpoint, metadata)
            VALUES (source.thread_id, source.checkpoint_ns, source.checkpoint, source.metadata)
            """
            response = self.workspace_client.statement_execution.execute_statement(
                statement=upsert_checkpoint_sql,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )
            _ = response.result

            logger.debug(f"Stored shallow checkpoint {checkpoint_id} for thread {thread_id}")

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        except Exception as e:
            logger.error(f"Failed to put checkpoint: {e}")
            raise

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        try:
            for params in self._dump_writes(
                thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, writes
            ):
                (
                    tid,
                    cns,
                    cid,
                    tsk_id,
                    tsk_path,
                    idx,
                    channel,
                    type_,
                    blob,
                ) = params

                import base64
                blob_str = base64.b64encode(blob).decode('utf-8')
                blob_escaped = self._escape_string(blob_str)
                
                insert_write_sql = f"""
                MERGE INTO {self.full_writes_table} AS target
                USING (SELECT
                    '{self._escape_string(tid)}' AS thread_id,
                    '{self._escape_string(cns)}' AS checkpoint_ns,
                    '{self._escape_string(cid)}' AS checkpoint_id,
                    '{self._escape_string(tsk_id)}' AS task_id,
                    '{self._escape_string(tsk_path)}' AS task_path,
                    {idx} AS idx,
                    '{self._escape_string(channel)}' AS channel,
                    '{self._escape_string(type_)}' AS type,
                    '{blob_escaped}' AS blob
                ) AS source
                ON target.thread_id = source.thread_id
                    AND target.checkpoint_ns = source.checkpoint_ns
                    AND target.checkpoint_id = source.checkpoint_id
                    AND target.task_id = source.task_id
                    AND target.idx = source.idx
                WHEN MATCHED THEN UPDATE SET
                    channel = source.channel,
                    type = source.type,
                    blob = source.blob
                WHEN NOT MATCHED THEN INSERT (
                    thread_id, checkpoint_ns, checkpoint_id, task_id, task_path,
                    idx, channel, type, blob
                )
                VALUES (
                    source.thread_id, source.checkpoint_ns, source.checkpoint_id,
                    source.task_id, source.task_path, source.idx, source.channel,
                    source.type, source.blob
                )
                """
                response: StatementResponse = (
                    self.workspace_client.statement_execution.execute_statement(
                        statement=insert_write_sql,
                        warehouse_id=self.warehouse_id,
                        wait_timeout="30s",
                    )
                )
                _ = response.result

            logger.debug(f"Stored {len(writes)} writes for checkpoint {checkpoint_id}")

        except Exception as e:
            logger.error(f"Failed to put writes: {e}")
            raise


class AsyncShallowUnityCatalogSaver(ShallowUnityCatalogSaver):
    """Async version of ShallowUnityCatalogSaver.

    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    Uses async/await pattern for non-blocking I/O operations.
    """

    lock: asyncio.Lock

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        checkpoints_table: str = "shallow_checkpoints",
        checkpoint_blobs_table: str = "shallow_checkpoint_blobs",
        writes_table: str = "shallow_checkpoint_writes",
        warehouse_id: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the async shallow Unity Catalog checkpoint saver."""
        super().__init__(
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            checkpoints_table=checkpoints_table,
            checkpoint_blobs_table=checkpoint_blobs_table,
            writes_table=writes_table,
            warehouse_id=warehouse_id,
            serde=serde,
        )
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously."""
        await asyncio.to_thread(super().setup)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints asynchronously."""
        for checkpoint_tuple in await asyncio.to_thread(
            list, self.list(config, filter=filter, before=before, limit=limit)
        ):
            yield checkpoint_tuple

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple asynchronously."""
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint asynchronously."""
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes asynchronously."""
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)


__all__ = ["ShallowUnityCatalogSaver", "AsyncShallowUnityCatalogSaver"]
