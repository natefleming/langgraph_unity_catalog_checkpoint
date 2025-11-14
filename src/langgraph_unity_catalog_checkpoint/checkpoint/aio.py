"""Async Unity Catalog checkpoint saver implementation.

This module contains the async checkpoint saver for Unity Catalog, following the
pattern from langgraph.checkpoint.postgres.aio.

Based on: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py
"""

from __future__ import annotations

import asyncio
import json
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
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph_unity_catalog_checkpoint.checkpoint.base import BaseUnityCatalogSaver
from langgraph_unity_catalog_checkpoint.logging_config import logger


class AsyncUnityCatalogCheckpointSaver(BaseUnityCatalogSaver):
    """Asynchronous checkpointer that stores checkpoints in Unity Catalog."""

    lock: asyncio.Lock

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
        skip_table_init: bool = False,
    ) -> None:
        """Initialize the async Unity Catalog checkpoint saver.

        Args:
            workspace_client: Databricks WorkspaceClient for executing SQL.
            catalog: Unity Catalog name.
            schema: Schema name within the catalog.
            checkpoints_table: Table name for checkpoints (default: "checkpoints").
            checkpoint_blobs_table: Table name for checkpoint blobs (default: "checkpoint_blobs").
            writes_table: Table name for checkpoint writes (default: "checkpoint_writes").
            warehouse_id: SQL warehouse ID for executing statements.
            serde: Serializer for checkpoint data.
            skip_table_init: Skip table initialization (used when tables already initialized).
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
        self.lock = asyncio.Lock()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, will be set later
            self.loop = None  # type: ignore[assignment]

        # Initialize tables synchronously during construction (unless skipped)
        if not skip_table_init:
            self._init_tables_sync()

    def _init_tables_sync(self) -> None:
        """Initialize the Unity Catalog tables (sync version for __init__)."""
        try:
            # Create checkpoints table
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

            # Create checkpoint blobs table
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

            # Create checkpoint writes table
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
                f"Initialized Unity Catalog tables: {self.full_checkpoints_table}, "
                f"{self.full_checkpoint_blobs_table}, {self.full_writes_table}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in Unity Catalog if they don't
        already exist. It MUST be called directly by the user the first time
        checkpointer is used.
        """
        await asyncio.to_thread(self._init_tables_sync)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from Unity Catalog based
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An asynchronous iterator of matching checkpoint tuples.
        """
        where, params = self._search_where(config, filter, before)
        query = (
            self.SELECT_CHECKPOINTS_SQL.format(table_name=self.full_checkpoints_table)
            + " "
            + where
            + " ORDER BY checkpoint_id DESC"
        )
        if limit:
            query += f" LIMIT {limit}"

        # Replace ? placeholders with positional format for Unity Catalog
        for param in params:
            query = query.replace("?", f"'{self._escape_string(str(param))}'", 1)

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            if not response.result or not response.result.data_array:
                return

            for row in response.result.data_array:
                checkpoint_tuple = await self._load_checkpoint_tuple(row)
                if checkpoint_tuple:
                    yield checkpoint_tuple

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from Unity Catalog based on the
        provided config. If the config contains a `checkpoint_id` key, the checkpoint
        with the matching thread ID and checkpoint_id is retrieved. Otherwise, the
        latest checkpoint for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            where = f"WHERE thread_id = '{self._escape_string(thread_id)}' AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}' AND checkpoint_id = '{self._escape_string(checkpoint_id)}'"
        else:
            where = f"WHERE thread_id = '{self._escape_string(thread_id)}' AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}' ORDER BY checkpoint_id DESC LIMIT 1"

        query = (
            self.SELECT_CHECKPOINTS_SQL.format(table_name=self.full_checkpoints_table) + " " + where
        )

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            if not response.result or not response.result.data_array:
                return None

            row = response.result.data_array[0]
            return await self._load_checkpoint_tuple(row)

        except Exception as e:
            logger.error(f"Failed to get checkpoint tuple: {e}")
            return None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to Unity Catalog. The checkpoint is associated
        with the provided config and its parent config (if any).

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
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        try:
            # Make a copy to avoid mutating the original checkpoint (following PostgreSQL pattern)
            copy = checkpoint.copy()
            copy["channel_values"] = copy["channel_values"].copy()
            
            # Separate inline primitive values from blob values (following PostgreSQL pattern)
            # Inline: None, str, int, float, bool
            # Blobs: everything else (lists, dicts, objects, etc.)
            channel_values = copy["channel_values"]
            blob_values = {}
            inline_values = {}
            
            for k, v in channel_values.items():
                if v is None or isinstance(v, (str, int, float, bool)):
                    inline_values[k] = v
                else:
                    blob_values[k] = v
            
            # Store blobs in blobs table
            blob_tuples = list(
                self._dump_blobs(thread_id, checkpoint_ns, blob_values, new_versions)
            )
            if blob_tuples:
                await self._upsert_blobs_batch(blob_tuples)

            # Store checkpoint with inline primitive values only
            copy["channel_values"] = inline_values
            await self._upsert_checkpoint(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                copy,
                metadata,
                config,
            )

            logger.debug(f"Stored checkpoint {checkpoint_id} for thread {thread_id}")

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

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

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
            # Collect all writes and insert in a single batch
            write_tuples = list(
                self._dump_writes(
                    thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, writes
                )
            )
            if write_tuples:
                await self._upsert_writes_batch(write_tuples)

            logger.debug(f"Stored {len(writes)} writes for checkpoint {checkpoint_id}")

        except Exception as e:
            logger.error(f"Failed to put writes: {e}")
            raise

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        try:
            # Delete from all three tables
            for table in [
                self.full_checkpoints_table,
                self.full_checkpoint_blobs_table,
                self.full_writes_table,
            ]:
                query = f"DELETE FROM {table} WHERE thread_id = '{self._escape_string(thread_id)}'"
                await asyncio.to_thread(
                    self.workspace_client.statement_execution.execute_statement,
                    statement=query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )

            logger.info(f"Deleted all data for thread {thread_id}")

        except Exception as e:
            logger.error(f"Failed to delete thread: {e}")
            raise

    async def _load_checkpoint_tuple(self, row: list[Any]) -> CheckpointTuple | None:
        """Convert a database row into a CheckpointTuple object asynchronously.

        Args:
            row: A row from the database containing checkpoint data.

        Returns:
            CheckpointTuple: A structured representation of the checkpoint.
        """
        try:
            import base64

            thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id = (
                row[0],
                row[1],
                row[2],
                row[3],
            )
            type_ = row[4] if row[4] else "msgpack"
            # Checkpoint is now stored as base64-encoded string
            checkpoint_str = row[5] if row[5] else None
            metadata_str = row[6] if row[6] else "{}"

            if not checkpoint_str:
                return None

            # Decode base64 string to bytes
            checkpoint_bytes = base64.b64decode(checkpoint_str)
            
            # Deserialize checkpoint with correct type
            checkpoint_data = await asyncio.to_thread(
                self.serde.loads_typed, (type_, checkpoint_bytes)
            )
            metadata = json.loads(metadata_str) if metadata_str else {}

            # Load channel values from blobs
            channel_values_from_blobs = await self._load_channel_values_async(
                thread_id, checkpoint_ns, checkpoint_data
            )

            # Load pending writes
            pending_writes = await self._load_pending_writes_async(
                thread_id, checkpoint_ns, checkpoint_id
            )

            # Merge inline channel_values (primitives) with blob channel_values (complex objects)
            # Following PostgreSQL pattern
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
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
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=pending_writes,
            )

        except Exception as e:
            logger.error(f"Failed to load checkpoint tuple: {e}")
            return None

    async def _load_channel_values_async(
        self, thread_id: str, checkpoint_ns: str, checkpoint_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Load channel values from blobs table asynchronously."""
        channel_versions = checkpoint_data.get("channel_versions", {})
        if not channel_versions:
            return {}

        channels = list(channel_versions.keys())
        if not channels:
            return {}

        channel_conditions = " OR ".join(
            [
                f"(channel = '{self._escape_string(ch)}' AND version = '{self._escape_string(str(ver))}')"
                for ch, ver in channel_versions.items()
            ]
        )

        query = f"""
        SELECT channel, type, blob
        FROM {self.full_checkpoint_blobs_table}
        WHERE thread_id = '{self._escape_string(thread_id)}'
            AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
            AND ({channel_conditions})
        """

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            if not response.result or not response.result.data_array:
                return {}

            import base64

            result = {}
            for row in response.result.data_array:
                channel, type_, blob_str = row[0], row[1], row[2]
                if type_ != "empty" and blob_str:
                    # Blob is stored as base64-encoded string
                    blob = base64.b64decode(blob_str)
                    result[channel] = await asyncio.to_thread(self.serde.loads_typed, (type_, blob))

            return result

        except Exception as e:
            logger.error(f"Failed to load channel values: {e}")
            return {}

    async def _load_pending_writes_async(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for a checkpoint asynchronously."""
        query = f"""
        SELECT task_id, channel, type, blob
        FROM {self.full_writes_table}
        WHERE thread_id = '{self._escape_string(thread_id)}'
            AND checkpoint_ns = '{self._escape_string(checkpoint_ns)}'
            AND checkpoint_id = '{self._escape_string(checkpoint_id)}'
        ORDER BY task_id, idx
        """

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            if not response.result or not response.result.data_array:
                return []

            import base64

            return [
                (
                    row[0],  # task_id
                    row[1],  # channel
                    await asyncio.to_thread(
                        self.serde.loads_typed, (row[2], base64.b64decode(row[3]))
                    ),  # type, blob (Unity Catalog returns BINARY as base64)
                )
                for row in response.result.data_array
            ]

        except Exception as e:
            logger.error(f"Failed to load pending writes: {e}")
            return []

    async def _upsert_blob(self, blob_tuple: tuple[str, str, str, str, str, bytes | None]) -> None:
        """Upsert a blob into the checkpoint_blobs table."""
        thread_id, checkpoint_ns, channel, version, type_, blob = blob_tuple

        blob_hex = self._bytes_to_hex(blob) if blob else "NULL"
        query = f"""
        MERGE INTO {self.full_checkpoint_blobs_table} AS target
        USING (SELECT
            '{self._escape_string(thread_id)}' AS thread_id,
            '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
            '{self._escape_string(channel)}' AS channel,
            '{self._escape_string(version)}' AS version,
            '{self._escape_string(type_)}' AS type,
            X'{blob_hex}' AS blob
        ) AS source
        ON target.thread_id = source.thread_id
            AND target.checkpoint_ns = source.checkpoint_ns
            AND target.channel = source.channel
            AND target.version = source.version
        WHEN MATCHED THEN UPDATE SET
            type = source.type,
            blob = source.blob
        WHEN NOT MATCHED THEN INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
        VALUES (source.thread_id, source.checkpoint_ns, source.channel, source.version, source.type, source.blob)
        """

        await asyncio.to_thread(
            self.workspace_client.statement_execution.execute_statement,
            statement=query,
            warehouse_id=self.warehouse_id,
            wait_timeout="30s",
        )

    async def _upsert_blobs_batch(
        self, blob_tuples: list[tuple[str, str, str, str, str, bytes | None]]
    ) -> None:
        """Upsert multiple blobs into the checkpoint_blobs table in a single query.

        This is a performance optimization that batches multiple blob inserts into one
        SQL statement, reducing the number of round trips to Unity Catalog.

        Args:
            blob_tuples: List of blob tuples to insert.
        """
        if not blob_tuples:
            return

        import base64

        # Build UNION ALL source rows for all blobs
        source_rows = []
        for thread_id, checkpoint_ns, channel, version, type_, blob in blob_tuples:
            # Convert blob bytes to base64 string
            if blob:
                blob_str = base64.b64encode(blob).decode('utf-8')
                blob_escaped = self._escape_string(blob_str)
                blob_value = f"'{blob_escaped}'"
            else:
                blob_value = "NULL"
            
            source_rows.append(f"""
            SELECT
                '{self._escape_string(thread_id)}' AS thread_id,
                '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
                '{self._escape_string(channel)}' AS channel,
                '{self._escape_string(version)}' AS version,
                '{self._escape_string(type_)}' AS type,
                {blob_value} AS blob
            """)

        source_query = " UNION ALL ".join(source_rows)

        query = f"""
        MERGE INTO {self.full_checkpoint_blobs_table} AS target
        USING ({source_query}) AS source
        ON target.thread_id = source.thread_id
            AND target.checkpoint_ns = source.checkpoint_ns
            AND target.channel = source.channel
            AND target.version = source.version
        WHEN MATCHED THEN UPDATE SET
            type = source.type,
            blob = source.blob
        WHEN NOT MATCHED THEN INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
        VALUES (source.thread_id, source.checkpoint_ns, source.channel, source.version, source.type, source.blob)
        """

        await asyncio.to_thread(
            self.workspace_client.statement_execution.execute_statement,
            statement=query,
            warehouse_id=self.warehouse_id,
            wait_timeout="30s",
        )

    async def _upsert_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint: dict[str, Any],
        metadata: CheckpointMetadata,
        config: RunnableConfig,
    ) -> None:
        """Upsert a checkpoint into the checkpoints table."""
        import base64
        
        type_, checkpoint_data = await asyncio.to_thread(self.serde.dumps_typed, checkpoint)
        # Store as base64-encoded string (easier to debug than binary)
        checkpoint_str = base64.b64encode(checkpoint_data).decode('utf-8')
        checkpoint_escaped = self._escape_string(checkpoint_str)
        
        metadata_json = json.dumps(get_serializable_checkpoint_metadata(config, metadata))
        metadata_str = self._escape_string(metadata_json)
        
        type_escaped = self._escape_string(type_)

        query = f"""
        MERGE INTO {self.full_checkpoints_table} AS target
        USING (SELECT
            '{self._escape_string(thread_id)}' AS thread_id,
            '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
            '{self._escape_string(checkpoint_id)}' AS checkpoint_id,
            {f"'{self._escape_string(parent_checkpoint_id)}'" if parent_checkpoint_id else "NULL"} AS parent_checkpoint_id,
            '{type_escaped}' AS type,
            '{checkpoint_escaped}' AS checkpoint,
            '{metadata_str}' AS metadata
        ) AS source
        ON target.thread_id = source.thread_id
            AND target.checkpoint_ns = source.checkpoint_ns
            AND target.checkpoint_id = source.checkpoint_id
        WHEN MATCHED THEN UPDATE SET
            parent_checkpoint_id = source.parent_checkpoint_id,
            type = source.type,
            checkpoint = source.checkpoint,
            metadata = source.metadata
        WHEN NOT MATCHED THEN INSERT (
            thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        )
        VALUES (
            source.thread_id, source.checkpoint_ns, source.checkpoint_id,
            source.parent_checkpoint_id, source.type, source.checkpoint, source.metadata
        )
        """

        await asyncio.to_thread(
            self.workspace_client.statement_execution.execute_statement,
            statement=query,
            warehouse_id=self.warehouse_id,
            wait_timeout="30s",
        )

    async def _upsert_write(
        self,
        write_tuple: tuple[str, str, str, str, str, int, str, str, bytes],
    ) -> None:
        """Upsert a write into the checkpoint_writes table."""
        import base64
        
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            task_path,
            idx,
            channel,
            type_,
            blob,
        ) = write_tuple

        blob_str = base64.b64encode(blob).decode('utf-8')
        blob_escaped = self._escape_string(blob_str)
        
        query = f"""
        MERGE INTO {self.full_writes_table} AS target
        USING (SELECT
            '{self._escape_string(thread_id)}' AS thread_id,
            '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
            '{self._escape_string(checkpoint_id)}' AS checkpoint_id,
            '{self._escape_string(task_id)}' AS task_id,
            '{self._escape_string(task_path)}' AS task_path,
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
            task_path = source.task_path,
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

        await asyncio.to_thread(
            self.workspace_client.statement_execution.execute_statement,
            statement=query,
            warehouse_id=self.warehouse_id,
            wait_timeout="30s",
        )

    async def _upsert_writes_batch(
        self, write_tuples: list[tuple[str, str, str, str, str, int, str, str, bytes]]
    ) -> None:
        """Upsert multiple writes into the checkpoint_writes table in a single query.

        This is a performance optimization that batches multiple write inserts into one
        SQL statement, reducing the number of round trips to Unity Catalog.

        Args:
            write_tuples: List of write tuples to insert.
        """
        if not write_tuples:
            return

        import base64

        # Build UNION ALL source rows for all writes
        source_rows = []
        for (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            task_path,
            idx,
            channel,
            type_,
            blob,
        ) in write_tuples:
            blob_str = base64.b64encode(blob).decode('utf-8')
            blob_escaped = self._escape_string(blob_str)
            
            source_rows.append(f"""
            SELECT
                '{self._escape_string(thread_id)}' AS thread_id,
                '{self._escape_string(checkpoint_ns)}' AS checkpoint_ns,
                '{self._escape_string(checkpoint_id)}' AS checkpoint_id,
                '{self._escape_string(task_id)}' AS task_id,
                '{self._escape_string(task_path)}' AS task_path,
                {idx} AS idx,
                '{self._escape_string(channel)}' AS channel,
                '{self._escape_string(type_)}' AS type,
                '{blob_escaped}' AS blob
            """)

        source_query = " UNION ALL ".join(source_rows)

        query = f"""
        MERGE INTO {self.full_writes_table} AS target
        USING ({source_query}) AS source
        ON target.thread_id = source.thread_id
            AND target.checkpoint_ns = source.checkpoint_ns
            AND target.checkpoint_id = source.checkpoint_id
            AND target.task_id = source.task_id
            AND target.idx = source.idx
        WHEN MATCHED THEN UPDATE SET
            task_path = source.task_path,
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

        await asyncio.to_thread(
            self.workspace_client.statement_execution.execute_statement,
            statement=query,
            warehouse_id=self.warehouse_id,
            wait_timeout="30s",
        )

    # Sync wrapper methods for compatibility
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncUnityCatalogCheckpointSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.alist(...)`."
                )
        except RuntimeError:
            loop = None  # type: ignore[assignment]

        # For list, we need to iterate through async generator
        if loop is None or self.loop is None:
            # No event loop running or no loop attached, use asyncio.run() for each item
            aiter_ = self.alist(config, filter=filter, before=before, limit=limit)

            async def _collect_all() -> list[CheckpointTuple]:
                return [item async for item in aiter_]

            items = asyncio.run(_collect_all())
            yield from items
        else:
            # Event loop is running, use run_coroutine_threadsafe
            aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
            while True:
                try:
                    yield asyncio.run_coroutine_threadsafe(
                        anext(aiter_),  # type: ignore[arg-type]
                        self.loop,
                    ).result()
                except StopAsyncIteration:
                    break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncUnityCatalogCheckpointSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)`."
                )
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(self.aget_tuple(config))

        # Event loop is running but not this instance's loop, use run_coroutine_threadsafe
        if self.loop:
            return asyncio.run_coroutine_threadsafe(self.aget_tuple(config), self.loop).result()
        else:
            # No loop attached to this instance, use asyncio.run()
            return asyncio.run(self.aget_tuple(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncUnityCatalogCheckpointSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aput(...)`."
                )
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))

        # Event loop is running but not this instance's loop, use run_coroutine_threadsafe
        if self.loop:
            return asyncio.run_coroutine_threadsafe(
                self.aput(config, checkpoint, metadata, new_versions), self.loop
            ).result()
        else:
            # No loop attached to this instance, use asyncio.run()
            return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncUnityCatalogCheckpointSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aput_writes(...)`."
                )
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(self.aput_writes(config, writes, task_id, task_path))

        # Event loop is running but not this instance's loop, use run_coroutine_threadsafe
        if self.loop:
            return asyncio.run_coroutine_threadsafe(
                self.aput_writes(config, writes, task_id, task_path), self.loop
            ).result()
        else:
            # No loop attached to this instance, use asyncio.run()
            return asyncio.run(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncUnityCatalogCheckpointSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.adelete_thread(...)`."
                )
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(self.adelete_thread(thread_id))

        # Event loop is running but not this instance's loop, use run_coroutine_threadsafe
        if self.loop:
            return asyncio.run_coroutine_threadsafe(
                self.adelete_thread(thread_id), self.loop
            ).result()
        else:
            # No loop attached to this instance, use asyncio.run()
            return asyncio.run(self.adelete_thread(thread_id))


__all__ = ["AsyncUnityCatalogCheckpointSaver"]
