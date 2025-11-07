"""Synchronous Unity Catalog checkpoint saver implementation.

This module contains the sync checkpoint saver for Unity Catalog, following the
pattern from langgraph.checkpoint.postgres.

The implementation uses the base class and adds sync-specific table initialization.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph_unity_catalog_checkpoint.checkpoint.aio import AsyncUnityCatalogCheckpointSaver
from langgraph_unity_catalog_checkpoint.checkpoint.base import BaseUnityCatalogSaver
from langgraph_unity_catalog_checkpoint.logging_config import logger


class UnityCatalogCheckpointSaver(BaseUnityCatalogSaver):
    """Synchronous checkpointer that stores checkpoints in Unity Catalog.

    This is the main checkpoint saver for Unity Catalog that provides
    synchronous checkpoint operations. It extends the base class and implements
    the synchronous checkpoint interface.
    """

    lock: threading.Lock

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
        """Initialize the sync Unity Catalog checkpoint saver.

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

        # Initialize tables synchronously during construction
        self._init_tables()

        # Create a single async saver instance to reuse across all method calls
        # This avoids reinitializing tables on every operation
        # Skip table init since we already initialized them above
        self._async_saver = AsyncUnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            checkpoints_table=checkpoints_table,
            checkpoint_blobs_table=checkpoint_blobs_table,
            writes_table=writes_table,
            warehouse_id=warehouse_id,
            serde=serde,
            skip_table_init=True,  # Tables already initialized
        )

    def _init_tables(self) -> None:
        """Initialize the Unity Catalog tables for storing checkpoints and writes."""
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
            # Access the result to ensure checkpoints table creation is complete
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
            # Access the result to ensure checkpoint blobs table creation is complete
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
            # Access the result to ensure writes table creation is complete
            _ = response3.result

            logger.info(
                f"Initialized Unity Catalog tables: {self.full_checkpoints_table}, "
                f"{self.full_checkpoint_blobs_table}, {self.full_writes_table}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in Unity Catalog if they don't
        already exist. It MUST be called directly by the user the first time
        checkpointer is used.
        """
        # Tables are already initialized in __init__
        pass

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from Unity Catalog based
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An iterator of matching checkpoint tuples.
        """
        import asyncio

        # Convert async generator to sync by running in asyncio
        async def _collect_items() -> list[CheckpointTuple]:
            items = []
            async for item in self._async_saver.alist(
                config, filter=filter, before=before, limit=limit
            ):
                items.append(item)
            return items

        items = asyncio.run(_collect_items())
        yield from items

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from Unity Catalog based on the
        provided config. If the config contains a `checkpoint_id` key, the checkpoint
        with the matching thread ID and checkpoint_id is retrieved. Otherwise, the
        latest checkpoint for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        import asyncio

        # Delegate to the cached async saver
        return asyncio.run(self._async_saver.aget_tuple(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

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
        import asyncio

        # Delegate to the cached async saver
        return asyncio.run(self._async_saver.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        import asyncio

        # Delegate to the cached async saver
        return asyncio.run(self._async_saver.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        import asyncio

        # Delegate to the cached async saver
        return asyncio.run(self._async_saver.adelete_thread(thread_id))


__all__ = ["UnityCatalogCheckpointSaver", "AsyncUnityCatalogCheckpointSaver"]
