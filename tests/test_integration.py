"""Integration tests for Unity Catalog persistence with real Databricks backend.

These tests use real Databricks resources and require:
- Environment variable: DATABRICKS_SQL_WAREHOUSE_ID
- Databricks workspace access via default authentication

Tests are idempotent and clean up all created tables.

Run tests with: pytest tests/test_integration.py -v
Skip with: pytest tests/ -v -m "not integration"
"""

import os

import pytest
from databricks.sdk import WorkspaceClient
from langchain_core.messages import HumanMessage

from langgraph_unity_catalog_checkpoint.checkpoint import (
    AsyncUnityCatalogCheckpointSaver,
    UnityCatalogCheckpointSaver,
)
from langgraph_unity_catalog_checkpoint.store import UnityCatalogStore
from tests.conftest import skip_if_no_warehouse

# Test configuration from environment
CATALOG = os.environ.get("DATABRICKS_CATALOG", "main")
SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "default")
WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", "")


@pytest.fixture
def workspace_client() -> WorkspaceClient:
    """Create a WorkspaceClient for testing."""
    return WorkspaceClient()


def drop_table_if_exists(workspace_client: WorkspaceClient, table_name: str) -> None:
    """Drop a table if it exists."""
    try:
        drop_sql = f"DROP TABLE IF EXISTS {table_name}"
        workspace_client.statement_execution.execute_statement(
            statement=drop_sql,
            warehouse_id=WAREHOUSE_ID,
            wait_timeout="30s",
        )
    except Exception:
        pass  # Ignore errors during cleanup


@pytest.fixture
def integration_store(workspace_client: WorkspaceClient) -> UnityCatalogStore:
    """Create a store for integration testing with cleanup."""
    table_name = f"{CATALOG}.{SCHEMA}.integration_test_store"

    # Clean up before test
    drop_table_if_exists(workspace_client, table_name)

    store = UnityCatalogStore(
        workspace_client=workspace_client,
        catalog=CATALOG,
        schema=SCHEMA,
        table="integration_test_store",
        warehouse_id=WAREHOUSE_ID,
    )

    yield store

    # Clean up after test
    drop_table_if_exists(workspace_client, table_name)


@pytest.fixture
def integration_checkpointer(
    workspace_client: WorkspaceClient,
) -> UnityCatalogCheckpointSaver:
    """Create a checkpointer for integration testing with cleanup."""
    checkpoints_table = f"{CATALOG}.{SCHEMA}.integration_test_checkpoints"
    blobs_table = f"{CATALOG}.{SCHEMA}.integration_test_checkpoint_blobs"
    writes_table = f"{CATALOG}.{SCHEMA}.integration_test_writes"

    # Clean up before test
    drop_table_if_exists(workspace_client, checkpoints_table)
    drop_table_if_exists(workspace_client, blobs_table)
    drop_table_if_exists(workspace_client, writes_table)

    checkpointer = UnityCatalogCheckpointSaver(
        workspace_client=workspace_client,
        catalog=CATALOG,
        schema=SCHEMA,
        checkpoints_table="integration_test_checkpoints",
        checkpoint_blobs_table="integration_test_checkpoint_blobs",
        writes_table="integration_test_writes",
        warehouse_id=WAREHOUSE_ID,
    )

    yield checkpointer

    # Clean up after test
    drop_table_if_exists(workspace_client, checkpoints_table)
    drop_table_if_exists(workspace_client, blobs_table)
    drop_table_if_exists(workspace_client, writes_table)


@pytest.fixture
def integration_async_checkpointer(
    workspace_client: WorkspaceClient,
) -> AsyncUnityCatalogCheckpointSaver:
    """Create an async checkpointer for integration testing with cleanup."""
    checkpoints_table = f"{CATALOG}.{SCHEMA}.integration_test_async_checkpoints"
    blobs_table = f"{CATALOG}.{SCHEMA}.integration_test_async_checkpoint_blobs"
    writes_table = f"{CATALOG}.{SCHEMA}.integration_test_async_writes"

    # Clean up before test
    drop_table_if_exists(workspace_client, checkpoints_table)
    drop_table_if_exists(workspace_client, blobs_table)
    drop_table_if_exists(workspace_client, writes_table)

    checkpointer = AsyncUnityCatalogCheckpointSaver(
        workspace_client=workspace_client,
        catalog=CATALOG,
        schema=SCHEMA,
        checkpoints_table="integration_test_async_checkpoints",
        checkpoint_blobs_table="integration_test_async_checkpoint_blobs",
        writes_table="integration_test_async_writes",
        warehouse_id=WAREHOUSE_ID,
    )

    yield checkpointer

    # Clean up after test
    drop_table_if_exists(workspace_client, checkpoints_table)
    drop_table_if_exists(workspace_client, blobs_table)
    drop_table_if_exists(workspace_client, writes_table)


@pytest.mark.integration
@skip_if_no_warehouse
class TestStoreIntegration:
    """Integration tests for UnityCatalogStore with real backend."""

    def test_store_table_creation(self, integration_store: UnityCatalogStore) -> None:
        """Test that the store table is created successfully."""
        assert integration_store.full_table_name == f"{CATALOG}.{SCHEMA}.integration_test_store"

    def test_store_put_get(self, integration_store: UnityCatalogStore) -> None:
        """Test setting and getting values in real Unity Catalog using BaseStore interface."""
        # Set values using put() method
        integration_store.put(("test",), "key_1", {"value": "test_value_1"})
        integration_store.put(("test",), "key_2", {"value": "test_value_2"})

        # Get values using get() method
        value1 = integration_store.get(("test",), "key_1")
        value2 = integration_store.get(("test",), "key_2")
        value3 = integration_store.get(("test",), "nonexistent")

        assert value1 == {"value": "test_value_1"}
        assert value2 == {"value": "test_value_2"}
        assert value3 is None

    def test_store_search(self, integration_store: UnityCatalogStore) -> None:
        """Test searching items with namespace prefix."""
        # Set values with different namespaces
        integration_store.put(("user", "1"), "name", {"data": "Alice"})
        integration_store.put(("user", "1"), "email", {"data": "alice@example.com"})
        integration_store.put(("user", "2"), "name", {"data": "Bob"})

        # Search with namespace prefix
        results = integration_store.search(("user", "1"), limit=10)

        assert len(results) == 2
        keys = [item.key for item in results]
        assert "name" in keys
        assert "email" in keys

    def test_store_delete(self, integration_store: UnityCatalogStore) -> None:
        """Test deleting keys with namespaces."""
        # Set values
        integration_store.put(("delete",), "me_1", {"value": "value1"})
        integration_store.put(("delete",), "me_2", {"value": "value2"})

        # Delete keys
        integration_store.delete(("delete",), "me_1")
        integration_store.delete(("delete",), "me_2")

        # Verify deletion
        value1 = integration_store.get(("delete",), "me_1")
        value2 = integration_store.get(("delete",), "me_2")
        assert value1 is None
        assert value2 is None


@pytest.mark.integration
@skip_if_no_warehouse
class TestCheckpointerIntegration:
    """Integration tests for UnityCatalogCheckpointSaver with real backend."""

    def test_checkpointer_table_creation(
        self, integration_checkpointer: UnityCatalogCheckpointSaver
    ) -> None:
        """Test that the checkpointer tables are created successfully."""
        assert (
            integration_checkpointer.full_checkpoints_table
            == f"{CATALOG}.{SCHEMA}.integration_test_checkpoints"
        )
        assert (
            integration_checkpointer.full_writes_table
            == f"{CATALOG}.{SCHEMA}.integration_test_writes"
        )

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_checkpointer_put_get(
        self, integration_checkpointer: UnityCatalogCheckpointSaver
    ) -> None:
        """Test saving and retrieving a checkpoint with the sync checkpointer.

        Note: Suppressing event loop cleanup warnings from sync wrapper creating
        temporary async instances. The test functionality is correct.
        """
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id="test_checkpoint_1",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": [HumanMessage(content="Hello")]},
            channel_versions={"messages": 1},
            versions_seen={"messages": {"messages": 1}},
            pending_sends=[],
        )

        config = {
            "configurable": {
                "thread_id": "integration_test_thread",
                "checkpoint_ns": "",
                "checkpoint_id": "test_checkpoint_1",
            }
        }

        metadata = CheckpointMetadata(
            source="input",
            step=1,
            writes={},
            parents={},
        )

        # Save checkpoint (sync wrapper needs event loop)
        saved_config = integration_checkpointer.put(config, checkpoint, metadata, {})

        # Retrieve checkpoint
        retrieved_tuple = integration_checkpointer.get_tuple(saved_config)
        assert retrieved_tuple is not None
        assert retrieved_tuple.checkpoint["id"] == "test_checkpoint_1"
        assert retrieved_tuple.metadata["step"] == 1


@pytest.mark.integration
@skip_if_no_warehouse
class TestAsyncCheckpointerIntegration:
    """Integration tests for AsyncUnityCatalogCheckpointSaver with real backend."""

    @pytest.mark.asyncio
    async def test_async_checkpointer_table_creation(
        self, integration_async_checkpointer: AsyncUnityCatalogCheckpointSaver
    ) -> None:
        """Test that the async checkpointer tables are created successfully."""
        assert (
            integration_async_checkpointer.full_checkpoints_table
            == f"{CATALOG}.{SCHEMA}.integration_test_async_checkpoints"
        )
        assert (
            integration_async_checkpointer.full_writes_table
            == f"{CATALOG}.{SCHEMA}.integration_test_async_writes"
        )

    @pytest.mark.asyncio
    async def test_async_checkpointer_put_get(
        self, integration_async_checkpointer: AsyncUnityCatalogCheckpointSaver
    ) -> None:
        """Test saving and retrieving a checkpoint asynchronously."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id="test_async_checkpoint_1",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": [HumanMessage(content="Async Hello")]},
            channel_versions={"messages": 1},
            versions_seen={"messages": {"messages": 1}},
            pending_sends=[],
        )

        config = {
            "configurable": {
                "thread_id": "integration_test_async_thread",
                "checkpoint_ns": "",
                "checkpoint_id": "test_async_checkpoint_1",
            }
        }

        metadata = CheckpointMetadata(
            source="input",
            step=1,
            writes={},
            parents={},
        )

        # Save checkpoint
        saved_config = await integration_async_checkpointer.aput(config, checkpoint, metadata, {})

        # Retrieve checkpoint
        retrieved_tuple = await integration_async_checkpointer.aget_tuple(saved_config)
        assert retrieved_tuple is not None
        assert retrieved_tuple.checkpoint["id"] == "test_async_checkpoint_1"
        assert retrieved_tuple.metadata["step"] == 1
