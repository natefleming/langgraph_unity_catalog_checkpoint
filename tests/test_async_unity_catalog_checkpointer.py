"""Unit tests for AsyncUnityCatalogCheckpointSaver."""

from unittest.mock import Mock

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph_unity_catalog_checkpoint.checkpoint import AsyncUnityCatalogCheckpointSaver
from tests.conftest import create_mock_result


class TestAsyncUnityCatalogCheckpointerInit:
    """Tests for AsyncUnityCatalogCheckpointSaver initialization."""

    def test_init_creates_tables(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test that initialization creates both tables."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        # Should create 3 tables (checkpoints, checkpoint_blobs, and writes)
        assert mock_workspace_client.statement_execution.execute_statement.call_count == 3

        # Verify attributes
        assert checkpointer.catalog == checkpointer_config["catalog"]
        assert checkpointer.schema == checkpointer_config["schema"]
        assert checkpointer.warehouse_id == checkpointer_config["warehouse_id"]


class TestAsyncUnityCatalogCheckpointerAPut:
    """Tests for aput method."""

    @pytest.mark.asyncio
    async def test_aput_basic_checkpoint(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test putting a basic checkpoint asynchronously."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )
        mock_workspace_client.statement_execution.execute_statement.reset_mock()

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "async_checkpoint_1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"test": "value"},
            "channel_versions": {},
            "versions_seen": {},
        }

        metadata: CheckpointMetadata = {"source": "async_test", "step": 1}

        result_config = await checkpointer.aput(config, checkpoint, metadata, {})

        # Verify returned config
        assert result_config["configurable"]["thread_id"] == "async_thread_1"
        assert result_config["configurable"]["checkpoint_id"] == "async_checkpoint_1"

        # Verify MERGE was executed
        call_args = mock_workspace_client.statement_execution.execute_statement.call_args
        assert "MERGE INTO" in call_args.kwargs["statement"]


class TestAsyncUnityCatalogCheckpointerAPutWrites:
    """Tests for aput_writes method."""

    @pytest.mark.asyncio
    async def test_aput_writes_empty(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aput_writes with empty writes does nothing."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )
        mock_workspace_client.statement_execution.execute_statement.reset_mock()

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_id": "async_checkpoint_1",
            }
        }

        await checkpointer.aput_writes(config, [], "async_task_1")

        # Should not execute any query
        mock_workspace_client.statement_execution.execute_statement.assert_not_called()

    @pytest.mark.asyncio
    async def test_aput_writes_multiple(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aput_writes with multiple writes."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )
        mock_workspace_client.statement_execution.execute_statement.reset_mock()

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_id": "async_checkpoint_1",
                "checkpoint_ns": "",
            }
        }

        writes = [
            ("channel_1", {"data": "async_value1"}),
            ("channel_2", {"data": "async_value2"}),
        ]

        await checkpointer.aput_writes(config, writes, "async_task_1")

        # Verify MERGE was executed (writes use upsert logic)
        call_args = mock_workspace_client.statement_execution.execute_statement.call_args
        assert "MERGE INTO" in call_args.kwargs["statement"]


class TestAsyncUnityCatalogCheckpointerAGetTuple:
    """Tests for aget_tuple method."""

    @pytest.mark.asyncio
    async def test_aget_tuple_not_found(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aget_tuple returns None when checkpoint not found."""
        mock_result = create_mock_result([])
        mock_workspace_client.statement_execution.execute_statement.return_value = mock_result

        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )
        mock_workspace_client.statement_execution.execute_statement.reset_mock()
        mock_workspace_client.statement_execution.execute_statement.return_value = mock_result

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        result = await checkpointer.aget_tuple(config)

        assert result is None

    @pytest.mark.asyncio
    async def test_aget_tuple_with_checkpoint(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aget_tuple retrieves checkpoint."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "async_checkpoint_1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }

        import base64

        type_, checkpoint_bytes = checkpointer.serde.dumps_typed(checkpoint)

        # Mock result needs all columns: thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        # Unity Catalog returns BINARY as base64
        checkpoint_result = create_mock_result(
            [
                [
                    "async_thread_1",
                    "",
                    "async_checkpoint_1",
                    None,
                    type_,
                    base64.b64encode(checkpoint_bytes).decode(),
                    '{"source": "async"}',
                ]
            ]
        )
        # Mock blob result (empty since channel_values is empty)
        blob_result = create_mock_result([])
        writes_result = create_mock_result([])

        mock_workspace_client.statement_execution.execute_statement.side_effect = [
            checkpoint_result,
            blob_result,  # for channel values query
            writes_result,
        ]

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        result = await checkpointer.aget_tuple(config)

        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "async_checkpoint_1"
        assert result.metadata == {"source": "async"}


class TestAsyncUnityCatalogCheckpointerAList:
    """Tests for alist method."""

    @pytest.mark.asyncio
    async def test_alist_no_checkpoints(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test alist with no checkpoints."""
        mock_result = create_mock_result([])
        mock_workspace_client.statement_execution.execute_statement.return_value = mock_result

        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )
        mock_workspace_client.statement_execution.execute_statement.reset_mock()
        mock_workspace_client.statement_execution.execute_statement.return_value = mock_result

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        result = []
        async for item in checkpointer.alist(config):
            result.append(item)

        assert result == []

    @pytest.mark.asyncio
    async def test_alist_with_checkpoints(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test alist with checkpoints."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "async_checkpoint_1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }

        import base64

        type_, checkpoint_bytes = checkpointer.serde.dumps_typed(checkpoint)

        # Mock result needs all columns: thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        # Unity Catalog returns BINARY as base64
        checkpoints_result = create_mock_result(
            [
                [
                    "async_thread_1",
                    "",
                    "async_checkpoint_1",
                    None,
                    type_,
                    base64.b64encode(checkpoint_bytes).decode(),
                    "{}",
                ]
            ]
        )
        blob_result = create_mock_result([])
        writes_result = create_mock_result([])

        mock_workspace_client.statement_execution.execute_statement.side_effect = [
            checkpoints_result,
            blob_result,  # for channel values query
            writes_result,
        ]

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        result = []
        async for item in checkpointer.alist(config):
            result.append(item)

        assert len(result) == 1
        assert result[0].config["configurable"]["checkpoint_id"] == "async_checkpoint_1"

    @pytest.mark.asyncio
    async def test_alist_with_limit(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test alist respects limit parameter."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "async_checkpoint_1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }

        import base64

        type_, checkpoint_bytes = checkpointer.serde.dumps_typed(checkpoint)

        # Mock result needs all columns: thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        # Unity Catalog returns BINARY as base64
        checkpoints_result = create_mock_result(
            [
                [
                    "async_thread_1",
                    "",
                    "async_checkpoint_1",
                    None,
                    type_,
                    base64.b64encode(checkpoint_bytes).decode(),
                    "{}",
                ]
            ]
        )
        blob_result = create_mock_result([])
        writes_result = create_mock_result([])

        mock_workspace_client.statement_execution.execute_statement.side_effect = [
            checkpoints_result,
            blob_result,  # for channel values query
            writes_result,
        ]

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        result = []
        async for item in checkpointer.alist(config, limit=10):
            result.append(item)

        # Verify LIMIT was in query (search all calls since fixture already did table init)
        all_statements = [
            call.kwargs.get("statement", "")
            for call in mock_workspace_client.statement_execution.execute_statement.call_args_list
        ]
        assert any(
            "LIMIT 10" in stmt for stmt in all_statements
        ), f"LIMIT 10 not found in any statement. Statements: {all_statements}"


class TestAsyncUnityCatalogCheckpointerErrorHandling:
    """Tests for error handling in async checkpointer."""

    @pytest.mark.asyncio
    async def test_aput_handles_execution_error(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aput handles execution errors."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        mock_workspace_client.statement_execution.execute_statement.side_effect = Exception(
            "Async write failed"
        )

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "async_checkpoint_1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
        }

        with pytest.raises(Exception) as exc_info:
            await checkpointer.aput(config, checkpoint, {}, {})

        assert "Async write failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_aget_tuple_handles_execution_error(
        self, mock_workspace_client: Mock, checkpointer_config: dict[str, str]
    ) -> None:
        """Test aget_tuple handles execution errors gracefully by returning None."""
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=mock_workspace_client, **checkpointer_config
        )

        mock_workspace_client.statement_execution.execute_statement.side_effect = Exception(
            "Async read failed"
        )

        config = {
            "configurable": {
                "thread_id": "async_thread_1",
                "checkpoint_ns": "",
            }
        }

        # aget_tuple should handle errors gracefully and return None
        result = await checkpointer.aget_tuple(config)
        assert result is None
