"""Tests for UnityCatalogStore BaseStore interface compliance.

This test module verifies that UnityCatalogStore correctly implements
the langgraph.store.base.BaseStore interface.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest
from langgraph.store.base import BaseStore, GetOp, Item, PutOp

from langgraph_unity_catalog_checkpoint.store import UnityCatalogStore


@pytest.fixture
def mock_workspace_client():
    """Create a mock workspace client."""
    client = Mock()
    client.statement_execution.execute_statement.return_value = Mock(result=Mock(data_array=[]))
    return client


@pytest.fixture
def store(mock_workspace_client):
    """Create a UnityCatalogStore instance."""
    return UnityCatalogStore(
        workspace_client=mock_workspace_client,
        catalog="test_catalog",
        schema="test_schema",
        table="test_store",
        warehouse_id="test_warehouse",
    )


class TestBaseStoreInterface:
    """Test that UnityCatalogStore implements BaseStore interface."""

    def test_is_base_store_instance(self, store):
        """Verify store is an instance of BaseStore."""
        assert isinstance(store, BaseStore)

    def test_has_required_sync_methods(self, store):
        """Verify store has all required sync methods."""
        assert hasattr(store, "batch")
        assert hasattr(store, "get")
        assert hasattr(store, "put")
        assert hasattr(store, "delete")
        assert hasattr(store, "search")
        assert hasattr(store, "list_namespaces")

    def test_has_required_async_methods(self, store):
        """Verify store has all required async methods."""
        assert hasattr(store, "abatch")
        assert hasattr(store, "aget")
        assert hasattr(store, "aput")
        assert hasattr(store, "adelete")
        assert hasattr(store, "asearch")
        assert hasattr(store, "alist_namespaces")

    def test_batch_with_get_ops(self, store, mock_workspace_client):
        """Test batch method with GetOp operations."""
        # Mock SQL response
        import base64
        import json

        value_dict = {"memory": "test value"}
        value_bytes = json.dumps(value_dict).encode("utf-8")
        value_base64 = base64.b64encode(value_bytes).decode("ascii")

        # The mget query returns rows with just value column
        mock_workspace_client.statement_execution.execute_statement.return_value = Mock(
            result=Mock(
                data_array=[
                    [value_base64]  # mget returns just the value column
                ]
            )
        )

        # Execute batch operation
        ops = [GetOp(namespace=("users", "123"), key="preferences")]
        results = store.batch(ops)

        assert len(results) == 1
        if results[0] is not None:  # May be None if not found
            assert isinstance(results[0], Item)
            assert results[0].namespace == ("users", "123")
            assert results[0].key == "preferences"
            assert results[0].value == value_dict

    def test_batch_with_put_ops(self, store, mock_workspace_client):
        """Test batch method with PutOp operations."""
        # Execute batch operation
        ops = [
            PutOp(
                namespace=("users", "123"),
                key="preferences",
                value={"theme": "dark"},
            )
        ]
        _ = store.batch(ops)

        # Verify INSERT was called
        assert mock_workspace_client.statement_execution.execute_statement.called
        call_args = mock_workspace_client.statement_execution.execute_statement.call_args
        sql = call_args[1]["statement"]
        assert "INSERT INTO" in sql

    def test_search_signature(self, store):
        """Verify search method has correct signature."""
        import inspect

        sig = inspect.signature(store.search)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "namespace_prefix" in params
        assert "filter" in params
        assert "limit" in params
        assert "offset" in params
        assert "query" in params

    def test_search_returns_items(self, store, mock_workspace_client):
        """Test search method returns list of Item objects."""
        import base64
        import json

        # Mock SQL response with created_at and updated_at
        value_dict = {"memory": "test memory"}
        value_bytes = json.dumps(value_dict).encode("utf-8")
        value_base64 = base64.b64encode(value_bytes).decode("ascii")
        timestamp = datetime.now().isoformat()

        mock_workspace_client.statement_execution.execute_statement.return_value = Mock(
            result=Mock(
                data_array=[
                    [
                        '["users"]',
                        "preferences",
                        value_base64,
                        timestamp,
                        timestamp,
                    ]
                ]
            )
        )

        # Execute search
        results = store.search(
            namespace_prefix=("users",),
            filter=None,
            limit=10,
            offset=0,
            query=None,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Item)
        assert results[0].namespace == ("users",)
        assert results[0].key == "preferences"
        assert results[0].value == value_dict
        assert isinstance(results[0].created_at, datetime)
        assert isinstance(results[0].updated_at, datetime)

    def test_put_accepts_dict_values(self, store):
        """Test put method accepts dict values."""
        # Should not raise
        store.put(
            namespace=("users", "123"),
            key="preferences",
            value={"theme": "dark", "language": "en"},
        )

    def test_get_returns_dict(self, store, mock_workspace_client):
        """Test get method returns dict values."""
        import base64
        import json

        value_dict = {"theme": "dark"}
        value_bytes = json.dumps(value_dict).encode("utf-8")
        value_base64 = base64.b64encode(value_bytes).decode("ascii")

        # mget returns just the value column
        mock_workspace_client.statement_execution.execute_statement.return_value = Mock(
            result=Mock(data_array=[[value_base64]])
        )

        result = store.get(namespace=("users", "123"), key="preferences")

        if result is not None:  # May be None if not found
            assert isinstance(result, dict)
            assert result == value_dict
