"""Unit tests for UnityCatalogStore."""

from unittest.mock import Mock

import pytest

from langgraph_unity_catalog_checkpoint.store import UnityCatalogStore


class TestUnityCatalogStoreInit:
    """Tests for UnityCatalogStore initialization."""

    def test_init_creates_table(
        self, mock_workspace_client: Mock, store_config: dict[str, str]
    ) -> None:
        """Test that initialization creates the table."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        # Verify execute_statement was called for table creation
        mock_workspace_client.statement_execution.execute_statement.assert_called_once()
        call_args = mock_workspace_client.statement_execution.execute_statement.call_args

        # Verify CREATE TABLE statement
        assert "CREATE TABLE IF NOT EXISTS" in call_args.kwargs["statement"]
        assert store.full_table_name in call_args.kwargs["statement"]
        assert call_args.kwargs["warehouse_id"] == store_config["warehouse_id"]

    def test_init_sets_attributes(
        self, mock_workspace_client: Mock, store_config: dict[str, str]
    ) -> None:
        """Test that initialization sets correct attributes."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        assert store.catalog == store_config["catalog"]
        assert store.schema == store_config["schema"]
        assert store.table == store_config["table"]
        assert store.warehouse_id == store_config["warehouse_id"]
        assert (
            store.full_table_name
            == f"{store_config['catalog']}.{store_config['schema']}.{store_config['table']}"
        )


class TestUnityCatalogStoreErrorHandling:
    """Tests for error handling."""

    def test_init_table_creation_failure(
        self, mock_workspace_client: Mock, store_config: dict[str, str]
    ) -> None:
        """Test that table creation failure is handled."""
        mock_workspace_client.statement_execution.execute_statement.side_effect = Exception(
            "Table creation failed"
        )

        with pytest.raises(Exception) as exc_info:
            UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        assert "Table creation failed" in str(exc_info.value)

    def test_get_execution_failure(
        self, mock_workspace_client: Mock, store_config: dict[str, str]
    ) -> None:
        """Test get handles execution failures."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        # Make subsequent call fail
        mock_workspace_client.statement_execution.execute_statement.side_effect = Exception(
            "Query failed"
        )

        with pytest.raises(Exception) as exc_info:
            store.get(namespace=("user",), key="key1")

        assert "Query failed" in str(exc_info.value)

    def test_put_execution_failure(
        self, mock_workspace_client: Mock, store_config: dict[str, str]
    ) -> None:
        """Test put handles execution failures."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        mock_workspace_client.statement_execution.execute_statement.side_effect = Exception(
            "Write failed"
        )

        with pytest.raises(Exception) as exc_info:
            store.put(namespace=("user",), key="key1", value={"data": "value"})

        assert "Write failed" in str(exc_info.value)


class TestUnityCatalogStoreHelpers:
    """Tests for helper methods."""

    def test_escape_string(self, mock_workspace_client: Mock, store_config: dict[str, str]) -> None:
        """Test string escaping for SQL."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        # Test single quote escaping
        assert store._escape_string("test'value") == "test''value"
        assert store._escape_string("test") == "test"
        assert store._escape_string("test''value") == "test''''value"

    def test_bytes_to_hex(self, mock_workspace_client: Mock, store_config: dict[str, str]) -> None:
        """Test bytes to hex conversion."""
        store = UnityCatalogStore(workspace_client=mock_workspace_client, **store_config)

        assert store._bytes_to_hex(b"hello") == "68656c6c6f"
        assert store._bytes_to_hex(b"") == ""
        assert store._bytes_to_hex(b"\x00\xff") == "00ff"
