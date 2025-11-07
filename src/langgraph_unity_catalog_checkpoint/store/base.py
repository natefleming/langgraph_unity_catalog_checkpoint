"""Base Unity Catalog store implementation with shared functionality.

This module provides the base class for Unity Catalog store implementations,
following the pattern from langgraph.store.postgres.base.

Based on: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/store/postgres/base.py
"""

from __future__ import annotations

import json

from databricks.sdk import WorkspaceClient

# SQL template for creating the store table
CREATE_STORE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    prefix STRING NOT NULL,
    key STRING NOT NULL,
    value BINARY NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (prefix, key)
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.feature.allowColumnDefaults' = 'supported',
    'description' = 'LangChain BaseStore key-value storage in Unity Catalog'
)
"""


class BaseUnityCatalogStore:
    """Base class for Unity Catalog store implementations.

    This class provides shared functionality for both sync and async store
    implementations, including SQL templates, table initialization, and helper methods.
    """

    workspace_client: WorkspaceClient
    catalog: str
    schema: str
    table: str
    warehouse_id: str
    full_table_name: str

    # SQL templates
    SELECT_SQL = """
    SELECT prefix, key, value
    FROM {table_name}
    WHERE prefix = ? AND key = ?
    """

    SELECT_NAMESPACE_SQL = """
    SELECT DISTINCT prefix
    FROM {table_name}
    WHERE prefix LIKE ?
    """

    # For batched operations
    BATCH_GET_SQL = """
    SELECT prefix, key, value
    FROM {table_name}
    WHERE (prefix, key) IN ({placeholders})
    """

    # Note: Delta Lake doesn't support standard SQL MERGE with complex VALUE clauses in some versions
    # Using INSERT with ON CONFLICT for better compatibility
    INSERT_SQL = """
    INSERT INTO {table_name} (prefix, key, value, created_at, updated_at)
    VALUES {values_placeholder}
    ON CONFLICT (prefix, key) DO UPDATE SET
        value = excluded.value,
        updated_at = excluded.updated_at
    """

    DELETE_SQL = """
    DELETE FROM {table_name}
    WHERE prefix = ? AND key = ?
    """

    BATCH_DELETE_SQL = """
    DELETE FROM {table_name}
    WHERE (prefix, key) IN ({placeholders})
    """

    LIST_KEYS_SQL = """
    SELECT key
    FROM {table_name}
    WHERE prefix = ?
    ORDER BY key
    """

    LIST_KEYS_WITH_PREFIX_SQL = """
    SELECT key
    FROM {table_name}
    WHERE prefix = ? AND key LIKE ?
    ORDER BY key
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        table: str = "store",
        warehouse_id: str | None = None,
    ) -> None:
        """Initialize the base Unity Catalog store.

        Args:
            workspace_client: Databricks WorkspaceClient instance.
            catalog: Unity Catalog catalog name.
            schema: Unity Catalog schema name.
            table: Table name for storing key-value pairs (default: "store").
            warehouse_id: SQL warehouse ID to use for queries (default: None).
        """
        self.workspace_client = workspace_client
        self.catalog = catalog
        self.schema = schema
        self.table = table
        self.warehouse_id = warehouse_id
        self.full_table_name = f"{catalog}.{schema}.{table}"

    @staticmethod
    def _escape_string(s: str) -> str:
        """Escape a string for use in SQL.

        Args:
            s: String to escape.

        Returns:
            Escaped string safe for SQL.
        """
        return s.replace("'", "''")

    @staticmethod
    def _bytes_to_hex(data: bytes) -> str:
        """Convert bytes to hex string for SQL BINARY.

        Args:
            data: Bytes to convert.

        Returns:
            Hex string representation.
        """
        return data.hex()

    @staticmethod
    def _encode_namespace(namespace: tuple[str, ...]) -> str:
        """Encode a namespace tuple as a string.

        Args:
            namespace: Tuple of namespace components.

        Returns:
            JSON-encoded string representation (compact format without spaces).
        """
        return json.dumps(namespace, separators=(",", ":"))

    @staticmethod
    def _decode_namespace(namespace_str: str) -> tuple[str, ...]:
        """Decode a namespace string back to a tuple.

        Args:
            namespace_str: JSON-encoded namespace string.

        Returns:
            Tuple of namespace components.
        """
        result = json.loads(namespace_str)
        if isinstance(result, list):
            return tuple(result)
        return result

    def _build_get_query(self, namespace: tuple[str, ...], key: str) -> str:
        """Build a query to get a single item.

        Args:
            namespace: The namespace tuple.
            key: The key to retrieve.

        Returns:
            SQL query string.
        """
        ns = self._escape_string(self._encode_namespace(namespace))
        k = self._escape_string(key)
        return f"""
        SELECT value
        FROM {self.full_table_name}
        WHERE prefix = '{ns}' AND key = '{k}'
        """

    def _build_batch_get_query(self, items: list[tuple[tuple[str, ...], str]]) -> str:
        """Build a query to get multiple items.

        Args:
            items: List of (namespace, key) tuples.

        Returns:
            SQL query string.
        """
        conditions = " OR ".join(
            [
                f"(prefix = '{self._escape_string(self._encode_namespace(ns))}' AND key = '{self._escape_string(k)}')"
                for ns, k in items
            ]
        )
        return f"""
        SELECT prefix, key, value
        FROM {self.full_table_name}
        WHERE {conditions}
        """

    def _build_put_query(self, items: list[tuple[tuple[str, ...], str, bytes]]) -> str:
        """Build a query to put multiple items.

        Args:
            items: List of (namespace, key, value) tuples.

        Returns:
            SQL query string.
        """
        values = ", ".join(
            [
                f"('{self._escape_string(self._encode_namespace(ns))}', '{self._escape_string(k)}', X'{v.hex()}', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"
                for ns, k, v in items
            ]
        )
        return f"""
        INSERT INTO {self.full_table_name} (prefix, key, value, created_at, updated_at)
        VALUES {values}
        """

    def _build_delete_query(self, items: list[tuple[tuple[str, ...], str]]) -> str:
        """Build a query to delete multiple items.

        Args:
            items: List of (namespace, key) tuples.

        Returns:
            SQL query string.
        """
        conditions = " OR ".join(
            [
                f"(prefix = '{self._escape_string(self._encode_namespace(ns))}' AND key = '{self._escape_string(k)}')"
                for ns, k in items
            ]
        )
        return f"""
        DELETE FROM {self.full_table_name}
        WHERE {conditions}
        """

    def _build_list_namespaces_query(self, prefix: tuple[str, ...] | None = None) -> str:
        """Build a query to list namespaces.

        Args:
            prefix: Optional namespace prefix to filter by.

        Returns:
            SQL query string.
        """
        if prefix:
            prefix_str = self._escape_string(self._encode_namespace(prefix))
            return f"""
            SELECT DISTINCT prefix
            FROM {self.full_table_name}
            WHERE prefix LIKE '{prefix_str}%'
            ORDER BY prefix
            """
        return f"""
        SELECT DISTINCT prefix
        FROM {self.full_table_name}
        ORDER BY prefix
        """


__all__ = ["BaseUnityCatalogStore", "CREATE_STORE_TABLE_SQL"]
