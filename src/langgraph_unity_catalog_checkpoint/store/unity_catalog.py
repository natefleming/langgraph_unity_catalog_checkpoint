"""Synchronous Unity Catalog store implementation.

This module provides a synchronous BaseStore implementation that uses Databricks
Unity Catalog as the storage backend, following the pattern from langgraph.store.postgres.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
)

from langgraph_unity_catalog_checkpoint.logging_config import logger
from langgraph_unity_catalog_checkpoint.store.base import (
    CREATE_STORE_TABLE_SQL,
    BaseUnityCatalogStore,
)


class UnityCatalogStore(BaseStore, BaseUnityCatalogStore):
    """Synchronous Unity Catalog implementation of LangGraph's BaseStore interface.

    This store uses Databricks Unity Catalog tables to persist key-value pairs
    organized by namespaces. It follows the pattern from langgraph.store.postgres.

    The store creates a table with the following schema:
    - prefix (STRING): JSON-encoded namespace tuple (part of primary key)
    - key (STRING): The key for the value (part of primary key)
    - value (BINARY): The binary value associated with the key
    - created_at (TIMESTAMP): When the record was created
    - updated_at (TIMESTAMP): When the record was last updated

    Example:
        ```python
        from langgraph_unity_catalog_checkpoint.store import UnityCatalogStore
        from databricks.sdk import WorkspaceClient

        store = UnityCatalogStore(
            workspace_client=WorkspaceClient(),
            catalog="main",
            schema="langgraph",
            warehouse_id="abc123"
        )

        # Store data with namespacing
        store.mset([(("users", "123"), "prefs", b'{"theme": "dark"}')])

        # Retrieve data
        items = store.mget([("users", "123", "prefs")])
        ```

    Args:
        workspace_client: Databricks WorkspaceClient instance.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        table: Table name for storing key-value pairs (default: "store").
        warehouse_id: SQL warehouse ID to use for queries.
    """

    lock: threading.Lock

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        table: str = "store",
        warehouse_id: str | None = None,
    ) -> None:
        """Initialize the Unity Catalog Store."""
        BaseStore.__init__(self)
        BaseUnityCatalogStore.__init__(
            self,
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            table=table,
            warehouse_id=warehouse_id,
        )
        self.lock = threading.Lock()

        # Initialize the table if it doesn't exist
        self._init_table()

    def _init_table(self) -> None:
        """Initialize the Unity Catalog table for storing key-value pairs."""
        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=CREATE_STORE_TABLE_SQL.format(table_name=self.full_table_name),
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            # Access the result to ensure completion
            _ = response.result
            logger.info(f"Initialized Unity Catalog table: {self.full_table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize table: {e}")
            raise

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations synchronously.

        Args:
            ops: Iterable of operations to execute.

        Returns:
            List of results corresponding to the operations.
        """
        from typing import cast

        # Group operations by type
        grouped: dict[type[Op], list[tuple[int, Op]]] = {}
        results: list[Result] = []
        for i, op in enumerate(ops):
            op_type = type(op)
            if op_type not in grouped:
                grouped[op_type] = []
            grouped[op_type].append((i, op))
            results.append(None)

        # Execute each group
        with self.lock:
            if GetOp in grouped:
                self._batch_get_ops(cast(Sequence[tuple[int, GetOp]], grouped[GetOp]), results)

            if PutOp in grouped:
                self._batch_put_ops(cast(Sequence[tuple[int, PutOp]], grouped[PutOp]))

            if ListNamespacesOp in grouped:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped[ListNamespacesOp],
                    ),
                    results,
                )

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        Args:
            ops: Iterable of operations to execute.

        Returns:
            List of results corresponding to the operations.
        """
        import asyncio

        # Convert to list to avoid consuming iterator multiple times
        ops_list = list(ops)
        return await asyncio.to_thread(self.batch, ops_list)

    def _batch_get_ops(self, get_ops: Sequence[tuple[int, GetOp]], results: list[Result]) -> None:
        """Execute a batch of get operations.

        Args:
            get_ops: Sequence of (index, GetOp) tuples.
            results: List to store results in.
        """
        if not get_ops:
            return

        # Use the _mget method to retrieve values
        keys = [(*op.namespace, op.key) for _, op in get_ops]
        values = self._mget(keys)

        # Convert to Item objects and store in results
        for (idx, op), value in zip(get_ops, values, strict=False):
            if value is not None:
                try:
                    # Deserialize value to dict
                    value_dict = json.loads(value.decode("utf-8"))
                    if not isinstance(value_dict, dict):
                        value_dict = {"value": value_dict}

                    results[idx] = Item(
                        value=value_dict,
                        key=op.key,
                        namespace=op.namespace,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If deserialization fails, return None
                    results[idx] = None
            else:
                results[idx] = None

    def _batch_put_ops(self, put_ops: Sequence[tuple[int, PutOp]]) -> None:
        """Execute a batch of put operations.

        Args:
            put_ops: Sequence of (index, PutOp) tuples.
        """
        if not put_ops:
            return

        # Convert to mset format
        items = []
        for _, op in put_ops:
            key = (*op.namespace, op.key)
            # Serialize value to bytes
            if op.value is None:
                # Delete operation
                self._mdelete([key])
            else:
                value_bytes = json.dumps(op.value).encode("utf-8")
                items.append((key, value_bytes))

        if items:
            self._mset(items)

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        """Execute a batch of list namespaces operations.

        Args:
            list_ops: Sequence of (index, ListNamespacesOp) tuples.
            results: List to store results in.
        """
        for idx, _op in list_ops:
            # Build SQL query to get unique prefixes
            sql_query = f"SELECT DISTINCT prefix FROM {self.full_table_name}"

            try:
                response: StatementResponse = (
                    self.workspace_client.statement_execution.execute_statement(
                        statement=sql_query,
                        warehouse_id=self.warehouse_id,
                        wait_timeout="30s",
                    )
                )

                namespaces = []
                if response.result and response.result.data_array:
                    for row in response.result.data_array:
                        if row and row[0]:
                            namespace = self._decode_namespace(row[0])
                            namespaces.append(namespace)

                results[idx] = namespaces

            except Exception as e:
                logger.error(f"Failed to list namespaces: {e}")
                results[idx] = []

    def _mget(self, keys: Sequence[tuple[str, ...]]) -> list[bytes | None]:
        """Get the values associated with the given namespaced keys (internal method).

        Args:
            keys: A sequence of (namespace, key) tuples to retrieve values for.

        Returns:
            A list of values (as bytes) or None for each key that doesn't exist.
            The order matches the order of the input keys.
        """
        if not keys:
            return []

        # Parse namespace and key from each tuple
        items = []
        for key_tuple in keys:
            if len(key_tuple) < 2:
                logger.warning(f"Invalid key tuple: {key_tuple}, need at least 2 elements")
                items.append(((), ""))
                continue
            namespace = key_tuple[:-1]
            key = key_tuple[-1]
            items.append((namespace, key))

        # Build query
        query = self._build_batch_get_query(items)

        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=query, warehouse_id=self.warehouse_id, wait_timeout="30s"
                )
            )
            # Access result to ensure completion
            result_data = response.result

            # Build a dictionary of results
            results_dict: dict[tuple[str, str], bytes] = {}
            if result_data and result_data.data_array:
                for row in result_data.data_array:
                    if row and len(row) >= 3:
                        prefix = row[0]
                        key_val = row[1]
                        value_b64 = row[2]
                        if prefix is not None and key_val is not None and value_b64:
                            try:
                                # Unity Catalog returns BINARY as base64, not hex
                                import base64

                                value = base64.b64decode(value_b64)
                                results_dict[(prefix, key_val)] = value
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Failed to decode value for {prefix}:{key_val}: {e}"
                                )

            # Return results in the same order as input keys
            output = []
            for namespace, key in items:
                prefix_str = self._encode_namespace(namespace)
                value = results_dict.get((prefix_str, key))
                output.append(value)

            return output

        except Exception as e:
            logger.error(f"Failed to get values for keys: {e}")
            raise

    def get(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None:
        """Get a single value from the store.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key within the namespace

        Returns:
            Value dict if found, None otherwise
        """
        results = self._mget([(*namespace, key)])
        if results and results[0]:
            # Deserialize bytes to dict
            try:
                return json.loads(results[0].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                return None
        return None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Store a single value for a key in a namespace.

        Args:
            namespace: Hierarchical namespace tuple (e.g., ("users", "123"))
            key: Key within the namespace
            value: Value to store as a dictionary
        """
        # Serialize dict value to bytes
        value_bytes = json.dumps(value).encode("utf-8")

        # Use _mset for single item
        self._mset([((*namespace, key), value_bytes)])

    def _mset(self, key_value_pairs: Sequence[tuple[tuple[str, ...], bytes]]) -> None:
        """Set the values for the given namespaced key-value pairs (internal method).

        Args:
            key_value_pairs: A sequence of ((namespace, key), value) tuples to set.
                The first element is a tuple containing the namespace components and key.
        """
        if not key_value_pairs:
            return

        # Parse namespace, key, and value from each tuple
        items = []
        for key_tuple, value in key_value_pairs:
            if len(key_tuple) < 2:
                logger.warning(f"Invalid key tuple: {key_tuple}, need at least 2 elements")
                continue
            namespace = key_tuple[:-1]
            key = key_tuple[-1]
            items.append((namespace, key, value))

        # Build query
        query = self._build_put_query(items)

        try:
            self.workspace_client.statement_execution.execute_statement(
                statement=query, warehouse_id=self.warehouse_id, wait_timeout="30s"
            )
            logger.debug(f"Set {len(items)} key-value pairs")
        except Exception as e:
            logger.error(f"Failed to set key-value pairs: {e}")
            raise

    def _mdelete(self, keys: Sequence[tuple[str, ...]]) -> None:
        """Delete the given namespaced keys and their associated values (internal method).

        Args:
            keys: A sequence of (namespace, key) tuples to delete.
        """
        if not keys:
            return

        # Parse namespace and key from each tuple
        items = []
        for key_tuple in keys:
            if len(key_tuple) < 2:
                logger.warning(f"Invalid key tuple: {key_tuple}, need at least 2 elements")
                continue
            namespace = key_tuple[:-1]
            key = key_tuple[-1]
            items.append((namespace, key))

        # Build query
        query = self._build_delete_query(items)

        try:
            self.workspace_client.statement_execution.execute_statement(
                statement=query, warehouse_id=self.warehouse_id, wait_timeout="30s"
            )
            logger.debug(f"Deleted {len(items)} keys")
        except Exception as e:
            logger.error(f"Failed to delete keys: {e}")
            raise

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        limit: int = 10,
        offset: int = 0,
        query: str | None = None,
    ) -> list[Item]:
        """Search for items in the store within a namespace.

        Args:
            namespace_prefix: Namespace prefix to search within
            filter: Optional key-value pairs to filter results
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination
            query: Optional search query for filtering (searches in keys)

        Returns:
            List of Item objects matching the search criteria
        """
        import base64

        prefix_str = self._encode_namespace(namespace_prefix)

        # Build SQL query with all filters
        where_clauses = [f"prefix = '{self._escape_string(prefix_str)}'"]

        if query:
            # Search for query string in keys
            where_clauses.append(f"key LIKE '%{self._escape_string(query)}%'")

        # Note: filter parameter for value-based filtering would require
        # deserializing all values which is expensive. For now, we only
        # support namespace and key-based filtering in SQL.
        # Value filtering can be done in memory after retrieval if needed.

        where_clause = " AND ".join(where_clauses)
        sql_query = f"""
        SELECT prefix, key, value, created_at, updated_at
        FROM {self.full_table_name}
        WHERE {where_clause}
        ORDER BY created_at DESC, key
        LIMIT {limit} OFFSET {offset}
        """

        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=sql_query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )

            results = []
            if response.result and response.result.data_array:
                for row in response.result.data_array:
                    if row and len(row) >= 5:
                        prefix = row[0]
                        key = row[1]
                        value_hex = row[2]
                        created_at_str = row[3]
                        updated_at_str = row[4]

                        if prefix and key and value_hex:
                            try:
                                # Decode namespace
                                namespace = self._decode_namespace(prefix)

                                # Decode value from base64 and deserialize
                                value_bytes = base64.b64decode(value_hex)
                                # Try to deserialize as JSON dict
                                try:
                                    value_dict = json.loads(value_bytes.decode("utf-8"))
                                    if not isinstance(value_dict, dict):
                                        # If not a dict, wrap it
                                        value_dict = {"value": value_dict}
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    # If not JSON, store raw bytes as hex string
                                    value_dict = {"data": value_hex}

                                # Parse timestamps
                                created_at = (
                                    datetime.fromisoformat(created_at_str)
                                    if created_at_str
                                    else datetime.now()
                                )
                                updated_at = (
                                    datetime.fromisoformat(updated_at_str)
                                    if updated_at_str
                                    else datetime.now()
                                )

                                # Apply filter if provided (in-memory filtering)
                                if filter:
                                    matches = all(value_dict.get(k) == v for k, v in filter.items())
                                    if not matches:
                                        continue

                                item = Item(
                                    value=value_dict,
                                    key=key,
                                    namespace=namespace,
                                    created_at=created_at,
                                    updated_at=updated_at,
                                )
                                results.append(item)

                            except Exception as e:
                                logger.warning(f"Failed to decode item {prefix}:{key}: {e}")

            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []


__all__ = ["UnityCatalogStore"]
