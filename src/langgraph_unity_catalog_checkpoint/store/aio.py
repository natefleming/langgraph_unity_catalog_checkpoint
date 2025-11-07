"""Async Unity Catalog store implementation.

This module contains the async store for Unity Catalog, following the
pattern from langgraph.store.postgres.aio.

Based on: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/langgraph/store/postgres/aio.py
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import datetime
from typing import Any, cast

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore

from langgraph_unity_catalog_checkpoint.logging_config import logger
from langgraph_unity_catalog_checkpoint.store.base import (
    CREATE_STORE_TABLE_SQL,
    BaseUnityCatalogStore,
)


class AsyncUnityCatalogStore(AsyncBatchedBaseStore, BaseUnityCatalogStore):
    """Asynchronous Unity Catalog-backed store for LangGraph.

    This store provides async key-value storage using Databricks Unity Catalog
    as the backend. It follows the pattern from langgraph.store.postgres.aio.

    Example:
        ```python
        from langgraph_unity_catalog_checkpoint.store import AsyncUnityCatalogStore
        from databricks.sdk import WorkspaceClient

        async with AsyncUnityCatalogStore(
            workspace_client=WorkspaceClient(),
            catalog="main",
            schema="langgraph",
            warehouse_id="abc123"
        ) as store:
            await store.setup()  # Create tables

            # Store data
            await store.aput(("users", "123"), "prefs", b'{"theme": "dark"}')

            # Retrieve data
            item = await store.aget(("users", "123"), "prefs")
        ```

    Note:
        The store automatically handles namespacing using JSON-encoded tuples as prefixes.
    """

    lock: asyncio.Lock

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        catalog: str,
        schema: str,
        table: str = "store",
        warehouse_id: str | None = None,
    ) -> None:
        """Initialize the async Unity Catalog store.

        Args:
            workspace_client: Databricks WorkspaceClient for executing SQL.
            catalog: Unity Catalog name.
            schema: Schema name within the catalog.
            table: Table name for storing key-value pairs (default: "store").
            warehouse_id: SQL warehouse ID for executing statements (default: None).
        """
        super().__init__()
        BaseUnityCatalogStore.__init__(
            self,
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            table=table,
            warehouse_id=warehouse_id,
        )
        self.lock = asyncio.Lock()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, will be set later
            self.loop = None  # type: ignore[assignment]

        # Initialize table synchronously during construction
        self._init_table_sync()

    def _init_table_sync(self) -> None:
        """Initialize the Unity Catalog table (sync version for __init__)."""
        try:
            response: StatementResponse = (
                self.workspace_client.statement_execution.execute_statement(
                    statement=CREATE_STORE_TABLE_SQL.format(table_name=self.full_table_name),
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )
            )
            _ = response.result
            logger.info(f"Initialized Unity Catalog store table: {self.full_table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize store table: {e}")
            raise

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary table in Unity Catalog if it doesn't
        already exist. It MUST be called directly by the user the first time
        the store is used.
        """
        await asyncio.to_thread(self._init_table_sync)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        Args:
            ops: Iterable of operations to execute.

        Returns:
            List of results corresponding to the operations.
        """
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
        async with self.lock:
            if GetOp in grouped:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped[GetOp]), results
                )

            if PutOp in grouped:
                await self._batch_put_ops(cast(Sequence[tuple[int, PutOp]], grouped[PutOp]))

            if ListNamespacesOp in grouped:
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped[ListNamespacesOp],
                    ),
                    results,
                )

        return results

    async def aget(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None:
        """Get a single value from the store asynchronously.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key within the namespace

        Returns:
            Value dict if found, None otherwise
        """
        results = await self.abatch([GetOp(namespace=namespace, key=key)])
        if results and isinstance(results[0], Item):
            return results[0].value
        return None

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Store a single value asynchronously.

        Args:
            namespace: Hierarchical namespace tuple
            key: Key within the namespace
            value: Value to store as a dictionary
        """
        # Serialize dict value to bytes
        value_bytes = json.dumps(value).encode("utf-8")

        await self.abatch([PutOp(namespace=namespace, key=key, value=value_bytes)])

    async def _batch_get_ops(
        self, get_ops: Sequence[tuple[int, GetOp]], results: list[Result]
    ) -> None:
        """Execute a batch of get operations.

        Args:
            get_ops: Sequence of (index, GetOp) tuples.
            results: List to store results in.
        """
        if not get_ops:
            return

        # Build query for all items
        items = [(op.namespace, op.key) for _, op in get_ops]
        query = self._build_batch_get_query(items)

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            # Build lookup map
            lookup: dict[tuple[str, str], bytes] = {}
            if response.result and response.result.data_array:
                for row in response.result.data_array:
                    if row and len(row) >= 3:
                        prefix = row[0]
                        key = row[1]
                        value_hex = row[2]
                        if prefix and key and value_hex:
                            try:
                                # Decode namespace to validate, but use original prefix for lookup
                                _ = self._decode_namespace(prefix)
                                value = bytes.fromhex(value_hex)
                                lookup[(prefix, key)] = value
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to decode value for {prefix}:{key}: {e}")

            # Fill results
            for idx, op in get_ops:
                prefix_str = self._encode_namespace(op.namespace)
                value = lookup.get((prefix_str, op.key))
                if value is not None:
                    results[idx] = Item(
                        value=value,
                        key=op.key,
                        namespace=op.namespace,
                        created_at=None,
                        updated_at=None,
                    )
                else:
                    results[idx] = None

        except Exception as e:
            logger.error(f"Failed to batch get: {e}")
            # Fill with None on error
            for idx, _ in get_ops:
                results[idx] = None

    async def _batch_put_ops(self, put_ops: Sequence[tuple[int, PutOp]]) -> None:
        """Execute a batch of put operations.

        Args:
            put_ops: Sequence of (index, PutOp) tuples.
        """
        if not put_ops:
            return

        # Build query for all items
        items = [(op.namespace, op.key, op.value) for _, op in put_ops]
        query = self._build_put_query(items)

        try:
            await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )
            logger.debug(f"Batch put {len(put_ops)} items")

        except Exception as e:
            logger.error(f"Failed to batch put: {e}")
            raise

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        """Execute a batch of list namespaces operations.

        Args:
            list_ops: Sequence of (index, ListNamespacesOp) tuples.
            results: List to store results in.
        """
        for idx, op in list_ops:
            query = self._build_list_namespaces_query(op.prefix)

            try:
                response: StatementResponse = await asyncio.to_thread(
                    self.workspace_client.statement_execution.execute_statement,
                    statement=query,
                    warehouse_id=self.warehouse_id,
                    wait_timeout="30s",
                )

                namespaces = []
                if response.result and response.result.data_array:
                    for row in response.result.data_array:
                        if row and row[0]:
                            try:
                                namespace = self._decode_namespace(row[0])
                                namespaces.append(namespace)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to decode namespace {row[0]}: {e}")

                results[idx] = namespaces

            except Exception as e:
                logger.error(f"Failed to list namespaces: {e}")
                results[idx] = []

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from the store asynchronously.

        Args:
            namespace: The namespace tuple.
            key: The key to delete.
        """
        query = self._build_delete_query([(namespace, key)])

        try:
            await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )
            logger.debug(f"Deleted {namespace}:{key}")

        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            raise

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002
        limit: int = 10,
        offset: int = 0,
        query: str | None = None,
    ) -> list[Item]:
        """Search for items in the store within a namespace asynchronously.

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
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=sql_query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
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

    async def alist_namespaces(
        self, *, prefix: tuple[str, ...] | None = None
    ) -> AsyncIterator[tuple[str, ...]]:
        """List all namespaces in the store asynchronously.

        Args:
            prefix: Optional namespace prefix to filter by.

        Yields:
            Namespace tuples.
        """
        query = self._build_list_namespaces_query(prefix)

        try:
            response: StatementResponse = await asyncio.to_thread(
                self.workspace_client.statement_execution.execute_statement,
                statement=query,
                warehouse_id=self.warehouse_id,
                wait_timeout="30s",
            )

            if response.result and response.result.data_array:
                for row in response.result.data_array:
                    if row and row[0]:
                        try:
                            namespace = self._decode_namespace(row[0])
                            yield namespace
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to decode namespace {row[0]}: {e}")

        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            return

    async def __aenter__(self) -> AsyncUnityCatalogStore:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:  # noqa: ANN401
        """Exit async context."""
        pass


__all__ = ["AsyncUnityCatalogStore"]
