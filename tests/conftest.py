"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from databricks.sdk import WorkspaceClient
from dotenv import find_dotenv, load_dotenv

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    print(f"✓ Loaded environment from {env_file}")
else:
    print("ℹ No .env file found")


def has_warehouse_id() -> bool:
    """Check if Databricks SQL warehouse ID is configured.

    Returns:
        bool: True if DATABRICKS_SQL_WAREHOUSE_ID environment variable is set.
    """
    return bool(os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring real Databricks connection",
    )


# Skip integration tests if warehouse ID is not configured
skip_if_no_warehouse = pytest.mark.skipif(
    not has_warehouse_id(),
    reason="DATABRICKS_SQL_WAREHOUSE_ID environment variable not set",
)


@pytest.fixture
def mock_workspace_client() -> Mock:
    """Create a mock WorkspaceClient for testing."""
    client = Mock(spec=WorkspaceClient)

    # Mock statement execution
    mock_statement_execution = Mock()
    mock_result = Mock()
    mock_result.result = Mock()
    mock_result.result.data_array = []

    mock_statement_execution.execute_statement = Mock(return_value=mock_result)
    client.statement_execution = mock_statement_execution

    return client


@pytest.fixture
def warehouse_id() -> str:
    """Test warehouse ID."""
    return "test_warehouse_123"


@pytest.fixture
def catalog() -> str:
    """Test catalog name."""
    return "test_catalog"


@pytest.fixture
def schema() -> str:
    """Test schema name."""
    return "test_schema"


@pytest.fixture
def store_config(catalog: str, schema: str, warehouse_id: str) -> dict[str, str]:
    """Configuration for store tests.

    Uses PostgreSQL/LangGraph naming convention: "store"
    """
    return {
        "catalog": catalog,
        "schema": schema,
        "table": "store",  # PostgreSQL naming convention
        "warehouse_id": warehouse_id,
    }


@pytest.fixture
def checkpointer_config(catalog: str, schema: str, warehouse_id: str) -> dict[str, str]:
    """Configuration for checkpointer tests.

    Uses PostgreSQL/LangGraph naming conventions:
    - checkpoints_table: "checkpoints"
    - writes_table: "checkpoint_writes"
    """
    return {
        "catalog": catalog,
        "schema": schema,
        "checkpoints_table": "checkpoints",  # PostgreSQL naming convention
        "writes_table": "checkpoint_writes",  # PostgreSQL naming convention
        "warehouse_id": warehouse_id,
    }


@pytest.fixture
def mock_execute_result() -> Mock:
    """Create a mock result for statement execution."""
    result = Mock()
    result.result = Mock()
    result.result.data_array = []
    return result


def create_mock_result(data: list[list[Any]]) -> Mock:
    """Helper to create mock result with data.

    Args:
        data: List of rows, where each row is a list of column values

    Returns:
        Mock result object
    """
    result = Mock()
    result.result = Mock()
    result.result.data_array = data
    return result
