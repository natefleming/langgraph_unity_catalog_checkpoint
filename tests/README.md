# Running Tests

This directory contains both unit tests and integration tests for the checkpoint-unity-catalog package.

## Test Types

### Unit Tests
- **Files**: `test_unity_catalog_store.py`, `test_unity_catalog_checkpointer.py`, `test_async_unity_catalog_checkpointer.py`
- **Purpose**: Test individual components with mocked dependencies
- **Requirements**: No real Databricks connection needed
- **Speed**: Fast (<1 second per test)

### Integration Tests
- **File**: `test_integration.py`
- **Purpose**: Test against real Databricks Unity Catalog
- **Requirements**:
  - Active Databricks workspace
  - Valid authentication (via environment variables or Databricks CLI)
  - `DATABRICKS_SQL_WAREHOUSE_ID` environment variable set
- **Speed**: Slower (2-5 seconds per test, requires network calls)
- **Cleanup**: Tests automatically drop created tables before and after execution

## Running Tests

### Run All Unit Tests (Fast)
```bash
# Skip integration tests
pytest tests/ -v -m "not integration"

# Or run specific test files
pytest tests/test_unity_catalog_store.py -v
pytest tests/test_unity_catalog_checkpointer.py -v
```

### Run Integration Tests
```bash
# Set environment variable for your warehouse
export DATABRICKS_SQL_WAREHOUSE_ID="your-warehouse-id"

# Optionally set catalog and schema (defaults to main.default)
export DATABRICKS_CATALOG="main"
export DATABRICKS_SCHEMA="default"

# Run integration tests
pytest tests/test_integration.py -v

# Or run specific integration test
pytest tests/test_integration.py::TestStoreIntegration::test_store_mset_mget -v
```

### Run All Tests (Unit + Integration)
```bash
# Set warehouse ID first
export DATABRICKS_SQL_WAREHOUSE_ID="your-warehouse-id"

# Run everything
pytest tests/ -v
```

### Skip Integration Tests Automatically
Integration tests are automatically skipped if `DATABRICKS_SQL_WAREHOUSE_ID` is not set:

```bash
# Without warehouse ID - integration tests are skipped
pytest tests/ -v
# Output: 8 skipped (integration), XX passed (unit)
```

## Test Configuration

### Environment Variables
- `DATABRICKS_SQL_WAREHOUSE_ID`: **Required** for integration tests
- `DATABRICKS_CATALOG`: Optional, defaults to `"main"`
- `DATABRICKS_SCHEMA`: Optional, defaults to `"default"`
- Standard Databricks SDK auth variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, etc.)

### .env File Support
Tests automatically load `.env` file if present. Create `.env` in project root:

```bash
# .env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-token
DATABRICKS_SQL_WAREHOUSE_ID=abc123...
DATABRICKS_CATALOG=main
DATABRICKS_SCHEMA=default
```

## Test Markers

### Available Markers
- `@pytest.mark.integration`: Marks tests as integration tests
- `@pytest.mark.asyncio`: Marks tests as async tests

### Using Markers
```bash
# Run only integration tests
pytest tests/ -v -m integration

# Run only unit tests (exclude integration)
pytest tests/ -v -m "not integration"

# Run only async tests
pytest tests/ -v -m asyncio
```

## Integration Test Details

### Tables Created
Integration tests create and clean up the following tables:
- `{catalog}.{schema}.integration_test_store`
- `{catalog}.{schema}.integration_test_checkpoints`
- `{catalog}.{schema}.checkpoint_blobs`
- `{catalog}.{schema}.integration_test_writes`
- `{catalog}.{schema}.integration_test_async_checkpoints`
- `{catalog}.{schema}.integration_test_async_writes`

### Cleanup Behavior
- **Before each test**: Drops test tables if they exist
- **After each test**: Drops test tables
- **On failure**: Tables are still cleaned up via pytest fixtures

### Idempotency
All integration tests are idempotent and can be run multiple times:
```bash
# Run twice - should produce same results
pytest tests/test_integration.py -v
pytest tests/test_integration.py -v
```

## Continuous Integration

For CI/CD pipelines, use conditional execution:

```yaml
# GitHub Actions example
- name: Run unit tests
  run: pytest tests/ -v -m "not integration"

- name: Run integration tests
  if: ${{ secrets.DATABRICKS_SQL_WAREHOUSE_ID }}
  env:
    DATABRICKS_SQL_WAREHOUSE_ID: ${{ secrets.DATABRICKS_SQL_WAREHOUSE_ID }}
    DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
    DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
  run: pytest tests/test_integration.py -v
```

## Troubleshooting

### Integration tests are slow
- **Expected**: Integration tests make real network calls to Databricks
- **Typical duration**: 2-5 seconds per test
- **Solution**: Run unit tests during development, integration tests before commits

### Integration tests hang
- **Cause**: Databricks warehouse may be starting up or overloaded
- **Solution**: 
  - Check warehouse status in Databricks UI
  - Increase `wait_timeout` in test fixtures if needed
  - Use a Serverless SQL warehouse for faster startup

### "DATABRICKS_SQL_WAREHOUSE_ID environment variable not set"
- **Cause**: Integration tests require warehouse ID
- **Solution**: Set the environment variable or skip integration tests with `-m "not integration"`

### Authentication errors
- **Cause**: Missing or invalid Databricks credentials
- **Solution**: 
  - Run `databricks auth login` for CLI authentication
  - Or set environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
  - Or create `.env` file with credentials

## Test Coverage

Generate coverage report:
```bash
pytest tests/ --cov=src/checkpoint_unity_catalog --cov-report=html
open htmlcov/index.html
```

## Performance Profiling

Profile slow tests:
```bash
pytest tests/ -v --durations=10
```
