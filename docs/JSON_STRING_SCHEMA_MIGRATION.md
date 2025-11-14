# JSON String Schema Migration

## Overview

**Date:** November 14, 2025  
**Change:** Migrated from BINARY columns to STRING (JSON) columns for all checkpoint data storage

## Why This Change?

The user reported that conversation history was still not working after the initial PostgreSQL-pattern fix. To simplify debugging and eliminate potential binary encoding/decoding issues with Unity Catalog, we switched from BINARY storage to JSON string storage.

### Benefits

1. **Easier Debugging**: Data can be queried directly in SQL without hex/base64 decoding
2. **Simplified Code**: Less encoding/decoding complexity
3. **Better Compatibility**: Delta Lake handles JSON strings natively
4. **Transparency**: You can see exactly what's stored in the database
5. **Reduced Errors**: Eliminates potential binary handling differences between Unity Catalog and PostgreSQL

## Schema Changes

### Before (BINARY)

```sql
CREATE TABLE checkpoints (
    ...
    checkpoint BINARY NOT NULL,
    ...
);

CREATE TABLE checkpoint_blobs (
    ...
    blob BINARY,
    ...
);

CREATE TABLE checkpoint_writes (
    ...
    blob BINARY NOT NULL,
    ...
);
```

### After (JSON Strings)

```sql
CREATE TABLE checkpoints (
    ...
    checkpoint STRING NOT NULL,  -- Base64-encoded serialized checkpoint
    ...
);

CREATE TABLE checkpoint_blobs (
    ...
    blob STRING,  -- Base64-encoded serialized blob
    ...
);

CREATE TABLE checkpoint_writes (
    ...
    blob STRING NOT NULL,  -- Base64-encoded serialized write
    ...
);
```

## Implementation Details

### Storage Format

All data is now stored as **base64-encoded strings**:

```python
# Storing
checkpoint_data = serde.dumps_typed(checkpoint)  # Returns bytes
checkpoint_str = base64.b64encode(checkpoint_data[1]).decode('utf-8')  # Convert to string
# Store checkpoint_str in STRING column

# Loading
checkpoint_str = row['checkpoint']  # Get string from database
checkpoint_bytes = base64.b64decode(checkpoint_str)  # Convert back to bytes
checkpoint = serde.loads_typed((type_, checkpoint_bytes))  # Deserialize
```

###Files Modified

1. **`src/langgraph_unity_catalog_checkpoint/checkpoint/base.py`**
   - Updated all CREATE TABLE statements to use STRING instead of BINARY
   - Updated table descriptions to note "(JSON strings)"

2. **`src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py`**
   - `_upsert_checkpoint`: Changed from `X'{hex}'` to `'{base64_string}'`
   - `_upsert_blobs_batch`: Changed from `X'{hex}'` to `'{base64_string}'`
   - `_upsert_write`: Changed from `X'{hex}'` to `'{base64_string}'`
   - `_upsert_writes_batch`: Changed from `X'{hex}'` to `'{base64_string}'`
   - `_load_checkpoint_tuple`: Changed to decode base64 strings
   - `_load_channel_values_async`: Already handled base64 strings correctly
   - `_load_pending_writes_async`: Already handled base64 strings correctly

3. **`src/langgraph_unity_catalog_checkpoint/checkpoint/shallow.py`**
   - Same changes as aio.py for sync methods

## Migration Path

### For New Users

No action needed! Just install and use:

```bash
pip install langgraph-unity-catalog-checkpoint
```

The tables will be created with the new schema automatically.

### For Existing Users

If you have existing checkpoints in BINARY format, you have two options:

#### Option 1: Fresh Start (Recommended)

Drop and recreate the tables:

```sql
-- In your Databricks workspace
DROP TABLE IF EXISTS <catalog>.<schema>.checkpoints;
DROP TABLE IF EXISTS <catalog>.<schema>.checkpoint_blobs;
DROP TABLE IF EXISTS <catalog>.<schema>.checkpoint_writes;
```

Then create a new checkpointer - tables will be recreated with the new schema:

```python
from langgraph_unity_catalog_checkpoint import AsyncUnityCatalogCheckpointSaver

checkpointer = AsyncUnityCatalogCheckpointSaver(
    workspace_client=workspace_client,
    catalog="your_catalog",
    schema="your_schema",
    warehouse_id="your_warehouse_id",
)
```

#### Option 2: Manual Migration (Advanced)

If you need to preserve existing checkpoints:

```sql
-- 1. Rename old tables
ALTER TABLE <catalog>.<schema>.checkpoints RENAME TO checkpoints_old;
ALTER TABLE <catalog>.<schema>.checkpoint_blobs RENAME TO checkpoint_blobs_old;
ALTER TABLE <catalog>.<schema>.checkpoint_writes RENAME TO checkpoint_writes_old;

-- 2. Create new tables (will happen automatically on first use)
-- Just initialize a checkpointer and it will create the new schema

-- 3. Migrate data (example for checkpoints table)
INSERT INTO <catalog>.<schema>.checkpoints
SELECT 
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    type,
    base64(checkpoint) AS checkpoint,  -- Convert BINARY to base64 STRING
    metadata,
    created_at
FROM <catalog>.<schema>.checkpoints_old;

-- Repeat for other tables...

-- 4. Drop old tables after verification
DROP TABLE <catalog>.<schema>.checkpoints_old;
DROP TABLE <catalog>.<schema>.checkpoint_blobs_old;
DROP TABLE <catalog>.<schema>.checkpoint_writes_old;
```

## Testing

All unit tests continue to pass with the new schema:

```bash
uv run pytest tests/test_unity_catalog_checkpointer.py tests/test_async_unity_catalog_checkpointer.py -v
# Result: 22 passed, 4 skipped ✅
```

## Debugging Benefits

### Before (BINARY)

```sql
SELECT checkpoint FROM checkpoints WHERE thread_id = 'test_001';
-- Result: <binary data - not human readable>
```

### After (JSON String)

```sql
SELECT checkpoint FROM checkpoints WHERE thread_id = 'test_001';
-- Result: "H4sIAAAAAAAA..." (base64 string - can be decoded)

-- To inspect (in Python):
import base64
import msgpack  # or json depending on serde

data = base64.b64decode("H4sIAAAAAAAA...")
checkpoint = msgpack.unpackb(data)
print(checkpoint)  # Human-readable dict!
```

## Performance Considerations

### Storage Size

- **BINARY**: Stores raw bytes directly
- **STRING (base64)**: ~33% larger due to base64 encoding overhead

For most use cases, this overhead is negligible compared to the benefits of easier debugging and better compatibility.

### Query Performance

No significant difference - both are indexed on the same keys (thread_id, checkpoint_ns, checkpoint_id).

## Backward Compatibility

**BREAKING CHANGE**: This is a schema change that requires table recreation or migration.

- **Version**: 0.0.1 → 0.0.2 (when released)
- **Impact**: Existing checkpoints in BINARY format are not automatically readable
- **Mitigation**: See migration options above

## FAQ

### Q: Why not use JSON directly instead of base64?

**A:** The serialized data contains binary structures (from msgpack) that can't be directly stored as JSON strings. Base64 encoding is necessary to convert bytes to strings.

### Q: Can I query the checkpoint data directly?

**A:** Yes, but you'll need to decode the base64 string first. Example:

```python
# In Databricks notebook
from pyspark.sql.functions import base64, unbase64

df = spark.sql("SELECT checkpoint FROM my_catalog.my_schema.checkpoints")
df.select(unbase64(df.checkpoint)).show()
```

### Q: Will this affect performance?

**A:** The performance impact is minimal. Base64 encoding/decoding is fast, and the ~33% storage overhead is usually negligible.

### Q: Is this change permanent?

**A:** Yes. The STRING schema is simpler, more debuggable, and better suited for Unity Catalog/Delta Lake than BINARY columns.

## Related Documents

- `docs/CONVERSATION_HISTORY_FIX_FINAL.md` - Original conversation history fix
- `docs/CONVERSATION_HISTORY_SOLUTION_SUMMARY.md` - Solution summary
- `src/langgraph_unity_catalog_checkpoint/checkpoint/base.py` - Table schemas

## Status

✅ **IMPLEMENTED AND TESTED**  
✅ All unit tests passing  
✅ Ready for testing with live Databricks credentials  

---

**Questions or Issues?**  
Please check the table schemas in `base.py` or open an issue on GitHub.

