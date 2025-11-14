# Conversation History Fix - Final Solution

## Problem Statement

Conversation history was not being preserved across multiple LangGraph interactions using Unity Catalog checkpointers. Specifically:

**Test Scenario:**
1. Ask: "What is the capital of France?"  → AI responds: "Paris"
2. Ask: "What was the last question I asked you?" → AI responds: "I don't know" ❌

**Expected Behavior:**
The AI should remember the previous question and respond: "You asked what the capital of France is." ✅

## Root Cause Analysis

After comparing our implementation with the [LangGraph PostgreSQL reference implementation](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py), we identified critical differences:

### Issue 1: Not Copying Checkpoint Before Modification
**Problem:** We were mutating the original checkpoint dictionary
**PostgreSQL Pattern:** Makes explicit copies to avoid mutation

```python
# ❌ Our old code - mutates original
channel_values = checkpoint.pop("channel_values", {})

# ✅ PostgreSQL pattern - safe copy
copy = checkpoint.copy()
copy["channel_values"] = copy["channel_values"].copy()
```

### Issue 2: Removing ALL Channel Values from Checkpoint
**Problem:** We stored ALL channel values in blobs and removed them from checkpoint
**PostgreSQL Pattern:** Separates primitive vs complex values

**The Key Insight:**
- **Primitive values** (None, str, int, float, bool) → stored INLINE in checkpoint
- **Complex values** (lists, dicts, objects) → stored in blobs table  
- **On load:** MERGE both sources together

### Issue 3: Not Merging Channel Values on Load
**Problem:** We only loaded blob values, ignoring inline values
**PostgreSQL Pattern:** Merges inline + blob channel values

```python
# ❌ Our old code - only blobs
"channel_values": channel_values_from_blobs

# ✅ PostgreSQL pattern - merge both
"channel_values": {
    **(checkpoint_data.get("channel_values") or {}),  # Inline primitives
    **channel_values_from_blobs,                       # Blob complexes
}
```

## The Solution

### Changes Made

#### 1. AsyncUnityCatalogCheckpointSaver (`aio.py`)

**In `aput` method:**
```python
# Make a copy to avoid mutating the original checkpoint (following PostgreSQL pattern)
copy = checkpoint.copy()
copy["channel_values"] = copy["channel_values"].copy()

# Separate inline primitive values from blob values (following PostgreSQL pattern)
# Inline: None, str, int, float, bool
# Blobs: everything else (lists, dicts, objects, etc.)
channel_values = copy["channel_values"]
blob_values = {}
inline_values = {}

for k, v in channel_values.items():
    if v is None or isinstance(v, (str, int, float, bool)):
        inline_values[k] = v
    else:
        blob_values[k] = v

# Store blobs in blobs table
blob_tuples = list(
    self._dump_blobs(thread_id, checkpoint_ns, blob_values, new_versions)
)
if blob_tuples:
    await self._upsert_blobs_batch(blob_tuples)

# Store checkpoint with inline primitive values only
copy["channel_values"] = inline_values
await self._upsert_checkpoint(...)
```

**In `_load_checkpoint_tuple` method:**
```python
# Load channel values from blobs
channel_values_from_blobs = await self._load_channel_values_async(
    thread_id, checkpoint_ns, checkpoint_data
)

# Merge inline channel_values (primitives) with blob channel_values (complex objects)
# Following PostgreSQL pattern
return CheckpointTuple(
    config={...},
    checkpoint={
        **checkpoint_data,
        "channel_values": {
            **(checkpoint_data.get("channel_values") or {}),  # Inline
            **channel_values_from_blobs,                       # Blobs
        },
    },
    ...
)
```

#### 2. ShallowUnityCatalogCheckpointSaver (`shallow.py`)

Applied the same pattern to the shallow (sync) checkpointer:
- Copy checkpoint before modification
- Separate primitive/complex values
- Store primitives inline, complexes in blobs
- Merge both on load

### Files Modified

1. **`src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py`**
   - Lines 268-303: Fixed `aput` to copy checkpoint and separate primitive/complex values
   - Lines 418-444: Fixed `_load_checkpoint_tuple` to merge inline + blob channel values

2. **`src/langgraph_unity_catalog_checkpoint/checkpoint/shallow.py`**
   - Lines 408-455: Fixed `put` to copy checkpoint and separate primitive/complex values  
   - Lines 227-253: Fixed `list` to merge inline + blob channel values

3. **`tests/test_conversation_integration.py`** (NEW)
   - Comprehensive integration tests for conversation history
   - Tests both sync and async checkpointers
   - Validates the exact user scenario

## Testing

### Unit Tests
All existing unit tests continue to pass:
```bash
uv run pytest tests/test_unity_catalog_checkpointer.py tests/test_async_unity_catalog_checkpointer.py -v
# Result: 22 passed, 4 skipped ✅
```

### Integration Tests
Created comprehensive integration test that validates the exact scenario:

```bash
uv run pytest tests/test_conversation_integration.py -v -s
```

**Test Flow:**
1. First interaction: "What is the capital of France?" → Gets "Paris"
2. Second interaction: "What was the last question I asked you?" → References "capital" and "France"
3. Verification: Ensures all 4 messages are preserved (2 from each interaction)

### Manual Validation
Run the async checkpointer example notebook:
```bash
# notebooks/async_checkpointer_example.ipynb
```

The notebook now correctly maintains conversation history across cells.

## Why This Matters

### The PostgreSQL Pattern
The LangGraph PostgreSQL implementation ([source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)) uses this pattern for good reasons:

1. **Performance**: Primitive values don't need serialization overhead
2. **Efficiency**: Reduces blob storage for simple types
3. **Reliability**: Keeps critical metadata in main checkpoint table
4. **Compatibility**: Maintains checkpoint structure across storage backends

### Impact

✅ Conversation history now persists correctly  
✅ Multi-turn conversations work as expected  
✅ Human-in-the-loop workflows maintain context  
✅ Long-term memory integrations (LangMem) function properly  
✅ No breaking API changes - fully backward compatible  
✅ Performance unchanged or improved (less blob storage for primitives)  

## Verification Checklist

- [x] Unit tests pass for sync and async checkpointers
- [x] Integration tests validate conversation history
- [x] No regression in existing functionality
- [x] Follows PostgreSQL reference implementation pattern
- [x] Code is documented with clear comments
- [x] All TODO items completed

## Example Usage

After the fix, this now works correctly:

```python
from langgraph_unity_catalog_checkpoint import AsyncUnityCatalogCheckpointSaver
from langchain_core.messages import HumanMessage

checkpointer = AsyncUnityCatalogCheckpointSaver(...)
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "conversation_1"}}

# First interaction
result1 = await graph.ainvoke(
    {"messages": [HumanMessage(content="What is the capital of France?")]},
    config=config
)
# AI: "The capital of France is Paris."

# Second interaction - NOW WORKS! ✅
result2 = await graph.ainvoke(
    {"messages": [HumanMessage(content="What was the last question I asked you?")]},
    config=config
)
# AI: "You asked me what the capital of France is."
```

## References

- [LangGraph PostgreSQL AsyncPostgresSaver](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py)
- [LangGraph Checkpoint Base](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py)
- [Python Dictionary Mutability](https://docs.python.org/3/reference/datamodel.html)

## Commit Message

```
fix: Implement PostgreSQL-pattern checkpoint storage for conversation history

Followed LangGraph PostgreSQL reference implementation pattern:
- Copy checkpoints before modification (no mutation)
- Separate primitive (inline) from complex (blob) channel values
- Store primitives in checkpoint, complexes in blobs table
- Merge both sources when loading

This fixes conversation history persistence where follow-up questions
couldn't access previous messages.

Fixes: Conversation history not preserved across interactions
Reference: https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres

Files modified:
- src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py
- src/langgraph_unity_catalog_checkpoint/checkpoint/shallow.py

Tests added:
- tests/test_conversation_integration.py
```

## Status: ✅ RESOLVED

Date: November 14, 2025  
Solution verified through unit and integration tests.

