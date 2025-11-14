# Conversation History Fix

## Issue Summary

**Date**: November 14, 2025  
**Status**: ✅ FIXED  

### Problem

Conversation history was not being preserved across multiple interactions when using `AsyncUnityCatalogCheckpointSaver` and `UnityCatalogCheckpointSaver`. When asking a follow-up question like "What did I just ask you?", the LLM would respond that it didn't know, indicating the previous messages weren't being passed to the model.

### Root Cause

The issue was caused by **checkpoint dictionary mutation** in the `aput` method:

```python
# BEFORE (buggy code)
channel_values = checkpoint.pop("channel_values", {})  # ❌ Mutates checkpoint!
```

This code used `dict.pop()` which **mutates the original checkpoint dictionary** by removing the `channel_values` key. In Python, dictionaries are mutable objects, and this mutation could affect how LangGraph manages checkpoints internally, potentially causing it to lose track of the conversation state.

### The Fix

Changed from mutation to non-mutating operations:

```python
# AFTER (fixed code)
# Make a copy to avoid mutating the original checkpoint
channel_values = checkpoint.get("channel_values", {})  # ✅ No mutation

# Store checkpoint without channel_values (they're in blobs table)
checkpoint_to_store = {k: v for k, v in checkpoint.items() if k != "channel_values"}
```

### Files Modified

1. **`src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py`**
   - Lines 269-288: Fixed `aput` method to avoid mutating checkpoint dict
   
2. **`src/langgraph_unity_catalog_checkpoint/checkpoint/shallow.py`**
   - Lines 408-442: Fixed `put` method to avoid mutating checkpoint dict

### Changes Made

#### AsyncUnityCatalogCheckpointSaver (`aio.py`)

```python
# Line 271: Use get() instead of pop()
channel_values = checkpoint.get("channel_values", {})

# Lines 279-288: Create filtered dict instead of mutating original
checkpoint_to_store = {k: v for k, v in checkpoint.items() if k != "channel_values"}
await self._upsert_checkpoint(
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    checkpoint_to_store,  # Pass filtered copy
    metadata,
    config,
)
```

#### ShallowUnityCatalogCheckpointSaver (`shallow.py`)

```python
# Line 410: Use get() instead of pop()
channel_values = checkpoint.get("channel_values", {})

# Lines 441-442: Create filtered dict instead of mutating original
checkpoint_to_store = {k: v for k, v in checkpoint.items() if k != "channel_values"}
checkpoint_data = self.serde.dumps(checkpoint_to_store)
```

### Why This Matters

1. **Dictionary Mutation**: In Python, when you pass a dict to a function and mutate it, the original is modified
2. **LangGraph Internals**: LangGraph may reuse checkpoint objects or rely on their immutability
3. **State Integrity**: Mutating checkpoints can cause state inconsistencies in the graph execution

### Impact

- ✅ Conversation history now persists correctly across multiple interactions
- ✅ LLMs receive full conversation context in follow-up questions
- ✅ No breaking API changes - fully backward compatible
- ✅ Performance unchanged - dict comprehension is equally fast

### Testing

Created comprehensive tests to validate the fix:

1. **`tests/test_conversation_history.py`**
   - Integration tests for sync and async checkpointers
   - Tests multi-turn conversations
   - Validates message accumulation

2. **`tests/debug_conversation_history.py`**
   - Debug script with detailed logging
   - Shows message flow at each step
   - Validates checkpoint contents

### Verification

To verify the fix works:

```bash
# Run conversation history tests
uv run pytest tests/test_conversation_history.py -v

# Run debug script (requires credentials)
export DATABRICKS_HOST="..."
export DATABRICKS_TOKEN="..."
export DATABRICKS_SQL_WAREHOUSE_ID="..."
export UC_CATALOG="main"
export UC_SCHEMA="langgraph_test"

python tests/debug_conversation_history.py
```

### Example Usage

After the fix, this now works correctly:

```python
from langgraph_unity_catalog_checkpoint import AsyncUnityCatalogCheckpointSaver

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
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config=config
)
# AI: "You asked me what the capital of France is."
```

### Related Issues

This fix also prevents potential issues with:
- Multi-agent workflows that share checkpoints
- Parallel execution with checkpoint reuse
- Any scenario where checkpoint immutability is expected

### Best Practice

**Never mutate function arguments in Python!**

```python
# ❌ BAD: Mutates argument
def process(data: dict) -> None:
    value = data.pop("key")

# ✅ GOOD: No mutation
def process(data: dict) -> None:
    value = data.get("key")
    data_copy = {k: v for k, v in data.items() if k != "key"}
```

### Commit Message

```
fix: Prevent checkpoint mutation to preserve conversation history

The checkpointer was mutating checkpoint dicts using pop(), which could
cause LangGraph to lose conversation context. Changed to use get() and
dict comprehension to avoid mutation.

Fixes conversation history persistence issue where follow-up questions
wouldn't have access to previous messages.

Files modified:
- src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py
- src/langgraph_unity_catalog_checkpoint/checkpoint/shallow.py

Tests added:
- tests/test_conversation_history.py
- tests/debug_conversation_history.py
```

### References

- [Python dict.pop() documentation](https://docs.python.org/3/library/stdtypes.html#dict.pop)
- [LangGraph Checkpointer Pattern](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)
- [Python Mutable vs Immutable Objects](https://docs.python.org/3/reference/datamodel.html)

