# Conversation History Investigation

## Issue Report

**Date**: November 14, 2025  
**Reporter**: User  
**Severity**: High - Core functionality affected

### Problem Description

When using the `AsyncUnityCatalogCheckpointSaver` with LangGraph, conversation history is not being properly restored across multiple interactions. Specifically:

1. First interaction: "What is the capital of France?" - Works correctly
2. Second interaction: "What did I just ask you?" - LLM responds that it doesn't know

This indicates that the conversation context (previous messages) is not being passed to the LLM in the second interaction.

### Expected Behavior

In a properly functioning checkpointer:
1. First `graph.ainvoke()` creates a checkpoint with 2 messages (Human + AI)
2. Second `graph.ainvoke()` should:
   - Load the previous checkpoint (2 messages)
   - Merge the new message using the `add_messages` reducer (3 messages total)
   - Pass all 3 messages to the LLM
   - Add the AI response (4 messages total)
   - Save the new checkpoint with 4 messages

### Investigation Steps

#### 1. Created Test Files

-**`tests/test_conversation_history.py`**: Integration tests that validate conversation history persistence
- **`tests/debug_conversation_history.py`**: Debug script with detailed logging

#### 2. Code Review

Reviewed the async checkpointer implementation:

**Storage Flow (`aput` method)**:
1. Line 270: `channel_values = checkpoint.pop("channel_values", {})` - Removes channel values
2. Lines 271-275: Stores channel values in blobs table
3. Lines 277-286: Stores checkpoint metadata (should include `channel_versions`)

**Retrieval Flow (`aget_tuple` method)**:
1. Lines 215-240: Queries checkpoints table
2. Line 236: Calls `_load_checkpoint_tuple`
3. Lines 396-398: Deserializes checkpoint data
4. Lines 402-404: Calls `_load_channel_values_async`
5. Line 421: Merges channel_values back into checkpoint

**Channel Values Loading (`_load_channel_values_async` method)**:
1. Line 446: Gets `channel_versions` from checkpoint_data
2. Lines 447-448: Returns empty dict if no channel_versions
3. Lines 454-459: Builds query for blobs table
4. Lines 470-490: Executes query and deserializes blobs

#### 3. Potential Issues Identified

**Issue #1: Missing channel_versions**
- If `checkpoint_data` doesn't contain `channel_versions`, the method returns an empty dict
- This would cause messages to be lost

**Issue #2: checkpoint.pop() mutates the dict**
- Line 270 mutates the checkpoint dict
- If LangGraph reuses the same checkpoint object, this could cause issues

**Issue #3: Binary data encoding/decoding**
- Checkpoint stored as hex: `X'{checkpoint_hex}'`
- Checkpoint loaded as base64: `base64.b64decode(row[5])`
- Unity Catalog converts BINARY → base64 on SELECT, so this should work
- But worth verifying

#### 4. Debug Script

Created `tests/debug_conversation_history.py` that:
- Prints all messages at each step
- Shows what the chatbot node receives
- Displays checkpoint contents
- Validates message counts

### Next Steps

1. **Run the debug script** with warehouse credentials:
   ```bash
   export DATABRICKS_HOST="..."
   export DATABRICKS_TOKEN="..."
   export DATABRICKS_SQL_WAREHOUSE_ID="..."
   export UC_CATALOG="main"
   export UC_SCHEMA="langgraph_test"
   
   python tests/debug_conversation_history.py
   ```

2. **Check the output** for:
   - Does the chatbot node receive 1 or 3 messages in the second interaction?
   - Does the checkpoint have `channel_versions`?
   - Are channel values being loaded from blobs?

3. **Add diagnostic logging** to the checkpointer:
   - Log checkpoint_data keys in `_load_channel_values_async`
   - Log query results from blobs table
   - Log final channel_values before returning

4. **Verify binary handling**:
   - Check if Unity Catalog returns BINARY as base64 or hex
   - Verify the checkpoint can be deserialized correctly

### Possible Fixes

**If channel_versions is missing**:
- Investigate why `channel_versions` isn't in the checkpoint
- Check if LangGraph is creating checkpoints correctly
- Verify serialization preserves `channel_versions`

**If blobs aren't being loaded**:
- Check if blobs are being written to the table
- Verify the query conditions match blob records
- Check blob deserialization

**If it's a mutation issue**:
- Copy the checkpoint before popping: `channel_values = checkpoint.copy().pop(...)`
- Or restore after: `checkpoint["channel_values"] = channel_values` (but this defeats the purpose)

### Test Commands

```bash
# Run integration tests (requires credentials)
uv run pytest tests/test_conversation_history.py -v -s

# Run debug script
uv run python tests/debug_conversation_history.py

# Run with detailed async checkpointer test
uv run pytest tests/test_conversation_history.py::TestConversationHistoryAsync::test_conversation_history_restored_async -v -s
```

### Related Files

- `src/langgraph_unity_catalog_checkpoint/checkpoint/aio.py` - Async checkpointer implementation
- `src/langgraph_unity_catalog_checkpoint/checkpoint/unity_catalog.py` - Sync checkpointer (delegates to async)
- `src/langgraph_unity_catalog_checkpoint/checkpoint/base.py` - Base class with SQL templates
- `notebooks/async_checkpointer_example.ipynb` - Example where issue was observed

### References

- [LangGraph Checkpoint Pattern](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)
- [BaseCheckpointSaver Interface](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py)

## Status

**Current**: Investigation in progress  
**Blocking**: Requires warehouse credentials to run debug script  
**Priority**: High - affects core conversation functionality

