"""Comprehensive integration test for conversation history persistence.

This test validates that conversation history is properly preserved across
multiple interactions, following the user's requirement:
1. Ask "What is the capital of France?"
2. Ask "What was the last question I asked you?"
3. Verify the model remembers the first question
"""

import asyncio
import os
from typing import Annotated

import pytest
from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_unity_catalog_checkpoint import (
    AsyncUnityCatalogCheckpointSaver,
    UnityCatalogCheckpointSaver,
)
from tests.conftest import skip_if_no_warehouse


class State(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]


def simple_chatbot(state: State) -> dict:
    """Simple chatbot that echoes back with conversation context."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Create a response that demonstrates conversation history
    if "capital" in last_message.lower() and "france" in last_message.lower():
        response = "The capital of France is Paris."
    elif "last question" in last_message.lower() or "what did i" in last_message.lower():
        # This should have access to previous messages
        if len(messages) > 1:
            first_question = messages[0].content if isinstance(messages[0], HumanMessage) else "unknown"
            response = f"Your last question was: '{first_question}'"
        else:
            response = "I don't have any previous questions in our conversation history."
    else:
        response = f"I received your message: {last_message}"
    
    return {"messages": [AIMessage(content=response)]}


async def async_chatbot(state: State) -> dict:
    """Async version of simple chatbot."""
    return simple_chatbot(state)


@skip_if_no_warehouse
class TestConversationHistoryIntegrationSync:
    """Integration test for conversation history with sync checkpointer."""

    def test_conversation_history_preserved(self) -> None:
        """Test that conversation history is preserved - SYNC version."""
        workspace_client = WorkspaceClient()
        
        catalog = os.getenv("UC_CATALOG", "main")
        schema = os.getenv("UC_SCHEMA", "langgraph_test")
        warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
        
        if not warehouse_id:
            pytest.skip("DATABRICKS_SQL_WAREHOUSE_ID not set")
        
        print(f"\n{'='*80}")
        print(f"Testing SYNC Conversation History")
        print(f"{'='*80}")
        print(f"Catalog: {catalog}, Schema: {schema}")
        
        # Drop and recreate tables for clean test
        checkpointer = UnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            warehouse_id=warehouse_id,
        )
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", simple_chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "test_sync_conversation_001"}}
        
        # FIRST INTERACTION: Ask about capital of France
        print(f"\n{'='*80}")
        print("FIRST INTERACTION")
        print(f"{'='*80}")
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="What is the capital of France?")]},
            config=config,
        )
        
        print(f"Q: What is the capital of France?")
        print(f"A: {result1['messages'][-1].content}")
        print(f"Total messages after first interaction: {len(result1['messages'])}")
        
        # Verify first interaction
        assert len(result1["messages"]) == 2, f"Expected 2 messages, got {len(result1['messages'])}"
        assert "Paris" in result1["messages"][-1].content, "Should mention Paris"
        
        # SECOND INTERACTION: Ask what the last question was
        print(f"\n{'='*80}")
        print("SECOND INTERACTION")
        print(f"{'='*80}")
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="What was the last question I asked you?")]},
            config=config,
        )
        
        print(f"Q: What was the last question I asked you?")
        print(f"A: {result2['messages'][-1].content}")
        print(f"Total messages after second interaction: {len(result2['messages'])}")
        
        # Verify second interaction
        assert len(result2["messages"]) == 4, (
            f"Expected 4 messages (2 from first + 2 from second), got {len(result2['messages'])}"
        )
        
        # The critical test: Does the response reference the first question?
        response = result2["messages"][-1].content.lower()
        assert "capital" in response or "france" in response, (
            f"Response should reference the first question about France's capital. "
            f"Got: {result2['messages'][-1].content}"
        )
        
        print(f"\n{'='*80}")
        print("✓ SYNC TEST PASSED - Conversation history is preserved!")
        print(f"{'='*80}\n")


@skip_if_no_warehouse
class TestConversationHistoryIntegrationAsync:
    """Integration test for conversation history with async checkpointer."""

    @pytest.mark.asyncio
    async def test_conversation_history_preserved_async(self) -> None:
        """Test that conversation history is preserved - ASYNC version."""
        workspace_client = WorkspaceClient()
        
        catalog = os.getenv("UC_CATALOG", "main")
        schema = os.getenv("UC_SCHEMA", "langgraph_test")
        warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
        
        if not warehouse_id:
            pytest.skip("DATABRICKS_SQL_WAREHOUSE_ID not set")
        
        print(f"\n{'='*80}")
        print(f"Testing ASYNC Conversation History")
        print(f"{'='*80}")
        print(f"Catalog: {catalog}, Schema: {schema}")
        
        # Create async checkpointer
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=catalog,
            schema=schema,
            warehouse_id=warehouse_id,
        )
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", async_chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "test_async_conversation_001"}}
        
        # FIRST INTERACTION: Ask about capital of France
        print(f"\n{'='*80}")
        print("FIRST INTERACTION")
        print(f"{'='*80}")
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is the capital of France?")]},
            config=config,
        )
        
        print(f"Q: What is the capital of France?")
        print(f"A: {result1['messages'][-1].content}")
        print(f"Total messages after first interaction: {len(result1['messages'])}")
        
        # Verify first interaction
        assert len(result1["messages"]) == 2, f"Expected 2 messages, got {len(result1['messages'])}"
        assert "Paris" in result1["messages"][-1].content, "Should mention Paris"
        
        # SECOND INTERACTION: Ask what the last question was
        print(f"\n{'='*80}")
        print("SECOND INTERACTION")
        print(f"{'='*80}")
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="What was the last question I asked you?")]},
            config=config,
        )
        
        print(f"Q: What was the last question I asked you?")
        print(f"A: {result2['messages'][-1].content}")
        print(f"Total messages after second interaction: {len(result2['messages'])}")
        
        # Print all messages for debugging
        print(f"\nAll messages in conversation:")
        for i, msg in enumerate(result2["messages"]):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"  {i}: {msg_type}: {msg.content[:80]}...")
        
        # Verify second interaction
        assert len(result2["messages"]) == 4, (
            f"Expected 4 messages (2 from first + 2 from second), got {len(result2['messages'])}"
        )
        
        # The critical test: Does the response reference the first question?
        response = result2["messages"][-1].content.lower()
        assert "capital" in response or "france" in response, (
            f"Response should reference the first question about France's capital. "
            f"Got: {result2['messages'][-1].content}"
        )
        
        # Additional verification: Check state directly
        state = await graph.aget_state(config)
        assert len(state.values["messages"]) == 4, (
            f"State should have 4 messages, got {len(state.values['messages'])}"
        )
        
        print(f"\n{'='*80}")
        print("✓ ASYNC TEST PASSED - Conversation history is preserved!")
        print(f"{'='*80}\n")

