"""Integration test for conversation history persistence."""

import os
from typing import Annotated

import pytest
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
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


def has_llm_endpoint() -> bool:
    """Check if Databricks LLM endpoint is configured."""
    return bool(os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"))


skip_if_no_llm = pytest.mark.skipif(
    not has_llm_endpoint(),
    reason="DATABRICKS_HOST and DATABRICKS_TOKEN environment variables not set",
)


@pytest.fixture
def test_catalog() -> str:
    """Test catalog name."""
    return os.getenv("UC_CATALOG", "main")


@pytest.fixture
def test_schema() -> str:
    """Test schema name."""
    return os.getenv("UC_SCHEMA", "langgraph_test")


@pytest.fixture
def warehouse_id_env() -> str:
    """Test warehouse ID from environment."""
    warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
    if not warehouse_id:
        pytest.skip("DATABRICKS_SQL_WAREHOUSE_ID environment variable not set")
    return warehouse_id


@skip_if_no_warehouse
@skip_if_no_llm
class TestConversationHistorySync:
    """Test conversation history persistence with sync checkpointer."""

    def test_conversation_history_restored(
        self, test_catalog: str, test_schema: str, warehouse_id_env: str
    ) -> None:
        """Test that conversation history is properly restored across interactions."""
        workspace_client = WorkspaceClient()
        
        # Create checkpointer
        checkpointer = UnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=test_catalog,
            schema=test_schema,
            warehouse_id=warehouse_id_env,
        )
        
        # Create a simple echo bot for testing (no LLM required)
        def echo_bot(state: State) -> dict:
            """Echo the last message with context info."""
            messages = state["messages"]
            last_msg = messages[-1].content if messages else "No messages"
            
            # Create response that shows we have conversation history
            context = f"I see {len(messages)} message(s) in history. "
            context += f"Your last question was: '{last_msg}'"
            
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=context)]}
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", echo_bot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        # Test conversation
        config = {"configurable": {"thread_id": "test_conversation_1"}}
        
        # First interaction
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="What is the capital of France?")]},
            config=config,
        )
        
        # Verify first response
        assert len(result1["messages"]) == 2  # Human + AI
        assert "capital of France" in result1["messages"][-1].content
        
        # Second interaction - ask about previous question
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="What did I just ask you?")]},
            config=config,
        )
        
        # Verify conversation history is preserved
        assert len(result2["messages"]) == 4  # 2 from first + 2 from second
        # The response should reference the previous question
        response_content = result2["messages"][-1].content.lower()
        assert "4 message" in response_content or "capital" in response_content, (
            f"Response should reference conversation history: {response_content}"
        )
        
        print(f"\n✓ Sync conversation history test passed")
        print(f"  First interaction: {result1['messages'][-1].content}")
        print(f"  Second interaction: {result2['messages'][-1].content}")


@skip_if_no_warehouse
@skip_if_no_llm
class TestConversationHistoryAsync:
    """Test conversation history persistence with async checkpointer."""

    @pytest.mark.asyncio
    async def test_conversation_history_restored_async(
        self, test_catalog: str, test_schema: str, warehouse_id_env: str
    ) -> None:
        """Test that conversation history is properly restored across async interactions."""
        workspace_client = WorkspaceClient()
        
        # Create async checkpointer
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=test_catalog,
            schema=test_schema,
            warehouse_id=warehouse_id_env,
        )
        
        # Create a simple echo bot for testing (no LLM required)
        async def echo_bot(state: State) -> dict:
            """Echo the last message with context info."""
            messages = state["messages"]
            last_msg = messages[-1].content if messages else "No messages"
            
            # Create response that shows we have conversation history
            context = f"I see {len(messages)} message(s) in history. "
            context += f"Your last question was: '{last_msg}'"
            
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=context)]}
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", echo_bot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        # Test conversation
        config = {"configurable": {"thread_id": "test_async_conversation_1"}}
        
        # First interaction
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is the capital of France?")]},
            config=config,
        )
        
        # Verify first response
        assert len(result1["messages"]) == 2  # Human + AI
        assert "capital of France" in result1["messages"][-1].content
        
        # Second interaction - ask about previous question
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="What did I just ask you?")]},
            config=config,
        )
        
        # Verify conversation history is preserved
        assert len(result2["messages"]) == 4  # 2 from first + 2 from second
        # The response should reference the previous question
        response_content = result2["messages"][-1].content.lower()
        assert "4 message" in response_content or "capital" in response_content, (
            f"Response should reference conversation history: {response_content}"
        )
        
        # Additional check: Get state directly from checkpointer
        state = await graph.aget_state(config)
        assert len(state.values["messages"]) == 4, (
            f"State should have 4 messages, got {len(state.values['messages'])}"
        )
        
        print(f"\n✓ Async conversation history test passed")
        print(f"  First interaction: {result1['messages'][-1].content}")
        print(f"  Second interaction: {result2['messages'][-1].content}")
        print(f"  Total messages in state: {len(state.values['messages'])}")


@skip_if_no_warehouse
@skip_if_no_llm
class TestConversationHistoryDetailed:
    """Detailed tests for conversation history behavior."""

    @pytest.mark.asyncio
    async def test_checkpoint_retrieval_sequence(
        self, test_catalog: str, test_schema: str, warehouse_id_env: str
    ) -> None:
        """Test the sequence of checkpoint saves and retrievals."""
        workspace_client = WorkspaceClient()
        
        checkpointer = AsyncUnityCatalogCheckpointSaver(
            workspace_client=workspace_client,
            catalog=test_catalog,
            schema=test_schema,
            warehouse_id=warehouse_id_env,
        )
        
        async def simple_node(state: State) -> dict:
            """Simple node that adds a response."""
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="Response")]}
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("node", simple_node)
        graph_builder.add_edge(START, "node")
        graph_builder.add_edge("node", END)
        graph = graph_builder.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "test_sequence_1"}}
        
        # First interaction
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Message 1")]},
            config=config,
        )
        
        print(f"\nAfter first interaction:")
        print(f"  Messages in result: {len(result1['messages'])}")
        
        # Get state after first interaction
        state1 = await graph.aget_state(config)
        print(f"  Messages in state: {len(state1.values['messages'])}")
        for i, msg in enumerate(state1.values['messages']):
            print(f"    {i}: {type(msg).__name__}: {msg.content}")
        
        # Second interaction
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Message 2")]},
            config=config,
        )
        
        print(f"\nAfter second interaction:")
        print(f"  Messages in result: {len(result2['messages'])}")
        
        # Get state after second interaction
        state2 = await graph.aget_state(config)
        print(f"  Messages in state: {len(state2.values['messages'])}")
        for i, msg in enumerate(state2.values['messages']):
            print(f"    {i}: {type(msg).__name__}: {msg.content}")
        
        # Verify messages accumulate
        assert len(result2["messages"]) == 4, (
            f"Expected 4 messages after 2 interactions, got {len(result2['messages'])}"
        )
        assert result1["messages"][0].content in [m.content for m in result2["messages"]], (
            "First message should still be in conversation"
        )

