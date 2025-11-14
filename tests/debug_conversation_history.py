"""Debug script for conversation history issue."""

import asyncio
import os
from typing import Annotated

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_unity_catalog_checkpoint import AsyncUnityCatalogCheckpointSaver


class State(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]


async def debug_chatbot(state: State) -> dict:
    """Debug chatbot that echoes what it sees."""
    messages = state["messages"]
    print(f"\n[DEBUG chatbot node] Received {len(messages)} messages:")
    for i, msg in enumerate(messages):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    
    # Echo back what we received
    response = f"I see {len(messages)} messages. Last message: '{messages[-1].content}'"
    print(f"[DEBUG chatbot node] Responding: {response}")
    return {"messages": [AIMessage(content=response)]}


async def main() -> None:
    """Run the debug scenario."""
    workspace_client = WorkspaceClient()
    
    catalog = os.getenv("UC_CATALOG", "main")
    schema = os.getenv("UC_SCHEMA", "langgraph_test")
    warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
    
    print(f"Using catalog: {catalog}")
    print(f"Using schema: {schema}")
    print(f"Using warehouse_id: {warehouse_id}")
    
    # Create checkpointer
    print("\n[DEBUG] Creating checkpointer...")
    checkpointer = AsyncUnityCatalogCheckpointSaver(
        workspace_client=workspace_client,
        catalog=catalog,
        schema=schema,
        warehouse_id=warehouse_id,
    )
    
    # Build graph
    print("[DEBUG] Building graph...")
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", debug_chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    # Test conversation
    config = {"configurable": {"thread_id": "debug_conversation_1"}}
    
    # First interaction
    print("\n" + "=" * 80)
    print("FIRST INTERACTION")
    print("=" * 80)
    print("[DEBUG] Invoking with first message...")
    result1 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        config=config,
    )
    print(f"\n[DEBUG] Result 1 has {len(result1['messages'])} messages:")
    for i, msg in enumerate(result1['messages']):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    
    # Check state after first interaction
    print("\n[DEBUG] Getting state after first interaction...")
    state1 = await graph.aget_state(config)
    print(f"[DEBUG] State has {len(state1.values['messages'])} messages:")
    for i, msg in enumerate(state1.values['messages']):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    
    # Check checkpoint directly
    print("\n[DEBUG] Getting checkpoint tuple directly from checkpointer...")
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if checkpoint_tuple:
        print(f"[DEBUG] Checkpoint exists with ID: {checkpoint_tuple.config['configurable']['checkpoint_id']}")
        print(f"[DEBUG] Checkpoint has channel_values: {list(checkpoint_tuple.checkpoint.get('channel_values', {}).keys())}")
        if 'messages' in checkpoint_tuple.checkpoint.get('channel_values', {}):
            messages = checkpoint_tuple.checkpoint['channel_values']['messages']
            print(f"[DEBUG] Channel 'messages' has {len(messages)} messages")
            for i, msg in enumerate(messages):
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    else:
        print("[DEBUG] No checkpoint found!")
    
    # Second interaction
    print("\n" + "=" * 80)
    print("SECOND INTERACTION")
    print("=" * 80)
    print("[DEBUG] Invoking with second message...")
    result2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What did I just ask you?")]},
        config=config,
    )
    print(f"\n[DEBUG] Result 2 has {len(result2['messages'])} messages:")
    for i, msg in enumerate(result2['messages']):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    
    # Check state after second interaction
    print("\n[DEBUG] Getting state after second interaction...")
    state2 = await graph.aget_state(config)
    print(f"[DEBUG] State has {len(state2.values['messages'])} messages:")
    for i, msg in enumerate(state2.values['messages']):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}: {msg_type}: {msg.content[:50]}...")
    
    # Verify
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    expected_messages = 4  # 2 from first + 2 from second
    actual_messages = len(result2['messages'])
    if actual_messages == expected_messages:
        print(f"✓ SUCCESS: Got expected {expected_messages} messages")
    else:
        print(f"✗ FAILURE: Expected {expected_messages} messages, got {actual_messages}")
        print("  This indicates conversation history is NOT being preserved!")


if __name__ == "__main__":
    asyncio.run(main())

