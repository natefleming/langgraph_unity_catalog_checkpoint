# LangGraph Unity Catalog Checkpoint

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Production-ready Unity Catalog persistence for LangChain and LangGraph applications using Databricks as the storage backend.**

Following the [LangGraph checkpoint-postgres pattern](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres/langgraph) for consistency with the LangGraph ecosystem.

---

## 🚀 Overview

This package provides enterprise-grade implementations of LangGraph's persistence interfaces backed by Databricks Unity Catalog:

- **`UnityCatalogStore`** / **`AsyncUnityCatalogStore`**: Implements [`langgraph.store.base.BaseStore`](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/__init__.py) for key-value storage
- **`UnityCatalogCheckpointSaver`** / **`AsyncUnityCatalogCheckpointSaver`**: Implements [`BaseCheckpointSaver`](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py) for graph state persistence

All implementations use Databricks Unity Catalog Delta tables via the WorkspaceClient SQL API, providing:

- ✅ **Enterprise-grade reliability** with ACID transactions
- ✅ **Scalability** with Delta Lake optimization
- ✅ **Governance** with built-in access control and audit trails
- ✅ **Time-travel** for debugging and recovery
- ✅ **Seamless Databricks integration** for production ML workflows
- ✅ **Performance optimized** with batch operations (2-10x faster)

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Databricks workspace with Unity Catalog enabled
- SQL warehouse with appropriate permissions

### Install from PyPI

The easiest way to install is from PyPI:

```bash
pip install langgraph-unity-catalog-checkpoint
```

This will automatically install all required dependencies including:
- `databricks-sdk`
- `langchain`
- `langgraph`
- `loguru`

### Install from Source

For development or to get the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/natefleming/langgraph_unity_catalog_checkpoint.git
cd langgraph_unity_catalog_checkpoint

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

---

## ⚡ Quick Start

### 1. Configure Databricks Authentication

Set up environment variables:

```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-access-token"
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"
export UC_CATALOG="your_catalog"
export UC_SCHEMA="your_schema"
```

Or use `~/.databrickscfg`:

```ini
[DEFAULT]
host = https://your-workspace.databricks.com
token = your-access-token
```

### 2. Using the Store for Key-Value Storage

```python
from databricks.sdk import WorkspaceClient
from langgraph_unity_catalog_checkpoint import UnityCatalogStore

# Initialize the store
workspace_client = WorkspaceClient()
store = UnityCatalogStore(
    workspace_client=workspace_client,
    catalog="main",
    schema="langgraph",
    table="my_store",  # Default: "store"
    warehouse_id="your-warehouse-id",  # Optional
)

# Store values with namespaced keys
store.put(("users", "123"), "preferences", {"theme": "dark", "language": "en"})

# Retrieve values
prefs = store.get(("users", "123"), "preferences")
print(prefs)  # {"theme": "dark", "language": "en"}

# Search within a namespace
items = store.search(("users",), limit=10)
for item in items:
    print(f"Key: {item.key}, Namespace: {item.namespace}")

# Delete a key
store.delete(("users", "123"), "preferences")
```

### 3. Using the Checkpointer for Graph Persistence

```python
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph_unity_catalog_checkpoint import UnityCatalogCheckpointSaver

# Define your graph state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create a simple chatbot node
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Create the checkpointer
workspace_client = WorkspaceClient()
checkpointer = UnityCatalogCheckpointSaver(
    workspace_client=workspace_client,
    catalog="main",
    schema="langgraph",
    # Default tables: "checkpoints", "checkpoint_blobs", "checkpoint_writes"
    warehouse_id="your-warehouse-id",  # Optional
)

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile with checkpointer for persistence
graph = graph_builder.compile(checkpointer=checkpointer)

# Run conversation with persistence
config = {"configurable": {"thread_id": "conversation_1"}}

# First interaction
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello! What's the weather like?")]},
    config=config
)

# Second interaction - conversation history is maintained!
result = graph.invoke(
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config=config
)
# The bot remembers the previous question! 🎉
```

### 4. Async Usage for High Performance

```python
from langgraph_unity_catalog_checkpoint import AsyncUnityCatalogCheckpointSaver
import asyncio

# Create async checkpointer
async_checkpointer = AsyncUnityCatalogCheckpointSaver(
    workspace_client=workspace_client,
    catalog="main",
    schema="langgraph",
    warehouse_id="your-warehouse-id",
)

# Async chatbot node
async def async_chatbot(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# Build and compile with async checkpointer
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", async_chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=async_checkpointer)

# Run asynchronously
config = {"configurable": {"thread_id": "async_conversation_1"}}
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="Hello async world!")]},
    config=config
)
```

---

## 🎯 Use Cases

### 1. **Conversational AI with Memory**

Maintain conversation history across multiple interactions:

```python
# Each user gets their own conversation thread
config = {"configurable": {"thread_id": f"user_{user_id}"}}
graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
```

### 2. **Human-in-the-Loop Workflows**

Pause execution for human review and resume seamlessly:

```python
# Interrupt before critical nodes
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"]
)

# Execute and pause at approval
result = graph.invoke(input_data, config)

# Human reviews and approves...

# Resume from checkpoint
result = graph.invoke(None, config)  # Continues from where it left off
```

### 3. **Long-Term Memory with LangMem**

Integrate with [LangMem](https://github.com/langchain-ai/langmem) for user preferences and memories:

```python
from langchain.agents import create_agent
from langmem.tools import get_langmem_tools

# Create store for LangMem
store = UnityCatalogStore(
    workspace_client=workspace_client,
    catalog="main",
    schema="langgraph",
)

# Get LangMem tools
langmem_tools = get_langmem_tools(store=store)

# Create agent with memory
agent = create_agent(llm, tools + langmem_tools)

# Use with user context
config = {
    "configurable": {
        "langgraph_user_id": "user_123"  # Isolates memories per user
    }
}
agent.invoke({"messages": [HumanMessage(content="I prefer dark mode")]}, config)
```

### 4. **Production ML Pipelines**

Reliable state management for complex workflows:

```python
# Automatic recovery from failures
# Time-travel debugging with Delta Lake
# Full audit trail via Unity Catalog
# Multi-agent coordination with isolated states
```

---

## 📊 Performance Optimizations

### Batch Write Operations (2-10x Faster)

The implementation uses **batched SQL operations** to minimize round trips to Unity Catalog:

```python
# Instead of N+1 SQL statements:
# - 1 per blob
# - 1 per write
# - 1 checkpoint

# We use just 3 SQL statements:
# - 1 batch for all blobs
# - 1 batch for all writes  
# - 1 for checkpoint

# For a checkpoint with 5 blobs and 3 writes:
# Before: 9 SQL statements
# After: 3 SQL statements
# Speedup: 3x faster! ⚡
```

See [docs/CHECKPOINT_BATCH_WRITE_OPTIMIZATION.md](docs/CHECKPOINT_BATCH_WRITE_OPTIMIZATION.md) for details.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│  LangChain/LangGraph Application     │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│  BaseStore / BaseCheckpointSaver     │
│  (LangGraph Interfaces)              │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│  Unity Catalog Implementation        │
│  - UnityCatalogStore                 │
│  - UnityCatalogCheckpointSaver       │
│  - Async variants                    │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│  Databricks WorkspaceClient          │
│  (SQL Statement Execution API)       │
└──────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────┐
│  Unity Catalog Delta Tables          │
│  - ACID transactions                 │
│  - Time-travel                       │
│  - Change Data Feed                  │
└──────────────────────────────────────┘
```

### Data Storage

- **Serialization**: Checkpoints and values are serialized using LangGraph's `JsonPlusSerializer`
- **Binary Storage**: BINARY columns for efficient blob storage (base64 encoded)
- **JSON Metadata**: Structured metadata for filtering and querying
- **Delta Lake**: ACID transactions, time-travel, and optimization

### Default Table Names

| Component | Default Tables |
|-----------|---------------|
| **Store** | `store` |
| **Checkpointer** | `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` |

Tables are automatically created on first use with optimized schemas.

---

## 📚 Examples

### Complete Jupyter Notebooks

Explore the [`notebooks/`](notebooks/) directory for interactive examples:

- **[`store_example.ipynb`](notebooks/store_example.ipynb)** - Store operations and LangMem integration
- **[`checkpointer_example.ipynb`](notebooks/checkpointer_example.ipynb)** - Synchronous graph checkpointing
- **[`async_checkpointer_example.ipynb`](notebooks/async_checkpointer_example.ipynb)** - Async graph execution

### Run in Databricks

1. Upload a notebook to your Databricks workspace
2. Attach to a cluster with Unity Catalog access
3. Set the required configuration (catalog, schema, warehouse_id)
4. Run all cells

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABRICKS_HOST` | Workspace URL | Yes |
| `DATABRICKS_TOKEN` | Access token | Yes |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse ID | No |
| `UC_CATALOG` | Default catalog name | Recommended |
| `UC_SCHEMA` | Default schema name | Recommended |

### Configuration Precedence

Configuration values are resolved in this order:

1. **Environment variables** (highest priority)
2. **Databricks widgets** (for notebooks)
3. **Constructor parameters** (explicit values)

See [docs/CONFIGURATION_PRECEDENCE.md](docs/CONFIGURATION_PRECEDENCE.md) for details.

### Warehouse ID

The `warehouse_id` parameter is optional and defaults to `None`. If not provided:
- Uses the default warehouse for the workspace
- Can be overridden per-operation if needed

---

## 🔒 Permissions Required

Ensure your Databricks principal has:

- `USE CATALOG` on the target catalog
- `USE SCHEMA` on the target schema  
- `CREATE TABLE` on the target schema (for initialization)
- `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `MODIFY` on the tables

---

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_unity_catalog_store.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Run Integration Tests

Integration tests require a live Databricks connection:

```bash
# Set required environment variables
export DATABRICKS_HOST="..."
export DATABRICKS_TOKEN="..."
export DATABRICKS_WAREHOUSE_ID="..."

# Run integration tests
uv run pytest tests/test_integration.py -v
```

### Linting and Formatting

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

---

## 📖 Documentation

### Core Documentation

- **[Usage Guide](docs/USAGE.md)** - Comprehensive usage examples
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Technical architecture
- **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - Development environment
- **[Quick Start](QUICKSTART.md)** - Getting started guide
- **[Install Guide](INSTALL.md)** - Installation instructions

### Technical Details

- **[Checkpoint Batch Write Optimization](docs/CHECKPOINT_BATCH_WRITE_OPTIMIZATION.md)** - Performance optimization details
- **[Configuration Precedence](docs/CONFIGURATION_PRECEDENCE.md)** - Configuration resolution
- **[Default Table Names](docs/DEFAULT_TABLE_NAMES.md)** - Table naming conventions
- **[MLflow Autolog Setup](docs/MLFLOW_AUTOLOG_SETUP.md)** - Observability with MLflow
- **[Logging](docs/LOGGING.md)** - Logging configuration

### Session Summaries

- **[Batch Optimization (2025-11-07)](docs/SESSION_SUMMARY_2025-11-07_BATCH_OPTIMIZATION.md)**
- **[MLflow Tracing Removal (2025-11-07)](docs/SESSION_SUMMARY_2025-11-07_MLFLOW_TRACING_REMOVAL.md)**

---

## 🚀 Features

### UnityCatalogStore

- ✅ Implements `langgraph.store.base.BaseStore` interface
- ✅ Batch operations (`batch`, `abatch`) for performance
- ✅ Namespaced key-value storage
- ✅ Search with filtering and pagination
- ✅ Automatic table initialization
- ✅ Sync and async implementations
- ✅ Compatible with LangMem for long-term memory

### UnityCatalogCheckpointSaver

- ✅ Implements `BaseCheckpointSaver` interface
- ✅ Full LangGraph checkpoint persistence
- ✅ Support for human-in-the-loop workflows
- ✅ Multi-turn conversation memory
- ✅ State recovery and time-travel
- ✅ Pending writes management
- ✅ Checkpoint listing and filtering
- ✅ Sync and async implementations
- ✅ Optimized batch writes (2-10x faster)
- ✅ Automatic table creation and schema management

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/natefleming/langgraph_unity_catalog_checkpoint.git
cd langgraph_unity_catalog_checkpoint

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Project Structure

```
langgraph_unity_catalog_checkpoint/
├── src/
│   └── langgraph_unity_catalog_checkpoint/
│       ├── store/              # Store implementations
│       │   ├── unity_catalog.py    # Sync store
│       │   ├── aio.py              # Async store
│       │   └── base.py             # Base store class
│       ├── checkpoint/         # Checkpointer implementations
│       │   ├── unity_catalog.py    # Sync checkpointer
│       │   ├── aio.py              # Async checkpointer
│       │   └── base.py             # Base checkpointer class
│       └── __init__.py         # Public API exports
├── tests/                      # Test suite
├── notebooks/                  # Example notebooks
├── docs/                       # Documentation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`make test`)
5. Format and lint (`make format lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based agent framework
- [LangMem](https://github.com/langchain-ai/langmem) - Long-term memory for agents
- [Databricks SDK](https://github.com/databricks/databricks-sdk-py) - Databricks API client
- [Unity Catalog](https://www.databricks.com/product/unity-catalog) - Data governance platform

---

## 📞 Support

For issues and questions:

- **GitHub Issues**: [Open an issue](https://github.com/natefleming/langgraph_unity_catalog_checkpoint/issues)
- **Documentation**: Check the [`docs/`](docs/) directory
- **Examples**: Review the [`notebooks/`](notebooks/) directory

---

## 🗺️ Roadmap

Planned enhancements:

- [ ] Connection pooling for improved performance
- [ ] Configurable TTL for automatic checkpoint cleanup
- [ ] Metrics and monitoring integration
- [ ] Query optimization hints and caching
- [ ] Support for alternative serialization formats
- [ ] Bulk import/export utilities
- [ ] Multi-region replication support

---

## ⚡ Quick Links

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Usage Examples](docs/USAGE.md)** - Detailed usage patterns
- **[Notebooks](notebooks/)** - Interactive examples
- **[API Reference](docs/IMPLEMENTATION_SUMMARY.md)** - Technical details

---

**Made with ❤️ for the LangChain community**
