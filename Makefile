TOP_DIR := .
SRC_DIR := $(TOP_DIR)/src
TEST_DIR := $(TOP_DIR)/tests
DOCS_DIR := $(TOP_DIR)/docs
DIST_DIR := $(TOP_DIR)/dist
REQUIREMENTS_FILE := $(TOP_DIR)/requirements.txt
LIB_NAME := langchain_unity_catalog_persistence
LIB_VERSION := $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)
LIB := $(LIB_NAME)-$(LIB_VERSION)-py3-none-any.whl
TARGET := $(DIST_DIR)/$(LIB)

ifeq ($(OS),Windows_NT)
    PYTHON := py.exe
else
    PYTHON := python3
endif

UV := uv
SYNC := $(UV) sync
BUILD := $(UV) build
PYTHON := $(UV) run python
EXPORT := $(UV) pip freeze --exclude-editable | grep -v -E "(databricks-vectorsearch|pyspark|databricks-connect)"
PUBLISH := $(UV) run twine upload
PYTEST := $(UV) run pytest -v -s
RUFF_CHECK := $(UV) run ruff check --fix --ignore E501
RUFF_FORMAT := $(UV) run ruff format
FIND := $(shell which find)
RM := rm -rf
CD := cd

.PHONY: all clean distclean dist check format publish help test install depends examples lint watch-test

all: dist

install: depends
	$(SYNC)

dist: install
	$(BUILD)

depends:
	@$(SYNC)
	@$(EXPORT) > $(REQUIREMENTS_FILE)

check:
	$(RUFF_CHECK) $(SRC_DIR) $(TEST_DIR)

format: check depends
	$(RUFF_FORMAT) $(SRC_DIR) $(TEST_DIR)

publish: dist
	$(PUBLISH) $(DIST_DIR)/*

clean:
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyc -exec rm -f {} \;
	$(FIND) $(SRC_DIR) $(TEST_DIR) -name \*.pyo -exec rm -f {} \;

distclean: clean
	$(RM) $(DIST_DIR)
	$(RM) $(SRC_DIR)/*.egg-info
	$(RM) $(TOP_DIR)/.pytest_cache
	$(RM) $(TOP_DIR)/.ruff_cache
	$(RM) $(TOP_DIR)/.mypy_cache
	$(RM) $(TOP_DIR)/htmlcov
	$(RM) $(TOP_DIR)/.coverage
	$(FIND) $(SRC_DIR) $(TEST_DIR) \( -name __pycache__ -a -type d \) -prune -exec rm -rf {} \;

test: 
	$(PYTEST) -ra --tb=short $(TEST_DIR)

examples:
	@echo "Running store example..."
	$(PYTHON) -m src.examples.store_example
	@echo ""
	@echo "Running checkpointer example..."
	$(PYTHON) -m src.examples.checkpointer_example
	@echo ""
	@echo "Running async checkpointer example..."
	$(PYTHON) -m src.examples.async_checkpointer_example

lint: check

watch-test:
	$(PYTEST) -f $(TEST_DIR)

help:
	$(info TOP_DIR: $(TOP_DIR))
	$(info SRC_DIR: $(SRC_DIR))
	$(info TEST_DIR: $(TEST_DIR))
	$(info DIST_DIR: $(DIST_DIR))
	$(info DOCS_DIR: $(DOCS_DIR))
	$(info LIB: $(LIB))
	$(info )
	$(info $$> make [all|dist|install|clean|distclean|format|depends|publish|test|help])
	$(info )
	$(info       all         - build library: [$(LIB)]. This is the default)
	$(info       dist        - build library: [$(LIB)])
	$(info       install     - installs dependencies)
	$(info       clean       - removes build artifacts)
	$(info       distclean   - removes library and caches)
	$(info       format      - format source code with ruff)
	$(info       check       - check code with ruff)
	$(info       lint        - alias for check)
	$(info       depends     - installs library dependencies)
	$(info       publish     - publish library to PyPI)
	$(info       test        - run all tests)
	$(info       examples    - run all example scripts)
	$(info       watch-test  - run tests in watch mode)
	$(info       help        - show this help message)
	@true

