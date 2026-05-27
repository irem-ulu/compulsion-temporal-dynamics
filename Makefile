.PHONY: setup test lint fmt run report compare power dashboard clean

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install || true

test:
	$(BIN)/pytest tests/ -q

lint:
	$(BIN)/ruff check src tests

fmt:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

run:
	$(BIN)/ctd run

report:
	$(BIN)/ctd report

compare:
	$(BIN)/ctd compare

power:
	$(BIN)/ctd power

dashboard:
	$(BIN)/streamlit run dashboard/app.py

clean:
	rm -rf results/ data/raw_data.csv .pytest_cache .ruff_cache
