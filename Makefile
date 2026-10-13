# TraceToPolicy — post-training + continuous learning for an NL-to-SQL agent.
#
# Usage:
#   make setup       create .venv and install dependencies
#   make warehouse   build the DuckDB warehouse (50k rows, deterministic)
#   make traces      harvest agent traces on the train split
#   make label       build SFT/DPO datasets from traces
#   make sft         LoRA SFT
#   make dpo         LoRA DPO (from the SFT adapter)
#   make eval        baseline vs SFT vs DPO held-out eval
#   make loop        run continuous-learning cycles (gates + rollback)
#   make dashboard   render results/dashboard.html
#   make test        run the fast test suite
#   make all         everything, in order

SHELL := /bin/bash
PYTHON_PIN ?= python3.11
VENV := .venv

# Prefer the project venv when it exists.
ifeq ($(wildcard $(VENV)/bin/python),)
PY := $(PYTHON_PIN)
else
PY := $(VENV)/bin/python
endif

.PHONY: setup warehouse traces label sft dpo eval loop dashboard test all clean

setup:
	$(PYTHON_PIN) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt
	$(VENV)/bin/python -m pip install -e . --no-deps

warehouse:
	$(PY) -m tracetopolicy.cli warehouse

traces:
	$(PY) -m tracetopolicy.cli harvest --split train --samples-per-task 12 --out data/traces_train.jsonl --batch-size 32

label:
	$(PY) -m tracetopolicy.cli label --traces data/traces_train.jsonl --outdir data/datasets

sft:
	$(PY) -m tracetopolicy.train_sft --config configs/sft.yaml

dpo:
	$(PY) -m tracetopolicy.train_dpo --config configs/dpo.yaml

eval:
	$(PY) -m tracetopolicy.cli eval --config configs/eval.yaml

loop:
	$(PY) -m tracetopolicy.cli loop --cycles 2

dashboard:
	$(PY) -m tracetopolicy.cli dashboard

test:
	$(PY) -m pytest -m "not slow" --cov=tracetopolicy --cov-report=term-missing

all: setup warehouse traces label sft dpo eval loop dashboard test

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache artifacts outputs wandb data/*.duckdb
