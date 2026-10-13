# TraceToPolicy — post-training + continuous learning for an NL-to-SQL agent.
#
# Usage:
#   make setup       create .venv and install dependencies
#   make warehouse   build the DuckDB warehouse (~54k rows, deterministic)
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
# Make's shell has a minimal PATH; make sure Homebrew/Linuxbrew locations are visible.
export PATH := /opt/homebrew/bin:/usr/local/bin:$(PATH)
# Prefer 3.11, then 3.12, then whatever python3 is available.
PYTHON_PIN ?= $(shell command -v python3.11 2>/dev/null || command -v python3.12 2>/dev/null || command -v python3)
VENV := .venv
PY := $(VENV)/bin/python
SENTINEL := $(VENV)/.installed

.PHONY: setup warehouse traces label sft dpo eval loop dashboard test all clean

# Create the venv once; re-run when the dependency files change.
$(SENTINEL): requirements.txt pyproject.toml
	$(PYTHON_PIN) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
	touch $(SENTINEL)

setup: $(SENTINEL)

warehouse: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli warehouse

traces: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli harvest --split train --samples-per-task 12 --out data/traces_train.jsonl --batch-size 32

label: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli label --traces data/traces_train.jsonl --outdir data/datasets

sft: | $(SENTINEL)
	$(PY) -m tracetopolicy.train_sft --config configs/sft.yaml

dpo: | $(SENTINEL)
	$(PY) -m tracetopolicy.train_dpo --config configs/dpo.yaml

eval: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli eval --config configs/eval.yaml

loop: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli loop --cycles 2

dashboard: | $(SENTINEL)
	$(PY) -m tracetopolicy.cli dashboard

test: | $(SENTINEL)
	$(PY) -m pytest -m "not slow" --cov=tracetopolicy --cov-report=term-missing

all: setup warehouse traces label sft dpo eval loop dashboard test

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache artifacts outputs wandb data/*.duckdb
