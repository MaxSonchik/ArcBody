.PHONY: help install dev test lint typecheck format serve train calibrate demo docker clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

.venv:
	python3 -m venv .venv && $(PIP) install --upgrade pip

install: .venv ## Install the package and its dev tools
	$(PIP) install -e ".[dev]"

dev: .venv ## Install everything, production perception backend included
	$(PIP) install -e ".[dev,yolo]"

test: ## Run the test suite
	$(PY) -m pytest tests/ -q

lint: ## Check style
	$(PY) -m ruff check arcbody tests scripts

format: ## Apply formatting fixes
	$(PY) -m ruff format arcbody tests scripts && $(PY) -m ruff check --fix arcbody tests scripts

typecheck: ## Static types
	$(PY) -m mypy arcbody

serve: ## Run the API locally with reload
	$(PY) -m uvicorn arcbody.api.main:app --reload --port 8000

train: ## Fine-tune the encoder on synthetic subjects
	$(PY) -m arcbody.training.train --output weights/arcbody.pt --metrics outputs/train_metrics.json

calibrate: ## Refit the ratio reference distribution used for prompt wording
	$(PY) scripts/calibrate_ratios.py --samples 400 --out outputs/ratio_calibration.json

demo: ## Render sample subjects and their control maps into outputs/
	$(PY) -m arcbody.training.synthetic --identities 3 --per-identity 2 --easy
	$(PY) scripts/demo.py

docker: ## Build the container
	docker compose build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs/demo build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
