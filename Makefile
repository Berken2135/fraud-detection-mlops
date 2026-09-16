.PHONY: setup train serve test lint drift retrain up down

VENV := .venv/Scripts

setup:
	python -m venv .venv
	$(VENV)/python -m pip install --upgrade pip
	$(VENV)/pip install -r requirements-dev.txt

train:
	$(VENV)/python -m src.train

serve:
	$(VENV)/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(VENV)/pytest tests/ -v

lint:
	$(VENV)/ruff check src/ api/ tests/ pipelines/ monitoring/

drift:
	$(VENV)/python -m monitoring.simulate_traffic
	$(VENV)/python -m monitoring.drift_check

retrain:
	$(VENV)/python -m pipelines.retrain_flow

up:
	docker compose up --build

down:
	docker compose down
