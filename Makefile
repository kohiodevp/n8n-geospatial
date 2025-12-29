SHELL := /bin/bash

.PHONY: up down build ps logs health backup restore mcp-ping fmt lint

up:
	docker compose up -d --build

e2e: up health

down:
	docker compose down -v

build:
	docker compose build

ps:
	docker ps

logs:
	docker logs --tail=100 n8n-geospatial || true; \
	docker logs --tail=100 mcp-server || true; \
	docker logs --tail=100 postgis || true; \
	docker logs --tail=100 redis || true

health:
	python3 scripts/health_check.py --json --url http://localhost:5678/healthz; \
	curl -fsS http://localhost:5001/healthz

backup:
	bash scripts/backup_db.sh

restore:
	@if [ -z "$(file)" ]; then echo "Usage: make restore file=./backups/DB_YYYYmmdd_HHMMSS.sql.gz"; exit 1; fi; \
	bash scripts/restore_db.sh $(file)

mcp-ping:
	docker exec n8n-geospatial sh -lc "curl -sS -H 'Content-Type: application/json' -d '{\"tool_name\":\"ping\",\"params\":{\"source\":\"make\"}}' http://mcp-server:5001/invoke"

fmt:
	pip install ruff; ruff format scripts || true

lint:
	pip install ruff; ruff check scripts || true
