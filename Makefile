COMPOSE ?= $(shell \
	if command -v docker >/dev/null 2>&1; then \
		echo "docker compose"; \
	elif command -v podman >/dev/null 2>&1; then \
		echo "podman compose"; \
	else \
		echo "docker compose"; \
	fi)

.PHONY: run run-build stop clean setup-local badges

run:
	@echo "Using compose backend: $(COMPOSE)"
	@echo "Pulling prebuilt image (if available)..."
	@$(COMPOSE) pull || true
	$(COMPOSE) up -d --no-build
	@echo ""
	@echo "🔥 TorchCode is running!"
	@echo "   Open http://localhost:8888"
	@echo ""

run-build:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) up --build -d
	@echo ""
	@echo "🔥 TorchCode is running (local build)!"
	@echo "   Open http://localhost:8888"
	@echo ""

stop:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down

clean:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down -v
	rm -f data/progress.json

setup-local:
	@mkdir -p notebooks/_original_templates
	@cp templates/*.ipynb notebooks/_original_templates/
	@cp templates/*.ipynb notebooks/
	@cp solutions/*.ipynb notebooks/
	@echo "✅ Local notebooks ready in ./notebooks/"

badges:
	python scripts/add_colab_badges.py
