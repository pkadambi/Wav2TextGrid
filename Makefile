.PHONY: format format-check lint lint-check mypy-check

format:
	@echo "==> Formatting code with Ruff..."
	uv run --only-group dev ruff format .

format-check:
	@echo "==> Checking code formatting with Ruff..."
	uv run --only-group dev ruff format --check .

lint:
	@echo "==> Linting with Ruff..."
	uv run --only-group dev ruff check --fix .

lint-check:
	@echo "==> Checking linting with Ruff..."
	uv run --only-group dev ruff check .

mypy-check:
	@echo "==> Running mypy for type checking..."
	uv run --only-group dev mypy .

fresh-slate:
	@echo "==> Removing virtual environment and lock file..."
	@read -p "Are you sure you want to proceed? [y/N] " confirm && [ $${confirm} = "y" ] || [ $${confirm} = "Y" ] && rm -rf uv.lock .venv || echo "Aborted."
