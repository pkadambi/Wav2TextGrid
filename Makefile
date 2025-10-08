.PHONY: black isort flake8 mypy code-quality

black:
	@echo "==> Checking code formatting with Black..."
	@black --check src/ || (echo "Black formatting issues found."; exit 1)
	@echo "Black passed."

isort:
	@echo "==> Checking import sorting with isort..."
	@isort --check-only src/ || (echo "isort import sorting issues found."; exit 1)
	@echo "isort passed."

flake8:
	@echo "==> Linting with flake8..."
	@flake8 src/ --max-line-length=100 || (echo "flake8 linting issues found."; exit 1)
	@echo "flake8 passed."

mypy:
	@echo "==> Type checking with mypy..."
	@mypy src/ --ignore-missing-imports || (echo "mypy type checking issues found."; exit 1)
	@echo "mypy passed."

code-quality: black isort flake8 mypy
	@echo "=========================================="
	@echo "All code quality checks passed! 🎉"
	@echo "=========================================="