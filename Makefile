install:
	poetry install --no-root
lint:
	poetry run ruff check src
format:
	poetry run black .
typecheck:
	poetry run mypy src
