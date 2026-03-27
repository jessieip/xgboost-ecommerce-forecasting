install:
	poetry install --no-root
lint:
	poetry run nbqa flake8 . --ignore=E501
format:
	poetry run black .