install:
	pip install -r requirements.txt
lint:
	nbqa flake8 . --ignore=E501
format:
	black . --include '\.ipynb$$'