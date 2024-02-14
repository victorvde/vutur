.PHONY: venv requirements.txt requirements-dev.txt install format lint

venv: 
	python3 -m venv .venv

requirements.txt:
	pip-compile --upgrade

requirements-dev.txt:
	pip-compile --extra dev -o requirements-dev.txt

install:
	pip-sync requirements.txt requirements-dev.txt

format:
	ruff format

lint:
	ruff check

typecheck:
	mypy . --disallow-untyped-defs
