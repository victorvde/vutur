.PHONY: venv requirements.txt requirements-dev.txt install format lint typecheck all

all: format lint typecheck

venv: 
	python3 -m venv .venv

requirements.txt:
	uv pip compile pyproject.toml --upgrade -o requirements.txt

requirements-dev.txt:
	uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

install: requirements-dev.txt
	uv pip sync requirements-dev.txt

format:
	ruff format

lint:
	ruff check

typecheck:
	mypy . --disallow-untyped-defs
