.PHONY: venv requirements.txt requirements-dev.txt install format lint typecheck test cov all

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

test:
	pytest -q # pytest -s -o log_cli=true -o log_cli_level=debug

cov:
	pytest -q --cov=vutur --cov-report html
	xdg-open htmlcov/index.html
