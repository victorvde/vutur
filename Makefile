.PHONY: venv requirements.txt requirements-dev.txt install format lint typecheck test test-v cov docs all

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
	mypy .
	ty check

test:
	pytest -q

test-v:
	pytest -s -o log_cli=true -o log_cli_level=debug

cov:
	pytest -q --cov=vgrad --cov-report html
	xdg-open htmlcov/index.html

docs:
	pdoc vgrad -o docs_build

vgrad/spirv_instructions.py: vgrad/tools/spirv_generator.py
	python vgrad/tools/spirv_generator.py  ../SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json ../SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json > vgrad/spirv_instructions.py
