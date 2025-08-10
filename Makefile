.PHONY: venv requirements.txt requirements-dev.txt install format lint typecheck test test-v cov docs all

all: format lint typecheck

venv: 
	uv venv

sync:
	uv sync

format:
	uv run ruff format

lint:
	uv run ruff check

typecheck:
	uv run  mypy .
	uv run ty check

test:
	uv run pytest -q

test-v:
	uv run pytest -s -o log_cli=true -o log_cli_level=debug

cov:
	uv run pytest -q --cov=vgrad --cov-report html
	xdg-open htmlcov/index.html

docs:
	uv run pdoc vgrad -o docs_build

vgrad/spirv_instructions.py: vgrad/tools/spirv_generator.py
	uv run python vgrad/tools/spirv_generator.py  ../SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json ../SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json > vgrad/spirv_instructions.py
