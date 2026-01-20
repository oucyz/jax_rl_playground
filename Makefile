_ruff_lint:
	uv run ruff check

_ruff_isort:
	uv run ruff check --select I --fix

_ruff_fmt:
	uv run ruff format

_mypy:
	uv run mypy ./

lint:
	make -j _ruff_lint _mypy

fmt:
	make -j _ruff_isort _ruff_fmt 

.PHONY: test
test:
	uv run pytest . -vv

_coverage_run:
	uv run coverage run -m pytest

_coverage_report:
	uv run coverage report

_coverage_html:
	uv run coverage html

_visualize_coverage:
	make _coverage_html && open htmlcov/index.html

coverage:
	make _coverage_run _coverage_report

vis_coverage:
	make coverage _visualize_coverage
