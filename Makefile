SHELL=/usr/bin/env bash

PYTHON_FILES := ./embedding/*.py
TEST_DIR := ./tests/*.py
export PYTHONPATH := $(PYTHONPATH):$(PWD)/embedding

# Specify the names of all executables to make.
PROG=check flake8 black test
.PHONY: ${PROG}

default:
	@echo "An explicit target is required. Available options: ${PROG}"

check: black-check flake8 pyright

precommit: black-fix check

flake8:
	flake8 .

black-check:
	isort --check .
	black --config pyproject.toml --check .

black-fix:
	isort .
	black --config pyproject.toml .

pyright:
	pyright .

test:
	pytest -v ${TEST_DIR}
