SHELL := /bin/sh

venv:
	python -m venv .venv

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

lint:
	ruff check quantfinlab tests

format:
	black quantfinlab tests

test:
	pytest
