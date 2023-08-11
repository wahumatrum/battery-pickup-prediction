SHELL := /bin/bash

.PHONY: setup
setup:
	pyenv local 3.9.8
	python -m venv .venv
	. .venv/bin/activate
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements_base.txt
	.venv/bin/python -m pip install -r requirements.txt
