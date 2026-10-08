PY := .venv/bin/python

.PHONY: check reproduce

check:
	$(PY) -m pytest -q

reproduce:
	$(PY) scripts/reproduce.py
