PY := .venv/bin/python

.PHONY: check reproduce viz record

check:
	$(PY) -m pytest -q

reproduce:
	$(PY) scripts/reproduce.py

viz:
	$(PY) viz/server.py

record:
	$(PY) viz/record.py
