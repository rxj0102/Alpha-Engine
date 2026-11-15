.PHONY: reproduce render check test

reproduce:  ## rerun the experiment on the frozen data and refresh README
	uv run python scripts/reproduce.py
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:      ## README numbers must match results/
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
