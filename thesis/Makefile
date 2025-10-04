PY=python3
PIP=pip3

setup:
	$(PIP) install -r requirements.txt

data:
	$(PY) scripts/generate_data.py

figures:
	$(PY) scripts/analyze_and_figures.py

wb:
	$(PY) scripts/fetch_world_bank.py

all: data figures wb
	@echo "Data and figures generated. Build LaTeX with your TeX toolchain."
