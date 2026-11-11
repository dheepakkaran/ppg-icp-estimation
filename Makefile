# Research prototype: ICP from finger PPG on open MIMIC-III waveforms.
PY := .venv/bin/python
# uv and pixi: taken from PATH; a local tools folder is used only if it exists
TOOLS := $(HOME)/workspace/.tools
ifneq ($(wildcard $(TOOLS)/bin/uv),)
UV ?= $(TOOLS)/bin/uv
export UV_CACHE_DIR ?= $(TOOLS)/uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(TOOLS)/python
else
UV ?= uv
endif
export PYTHONHASHSEED := 0
# single-threaded BLAS/OpenMP: the machine is shared; HistGradientBoosting crawls when oversubscribed
export OMP_NUM_THREADS := 1
SUBJECTS ?= 12
HOURS ?= 2

.PHONY: reproduce env find fetch features models matlab report test

reproduce: env fetch features test models matlab report

env:
	$(UV) sync --frozen --python 3.12

# Header scan of the matched subset (slow, ~5 min per 6k records); its output
# results/candidates.csv is committed, so `reproduce` starts from it.
find:
	python3 -I scripts/find_records.py 12 7 0 results/candidates_part1.csv
	python3 -I scripts/find_records.py 20 6 6400 results/candidates_part2.csv
	(cat results/candidates_part1.csv; tail -n +2 results/candidates_part2.csv) | awk '!seen[$$0]++' > results/candidates.csv

fetch:
	$(PY) scripts/fetch.py $(SUBJECTS) $(HOURS)

features:
	$(PY) scripts/build_features.py

test:
	$(PY) -m pytest -q tests

models:
	$(PY) scripts/run_models.py

# MATLAB-syntax check under GNU Octave via pixi (conda-forge octave + octave-signal); skipped if pixi is missing.
# In MATLAB: cd to the repo root and run `addpath matlab; preprocess_check`, then `python scripts/compare_matlab.py`.
matlab:
	@if [ -f matlab/env.sh ]; then . matlab/env.sh; fi; if command -v pixi >/dev/null 2>&1; then \
	  cd matlab && pixi run octave --no-gui --quiet --eval "disp(version)" 2>/dev/null | tail -1 | sed 's/^/GNU Octave /' > ../results/matlab_engine.txt && \
	  pixi run octave --no-gui --quiet --eval "cd('..'); addpath('matlab'); preprocess_check" 2>/dev/null | tee ../results/matlab_check_octave.txt && \
	  cd .. && $(PY) scripts/compare_matlab.py; \
	else echo "pixi not found: Octave check skipped"; fi

report:
	$(PY) scripts/make_results.py
