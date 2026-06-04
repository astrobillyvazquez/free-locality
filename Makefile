# Tier-A reproduction: regenerate every figure and number from committed data with NumPy/Matplotlib.
TRACE ?= data/traces/moe_Qwen3-30B-A3B_xdoc.npz

.PHONY: figures cis test all clean
all: figures cis test

figures:        ## regenerate paper figures into ./figures from the committed trace + CSVs
	python scripts/make_paper_figures.py --moe-trace $(TRACE)

cis:            ## recompute confidence intervals + LRU/Belady baseline (writes data/eval/residency_cis_lru.csv)
	python scripts/analyze_residency_cis.py

test:           ## run the NumPy-only unit tests
	pytest

clean:
	rm -rf .pytest_cache **/__pycache__
