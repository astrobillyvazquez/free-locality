# Tier-A reproduction: regenerate every figure and number from committed data with NumPy/Matplotlib.
TRACE30  ?= data/traces/moe_Qwen3-30B-A3B_xdoc.npz
TRACE235 ?= data/traces/moe_Qwen3-235B-A22B_xdoc.npz

.PHONY: figures cis test all clean
all: figures cis test

figures:        ## regenerate paper figures into ./figures (30B projected + 235B measured iso-VRAM)
	python scripts/make_paper_figures.py --moe-trace $(TRACE30)
	python scripts/make_paper_figures.py --only vram --moe-trace $(TRACE235)
	cp -f figures/paper_vram_isoquality.png figures/paper_vram_isoquality_235B_measured.png
	python scripts/make_paper_figures.py --only vram --moe-trace $(TRACE30)   # restore 30B as the default-named figure

cis:            ## recompute CIs + LRU/Belady baseline for BOTH traces
	python scripts/analyze_residency_cis.py $(TRACE30)
	python scripts/analyze_residency_cis.py $(TRACE235)

test:           ## NumPy-only unit tests
	pytest

clean:
	rm -rf .pytest_cache **/__pycache__
