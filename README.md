# Free Locality — reproducible artifacts

Reproducible artifacts for the paper **"Free Locality: When an LLM's Knowledge Memory Is DDR5-Pageable,
and Why Training for It Backfires."**

Addressable LLM knowledge stores — product-key **memory layers** and **MoE experts** — access only a few
percent of their parameters per token, and that access is **cache-local without any locality training**.
This repo lets you verify the paper's measurements: the universe-size law, the training-for-locality
negative, and the iso-quality-VRAM result, plus confidence intervals and a dynamic-cache baseline —
all from committed data with NumPy.

## Install
```bash
pip install -e ".[dev]"      # numpy, matplotlib, pytest
```

## Reproduce everything (Tier A — pure compute, offline)
```bash
make figures   # regenerate every paper figure into ./figures from the committed trace + CSVs
make cis       # recompute the confidence intervals + LRU/Belady dynamic-cache baseline
make test      # NumPy-only unit tests for the locality engine
```
CI (`.github/workflows/ci.yml`) runs all three on every push.

## Claim → artifact → expected number

| Paper claim | Command | Expected output |
|---|---|---|
| Held-out iso-quality-VRAM: per-layer frequency residency hits **0.334** at the 8% anchor vs **0.080** blind | `python scripts/analyze_residency_cis.py` | `held-out hit@8% = 0.334` |
| **1.76×** less resident VRAM at iso-0.95 hit | `python scripts/analyze_residency_cis.py` | `iso-0.95 VRAM ratio = 1.77x` |
| Confidence intervals (all 70 calib/eval splits) | `python scripts/analyze_residency_cis.py` | hit@8% band `[0.291, 0.354]`; ratio band `[1.64, 1.82]x` |
| Dynamic-cache baseline (LRU/Belady > static > blind) | `python scripts/analyze_residency_cis.py` | table → `data/eval/residency_cis_lru.csv` |
| iso-quality-VRAM figure | `make figures` | `figures/paper_vram_isoquality.png` |
| Live KTransformers knee (frequency vs random, 1.02×→3.60×) | `make figures` (data: `data/eval/ktransformers_live_h100_ratio_sweep.csv`) | `figures/paper_livedemo_knee.png` |
| KTransformers `--init-expert-location` frequency map | `python scripts/trace_to_ktransformers_stats.py data/traces/moe_Qwen3-30B-A3B_xdoc.npz -o /tmp/stats.pt` (needs `pip install ".[capture]"`) | `static-freq hit @ 7.8% = 0.359` |
| **235B (measured) held-out hit@8% = 0.366** vs 0.080 blind | `python scripts/analyze_residency_cis.py data/traces/moe_Qwen3-235B-A22B_xdoc.npz` | `held-out hit@8% = 0.366` |
| **235B (measured) iso-0.95 VRAM 1.72x** (95% CV [1.58,1.77]) | (same command) | `iso-0.95 VRAM ratio = 1.72x` |
| 235B measured iso-VRAM figure (native geometry) | `make figures` | `figures/paper_vram_isoquality_235B_measured.png` |
| Locality engine known-answer tests | `pytest` | all pass |


## Scale confirmation (30B → 235B, measured)
The 30B-A3B iso-quality-VRAM result is **confirmed by direct measurement** on a real Qwen3-235B-A22B
routing trace (94 MoE layers × 128 experts, top-8, 32 cross-document docs) at the same universe/k = 16.
Per-layer locality holds across the ~8× parameter scale-up: held-out hit@8% **0.334 (30B) → 0.366 (235B)**,
iso-0.95 VRAM **1.76× → 1.71×** — the 235B GB axis is now *measured*, not projected. Reproduce with
`python scripts/analyze_residency_cis.py data/traces/moe_Qwen3-235B-A22B_xdoc.npz`.

## Reproducibility tiers (honest scope)
- **Tier A (this repo, push-button):** every number/figure above regenerates from committed data with
  NumPy/Matplotlib. This is the credibility core.
- **Tier B (GPU, scripted, not in CI):** capturing a fresh routing trace from a model
  (`scripts/capture_moe_routing.py`, not shipped here) needs a GPU + `transformers`/`bitsandbytes`. The
  committed `.npz` trace is the *output* of that step, so Tier-A verification needs no GPU.
- **Tier C (hardware-dependent, protocol + recorded outputs):** the live KTransformers
  `frequency`-vs-`random` deployment numbers (`data/eval/ktransformers_live_h100_ratio_sweep.csv`) are
  measured outputs from an H100 run, included as data + figure; the live tok/s itself is engine/hardware
  dependent and not reproduced by `make`.

The large memory-bank (product-key) universe sweep behind the headline universe-size-law figure
(`figures/25_universe_law.png`, `figures/paper_hitrate_curve.png`) used multi-GB traces not committed here;
the figures ship pre-rendered and the traces are available on request.

## Data provenance
Routing traces were captured by running the (Apache-2.0) **Qwen3-30B-A3B** model over **WikiText-2**
(CC-BY-SA) documents and recording per-token top-k expert selections. Committed `.npz`/`.csv`/`.pt` files
contain only **aggregated routing indices and measured metrics** — no model weights, no document text.
See `data/traces/README.md`.

## Citation
See `CITATION.cff` (arXiv id added when the preprint is posted).

## License
MIT — see `LICENSE`.
