#!/usr/bin/env python
"""Confidence intervals + dynamic-policy baselines for the MoE iso-quality-VRAM result.

Addresses two reviewer asks (a pre-submission review:
  (a) bootstrap/CV confidence intervals on the cross-document held-out hit (0.334 @ 8%) and the
      iso-0.95 VRAM ratio (1.76x), and
  (b) a *dynamic* cache baseline (LRU, Belady/OPT) scored on the SAME held-out stream as the static
      frequency map, so "frequency vs random" is no longer the only comparison.

Methodology, kept identical to turboquant.core.memlocality.static_frequency_curve:
  - per layer, residency keeps the top-m frequency-ranked experts (m = round(f * n_slots));
  - aggregate hit = sum_L captured / sum_L total (access-weighted across the 48 layers);
  - held-out = rank residency on a CALIB document set, score captured accesses on a DISJOINT EVAL set.

CI estimator: because the trace has only 8 documents (4 calib / 4 eval in the canonical even/odd
split), a 4-doc bootstrap is coarse. We instead enumerate ALL C(8,4)=70 calib/eval partitions and
report the median and 2.5/97.5 percentile of the held-out statistic across splits (a leave-set-out
cross-validation band). The canonical even/odd split value (the paper's 0.334 / 1.76x) is reported
alongside and should sit inside the band. A doc-level bootstrap over the 4 eval docs is also printed
for reference. The 235B D2 run (32 docs) will tighten this materially.

    .venv/bin/python scripts/analyze_residency_cis.py
"""
from __future__ import annotations
import itertools
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

TRACE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/traces/moe_Qwen3-30B-A3B_xdoc.npz")
FRACS = (0.08, 0.10, 0.30, 0.50)     # per-layer resident fractions to report LRU/static/blind at
TARGET = 0.95                         # iso-hit-rate for the VRAM-ratio metric
ANCHOR_F = 0.08                       # the paper's 8% anchor
MAX_ENUM = 200                        # enumerate all calib/eval splits if C(n, n/2) <= this, else sample
N_RANDOM_SPLITS = 400                 # random balanced splits when enumeration is too large


def load():
    d = np.load(TRACE)
    layers = sorted(int(k[1:-4]) for k in d.files if k.startswith("L") and k.endswith("_idx"))
    n_slots = int(d["n_slots"])
    doc_id = d["doc_id"]
    docs = np.unique(doc_id).tolist()
    idx = {L: d[f"L{L}_idx"] for L in layers}
    k = int(idx[layers[0]].shape[1])
    counts = {L: {dd: np.bincount(idx[L][doc_id == dd].reshape(-1), minlength=n_slots)[:n_slots]
                  .astype(np.float64) for dd in docs} for L in layers}
    return layers, n_slots, k, docs, doc_id, idx, counts


def heldout_hit_at_m(layers, counts, calib_docs, eval_docs, m):
    """Aggregate held-out hit @ top-m: rank by calib counts per layer, score captured frac on eval."""
    captured = total = 0.0
    for L in layers:
        cal = sum(counts[L][dd] for dd in calib_docs)
        evl = sum(counts[L][dd] for dd in eval_docs)
        order = np.argsort(cal)[::-1]                 # experts by calib frequency
        ev_by_rank = evl[order]
        captured += ev_by_rank[:m].sum()
        total += evl.sum()
    return captured / total if total else 0.0


def heldout_curve(layers, counts, calib_docs, eval_docs, n_slots):
    """held-out agg hit for every m in 0..n_slots (vectorized over layers)."""
    cum = np.zeros(n_slots + 1)
    total = 0.0
    for L in layers:
        cal = sum(counts[L][dd] for dd in calib_docs)
        evl = sum(counts[L][dd] for dd in eval_docs)
        order = np.argsort(cal)[::-1]
        cum[1:] += np.cumsum(evl[order])
        total += evl.sum()
    return cum / total if total else cum                # cum[m] = captured frac by top-m


def iso_ratio(curve, n_slots, target=TARGET):
    """VRAM ratio at iso-hit=target: blind needs frac=target; freq needs smallest m with hit>=target."""
    m_target = next((m for m in range(1, n_slots + 1) if curve[m] >= target), n_slots)
    blind_resident = round(target * n_slots)
    return blind_resident / m_target, m_target


def agg_lru_or_opt(layers, idx, doc_id, eval_docs, m, policy="lru"):
    """Aggregate (access-weighted) hit of a DYNAMIC cache holding m rows/layer, scored on eval stream."""
    emask = np.isin(doc_id, eval_docs)
    hits = total = 0
    for L in layers:
        acc = idx[L][emask]                              # (T_eval, k) in token order
        flat = acc.reshape(-1).astype(np.int64)
        if policy == "lru":
            cache: "OrderedDict[int,None]" = OrderedDict()
            for s in flat:
                s = int(s); total += 1
                if s in cache:
                    hits += 1; cache.move_to_end(s)
                else:
                    cache[s] = None
                    if len(cache) > m:
                        cache.popitem(last=False)
        else:  # belady / OPT
            from collections import defaultdict
            import heapq
            nxt = defaultdict(list)
            for i, s in enumerate(flat):
                nxt[int(s)].append(i)
            ptr = defaultdict(int); cache = set(); heap = []
            for i, s in enumerate(flat):
                s = int(s); total += 1; ptr[s] += 1
                if s in cache:
                    hits += 1
                    nu = nxt[s][ptr[s]] if ptr[s] < len(nxt[s]) else 10**9
                    heapq.heappush(heap, (-nu, s)); continue
                if len(cache) >= m:
                    while heap:
                        negnu, victim = heapq.heappop(heap)
                        if victim in cache and -negnu == (nxt[victim][ptr[victim]] if ptr[victim] < len(nxt[victim]) else 10**9):
                            cache.discard(victim); break
                cache.add(s)
                nu = nxt[s][ptr[s]] if ptr[s] < len(nxt[s]) else 10**9
                heapq.heappush(heap, (-nu, s))
    return hits / total if total else 0.0


def main():
    layers, n_slots, k, docs, doc_id, idx, counts = load()
    m8 = round(ANCHOR_F * n_slots)
    print(f"trace: {len(layers)} layers x {n_slots} experts, top-{k}, {len(docs)} docs; anchor m={m8} ({ANCHOR_F:.0%})\n")

    # --- canonical even/odd split (the paper's numbers) ---
    cal0, ev0 = docs[::2], docs[1::2]
    hit0 = heldout_hit_at_m(layers, counts, cal0, ev0, m8)
    curve0 = heldout_curve(layers, counts, cal0, ev0, n_slots)
    ratio0, mt0 = iso_ratio(curve0, n_slots)
    print(f"[canonical even/odd split]  held-out hit@8% = {hit0:.3f}   iso-0.95 VRAM ratio = {ratio0:.2f}x (freq m={mt0})")

    # --- CV band over calib/eval partitions: enumerate all if few docs, else sample random balanced splits ---
    nd = len(docs); half = nd // 2
    n_comb = math.comb(nd, half)
    rng = np.random.default_rng(0)
    if n_comb <= MAX_ENUM:
        split_iter = ([list(cal), [d for d in docs if d not in cal]]
                      for cal in itertools.combinations(docs, half))
        n_splits, how = n_comb, f"all {n_comb} calib/eval splits"
    else:
        def _rand():
            for _ in range(N_RANDOM_SPLITS):
                p = rng.permutation(docs); yield list(p[:half]), list(p[half:])
        split_iter, n_splits, how = _rand(), N_RANDOM_SPLITS, f"{N_RANDOM_SPLITS} random {half}/{nd-half} splits"
    hits, ratios = [], []
    for cal, ev in split_iter:
        hits.append(heldout_hit_at_m(layers, counts, cal, ev, m8))
        ratios.append(iso_ratio(heldout_curve(layers, counts, cal, ev, n_slots), n_slots)[0])
    hits, ratios = np.array(hits), np.array(ratios)
    def band(a): return np.percentile(a, 2.5), np.median(a), np.percentile(a, 97.5)
    h_lo, h_md, h_hi = band(hits); r_lo, r_md, r_hi = band(ratios)
    print(f"[{how}] held-out hit@8% : median {h_md:.3f}  95% CV band [{h_lo:.3f}, {h_hi:.3f}]")
    print(f"[{how}] iso-0.95 ratio  : median {r_md:.2f}x 95% CV band [{r_lo:.2f}, {r_hi:.2f}]x")

    # --- doc-level bootstrap over the canonical eval docs ---
    B = 2000; bh = []
    for _ in range(B):
        samp = list(rng.choice(ev0, size=len(ev0), replace=True))
        bh.append(heldout_hit_at_m(layers, counts, cal0, samp, m8))
    bh = np.array(bh)
    print(f"[bootstrap eval docs n={len(ev0)}]  held-out hit@8% : 95% CI [{np.percentile(bh,2.5):.3f}, {np.percentile(bh,97.5):.3f}]\n")

    # --- dynamic-policy baselines on the held-out (eval) stream ---
    print(f"Dynamic vs static, scored on the SAME held-out eval stream ({len(ev0)} docs):")
    print(f"  {'f':>5} {'m/layer':>7} {'static-freq(held-out)':>22} {'LRU':>8} {'Belady/OPT':>11} {'blind(=f)':>10}")
    rows = [("frac", "m", "static_heldout", "lru", "belady", "blind")]
    for f in FRACS:
        m = round(f * n_slots)
        st = float(curve0[m])
        lru = agg_lru_or_opt(layers, idx, doc_id, ev0, m, "lru")
        opt = agg_lru_or_opt(layers, idx, doc_id, ev0, m, "opt")
        print(f"  {f:>5.2f} {m:>7d} {st:>22.3f} {lru:>8.3f} {opt:>11.3f} {f:>10.3f}")
        rows.append((f"{f:.2f}", m, f"{st:.4f}", f"{lru:.4f}", f"{opt:.4f}", f"{f:.4f}"))

    tag = "" if "30B" in TRACE.name else "_" + TRACE.name.replace("moe_", "").replace("_xdoc.npz", "")
    out = Path(f"data/eval/residency_cis_lru{tag}.csv")
    out.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
    print(f"\nwrote {out}")
    # machine-readable summary line for the findings doc
    print(f"\nSUMMARY hit8={hit0:.3f} cv=[{h_lo:.3f},{h_hi:.3f}] ratio={ratio0:.2f} cv=[{r_lo:.2f},{r_hi:.2f}]")


if __name__ == "__main__":
    main()
