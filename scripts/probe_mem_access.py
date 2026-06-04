#!/usr/bin/env python
"""Probe memory-access locality: is a paged-knowledge store cacheable enough for DDR5 offload?

The measure-first gate from the paper. Runs on a
captured access trace (.npz from a Memory-Layer/PEER/MoE model) or on --synthetic regimes that bracket
the three outcomes. Prints the headline hit-rate-vs-VRAM curve, the bandwidth-viability frontier, the
load-balance-vs-locality signal, and a GO/NO-GO verdict for the locality-training thesis.

    uv run python scripts/probe_mem_access.py --synthetic            # all three regimes
    uv run python scripts/probe_mem_access.py --synthetic uniform    # one regime
    uv run python scripts/probe_mem_access.py --trace logs/mem.npz --layer 0 --target-tok-s 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turboquant.core.memlocality import (  # noqa: E402
    DEFAULT_FETCH_LATENCY_S,
    access_skew,
    list_trace_layers,
    load_trace,
    lru_hit_rate,
    resident_fraction_for_hit,
    synth_bursty,
    synth_uniform,
    synth_zipfian,
    temporal_locality,
    viability_frontier,
)

SYNTH = {"zipfian": synth_zipfian, "uniform": synth_uniform, "bursty": synth_bursty}


def report(name, trace, target_tok_s, bandwidth, hitrate_target=0.9, target_hit=0.95):
    print(f"\n=== {name} ===  (T={trace.n_tokens}, k={trace.k}, n_slots={trace.n_slots}, "
          f"slot={trace.slot_bytes}B)")
    skew = access_skew(trace)
    loc = temporal_locality(trace)
    print(f"  skew: entropy_ratio={skew['entropy_ratio']:.3f} (1=flat)  gini={skew['gini']:.3f} "
          f"(1=skewed)  top10%-coverage={skew['top10pct_coverage']:.3f}")
    print(f"  temporal: cold-first-touch={loc['cold_first_touch_frac']:.3f}  "
          f"reuse<=64tok={loc['reuse_within_64tok']:.3f}  reuse<=1024tok={loc['reuse_within_1024tok']:.3f}")
    print(f"  hit-rate vs VRAM budget (LRU). Viable = hit-rate>={hitrate_target:.2f} AND cold-fetch "
          f"ceiling>={target_tok_s:.0f} tok/s @ {bandwidth:.0f} GB/s, {DEFAULT_FETCH_LATENCY_S*1e6:.0f}us/fetch:")
    print(f"    {'cacheFrac':>9} {'cacheMB':>9} {'hitRate':>8} {'missGB/s':>9} {'ceilTok/s':>10} {'viable':>7}")
    frontier = viability_frontier(trace, target_tok_s, bandwidth, hitrate_target=hitrate_target)
    hr_by_frac = {}
    for r in frontier:
        tps = r["max_tok_per_s"]
        tps_s = "inf" if tps == float("inf") else f"{tps:.0f}"
        mark = "YES" if r["viable_at_target"] else "no"
        hr_by_frac[r["cache_frac"]] = r["hit_rate"]
        print(f"    {r['cache_frac']:>9.2f} {r['cache_bytes']/1e6:>9.1f} {r['hit_rate']:>8.3f} "
              f"{r['miss_GBps_at_target']:>9.2f} {tps_s:>10} {mark:>7}")
    # universe-size-law metric: smallest resident fraction reaching the target hit (default 0.95).
    rf = resident_fraction_for_hit(trace, target_hit)
    print(f"  UNIVERSE-LAW: universe(n_slots)={rf['n_slots']}  k={trace.k}  "
          f"universe/k={rf['n_slots']/max(1,trace.k):.0f}  working_set={rf['working_set']}  "
          f"rf@{target_hit:.2f}={rf['fraction']:.3f} ({rf['cache_slots']} slots, hit {rf['achieved_hit']:.3f})  "
          f"[resident fraction for {target_hit:.0%} hit; plot vs universe/k]")
    # Verdict keys off the CURVE SHAPE (hardware-independent), not absolute tok/s (constant-sensitive).
    hr_small = hr_by_frac.get(0.05, 0.0)          # hit-rate at a small (~5%) VRAM cache
    hr_big = max(hr_by_frac.values())             # best achievable hit-rate
    print(f"  VERDICT (hit-rate @5%-cache={hr_small:.2f}, best={hr_big:.2f}):")
    if hr_small >= 0.6:
        print("    [ALREADY-LOCAL] a small cache already absorbs most accesses -> DDR5 paging viable "
              "TODAY; a systems result, not a training-objective paper.")
    elif hr_big < 0.3:
        print("    [GAP] even a large cache can't absorb the accesses -> cache-HOSTILE as captured. "
              "The locality-training thesis is LIVE (Phase B): train for the working set.")
    else:
        print("    [PARTIAL] cacheable only with a large cache -> a locality objective that lifts the "
              "small-cache hit-rate is worth measuring (Phase B, quantified gap).")
    return skew, loc, frontier


def classify(hr_small, hr_big):
    if hr_small >= 0.6:
        return "ALREADY-LOCAL"
    if hr_big < 0.3:
        return "GAP"
    return "PARTIAL"


def sweep_layers(path, slot_bytes, target_tok_s, bandwidth, hitrate_target, target_hit=0.95):
    """One-line-per-layer summary across all captured layers (find where the working set lives).

    The verdict uses a k-aware 'small cache': a cache holding ~2x the per-token fetch (2*k slots).
    At small universes (MoE experts) a fixed 5%-of-universe cache can be < k, making hr@small
    structurally 0 and the verdict meaningless — 2*k is the honest 'is there a hot-set' probe.
    """
    layers = list_trace_layers(path)
    if not layers:
        print(f"No 'L{{idx}}_idx' arrays in {path}", file=sys.stderr)
        return 1
    print(f"\n=== per-layer sweep: {Path(path).name} ({len(layers)} layers) ===")
    print(f"  {'layer':>5} {'entropyR':>9} {'gini':>6} {'reuse64':>8} "
          f"{'hrSmall':>8} {'hr@50%':>7} {f'rf@{target_hit:.2f}':>8} {'verdict':>14}")
    agg = {"ALREADY-LOCAL": 0, "PARTIAL": 0, "GAP": 0}
    rfracs = []
    n_slots = k = ws = 0
    for L in layers:
        tr = load_trace(path, L, slot_bytes)
        skew = access_skew(tr)
        loc = temporal_locality(tr)
        # small cache = max(2*k, 1% of universe); big cache = 50% of universe
        small_slots = max(2 * tr.k, int(0.01 * tr.n_slots))
        big_slots = int(0.5 * tr.n_slots)
        hr_small = lru_hit_rate(tr, small_slots)
        hr_big = lru_hit_rate(tr, big_slots)
        rf = resident_fraction_for_hit(tr, target_hit)   # universe-size-law y-axis
        rfracs.append(rf["fraction"]); n_slots, k, ws = rf["n_slots"], rf["k"], rf["working_set"]
        v = classify(hr_small, hr_big)
        agg[v] += 1
        print(f"  {L:>5} {skew['entropy_ratio']:>9.3f} {skew['gini']:>6.3f} "
              f"{loc['reuse_within_64tok']:>8.3f} {hr_small:>8.3f} {hr_big:>7.3f} "
              f"{rf['fraction']:>8.3f} {v:>14}")
    med_rf = float(np.median(rfracs)) if rfracs else 1.0
    print(f"\n  SUMMARY: {agg['ALREADY-LOCAL']} already-local, {agg['PARTIAL']} partial, "
          f"{agg['GAP']} gap (of {len(layers)} layers)")
    # the universe-size-law row: median resident fraction for the target hit, with the axis variables.
    print(f"  UNIVERSE-LAW: n_slots(universe)={n_slots}  k={k}  universe/k={n_slots/max(1,k):.0f}  "
          f"median rf@{target_hit:.2f}={med_rf:.3f}  (resident fraction for {target_hit:.0%} hit; "
          f"smaller=pages better)")
    print(f"  hrSmall = hit-rate at max(2*k, 1% universe); rf@{target_hit:.2f} = smallest resident "
          f"fraction reaching {target_hit:.0%} LRU hit (THE universe-law metric — plot vs universe/k\n"
          f"  across substrates). n_slots small (MoE) => SELECTION-locality, not the large-bank test.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="captured access .npz (L{layer}_idx, n_slots)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--all-layers", action="store_true",
                    help="sweep every captured layer (compact per-layer summary table)")
    ap.add_argument("--synthetic", nargs="?", const="all",
                    help="run synthetic regime(s): all|zipfian|uniform|bursty")
    ap.add_argument("--target-tok-s", type=float, default=30.0)
    ap.add_argument("--bandwidth-gbps", type=float, default=64.0)
    ap.add_argument("--hitrate-target", type=float, default=0.9,
                    help="hit-rate a VRAM budget must reach to count as 'viable' (robust signal)")
    ap.add_argument("--target-hit", type=float, default=0.95,
                    help="universe-law metric: target LRU hit for the smallest-resident-fraction search "
                         "(rf@T). Plot rf@T vs universe/k across substrates (memory layers vs MoE).")
    ap.add_argument("--slot-bytes", type=int, default=4096 * 2)
    args = ap.parse_args(argv)

    if args.trace and args.all_layers:
        return sweep_layers(args.trace, args.slot_bytes, args.target_tok_s,
                            args.bandwidth_gbps, args.hitrate_target, args.target_hit)
    elif args.trace:
        trace = load_trace(args.trace, args.layer, args.slot_bytes)
        report(f"trace:{Path(args.trace).name}#L{args.layer}", trace, args.target_tok_s,
               args.bandwidth_gbps, args.hitrate_target, args.target_hit)
    elif args.synthetic:
        regimes = SYNTH if args.synthetic == "all" else {args.synthetic: SYNTH[args.synthetic]}
        for name, gen in regimes.items():
            report(f"synthetic:{name}", gen(), args.target_tok_s, args.bandwidth_gbps,
                   args.hitrate_target, args.target_hit)
        print("\nNOTE: synthetic validates the analysis + the three outcome brackets. The real "
              "verdict needs a captured trace from a Memory-Layer/PEER/MoE model "
              "(scripts/capture_mem_access.py, GPU).")
    else:
        ap.error("pass --trace <file> or --synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
