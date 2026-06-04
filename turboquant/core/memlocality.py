"""Memory-access locality analysis for DDR5-paged knowledge stores (measure-first probe).

Implements the metrics in the paper: given a
decode-time access trace (the top-k retrieved slot indices per token, per memory layer), quantify
how *cacheable* the access pattern is and back-solve the DDR5/PCIe bandwidth-viability frontier.

The whole breakthrough hypothesis reduces to one question this module answers: **at a VRAM hot-cache
budget of B bytes, what decode-time hit-rate does the access pattern achieve — and is that enough to
keep the cold bank in DDR5 at target tok/s?** Runs offline on a real trace (from a Memory-Layer/PEER/
MoE capture) or on the synthetic generators here that bracket the three outcomes.

NOVELTY: none in this code — it's standard cache analysis (LRU/Belady) + Zipf/reuse-distance stats
applied to model memory access. The research contribution (if any) is Phase B: a *training objective*
that improves these curves. See the proposal doc.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Bandwidth anchors (GB/s) — re-measure on the target box; defaults are PCIe-5 / DDR5 ballpark.
DEFAULT_BW_GBPS = 64.0


@dataclass(frozen=True)
class AccessTrace:
    """A decode-time access trace for ONE memory layer.

    accesses: (T, k) int — the top-k slot indices retrieved at each of T tokens.
    n_slots:  total number of slots in the bank (the universe).
    slot_bytes: bytes per slot row (e.g. d_value * bytes_per_elem) — for bandwidth accounting.
    """

    accesses: NDArray[np.integer]   # (T, k)
    n_slots: int
    slot_bytes: int = 4096 * 2      # default: d=4096 value row in bf16

    @property
    def n_tokens(self) -> int:
        return self.accesses.shape[0]

    @property
    def k(self) -> int:
        return self.accesses.shape[1]


# --------------------------------------------------------------------------- #
# Locality / skew metrics                                                     #
# --------------------------------------------------------------------------- #
def access_skew(trace: AccessTrace) -> dict[str, float]:
    """How concentrated are accesses on a few hot slots? (skew enables small hot-caches)."""
    flat = trace.accesses.ravel()
    counts = np.bincount(flat, minlength=trace.n_slots).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    max_entropy = float(np.log2((counts > 0).sum())) if (counts > 0).any() else 0.0
    # Gini of the access-count distribution
    srt = np.sort(counts)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if srt.sum() else 0.0
    # fraction of total accesses captured by the hottest 10% of touched slots
    touched = counts[counts > 0]
    srt_desc = np.sort(touched)[::-1]
    top10 = srt_desc[: max(1, len(srt_desc) // 10)].sum() / srt_desc.sum() if srt_desc.sum() else 0.0
    return {
        "entropy_bits": entropy,
        "entropy_ratio": entropy / max_entropy if max_entropy else 0.0,  # 1.0 = uniform/flat
        "gini": gini,                                                    # 1.0 = maximally skewed
        "top10pct_coverage": float(top10),
        "n_unique_slots": float((counts > 0).sum()),
    }


def reuse_distances(trace: AccessTrace) -> NDArray[np.integer]:
    """Per-access reuse distance (# distinct slots seen since this slot was last accessed).

    The signal a cache actually exploits — temporal locality, not just skew. Returns -1 for
    first-touch (cold) accesses. Computed over the flattened per-token access stream.
    """
    last_seen: dict[int, int] = {}
    out = []
    seen_order: "OrderedDict[int, None]" = OrderedDict()
    pos = 0
    for t in range(trace.n_tokens):
        for slot in trace.accesses[t]:
            slot = int(slot)
            if slot in last_seen:
                # distinct slots touched since last use of `slot`
                # (approximate stack-distance via insertion order)
                seen_order.move_to_end(slot)
                # distance = number of entries after `slot` would be O(n); use a cheap proxy:
                out.append(pos - last_seen[slot])
            else:
                out.append(-1)
                seen_order[slot] = None
            last_seen[slot] = pos
            pos += 1
    return np.array(out, dtype=np.int64)


def temporal_locality(trace: AccessTrace, windows=(16, 64, 256, 1024)) -> dict[str, float]:
    """Fraction of accesses whose slot was touched within the last W token-positions."""
    rd = reuse_distances(trace)
    valid = rd[rd >= 0]
    out = {"cold_first_touch_frac": float((rd < 0).mean())}
    for w in windows:
        # window in *accesses*; multiply by k to approximate token window
        out[f"reuse_within_{w}tok"] = float((valid <= w * trace.k).mean()) if valid.size else 0.0
    return out


# --------------------------------------------------------------------------- #
# Cache simulation                                                            #
# --------------------------------------------------------------------------- #
def lru_hit_rate(trace: AccessTrace, cache_slots: int) -> float:
    """Realized hit-rate under an LRU cache holding `cache_slots` rows."""
    if cache_slots <= 0:
        return 0.0
    cache: "OrderedDict[int, None]" = OrderedDict()
    hits = total = 0
    for t in range(trace.n_tokens):
        for slot in trace.accesses[t]:
            slot = int(slot)
            total += 1
            if slot in cache:
                hits += 1
                cache.move_to_end(slot)
            else:
                cache[slot] = None
                if len(cache) > cache_slots:
                    cache.popitem(last=False)
    return hits / total if total else 0.0


def resident_fraction_for_hit(trace: AccessTrace, target_hit: float = 0.95,
                              grid: tuple[float, ...] | None = None) -> dict[str, float]:
    """Smallest VRAM-resident cache FRACTION (of n_slots) whose LRU hit-rate >= target_hit.

    THE universe-size-law metric: "how much of the bank must stay resident to hit the target?".
    A large-universe store where the per-context working set is a tiny fraction of the bank needs a
    SMALL resident fraction; a small-universe store (MoE experts, where the working set ~ the whole
    universe) needs a large one. Reported alongside n_slots so resident_fraction can be plotted vs
    universe-size / top-k across substrates (memory layers vs MoE) on one axis.

    Returns {fraction, cache_slots, achieved_hit, working_set, n_slots, k}. fraction=1.0 (full bank)
    if even that can't reach target. The scan walks a fraction grid; LRU is cache-size-specific so
    each grid point is a separate pass (cheap for MoE traces, seconds for the ~2M-access memory probe).
    """
    if grid is None:
        grid = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
                0.20, 0.30, 0.40, 0.50, 0.70, 1.0)
    n = trace.n_slots
    ws = int(len(np.unique(trace.accesses)))
    best_frac, best_c, best_hr = 1.0, n, lru_hit_rate(trace, n)
    for f in grid:
        c = max(1, int(round(f * n)))
        hr = lru_hit_rate(trace, c)
        if hr >= target_hit:
            best_frac, best_c, best_hr = f, c, hr
            break
    return {"fraction": float(best_frac), "cache_slots": int(best_c),
            "achieved_hit": float(best_hr), "working_set": ws,
            "n_slots": int(n), "k": int(trace.k)}


def belady_hit_rate(trace: AccessTrace, cache_slots: int) -> float:
    """Belady/OPT optimal hit-rate (upper bound) — evict the slot used farthest in the future."""
    if cache_slots <= 0:
        return 0.0
    flat = trace.accesses.ravel().astype(np.int64)
    # precompute next-use positions
    next_use: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(flat):
        next_use[int(s)].append(i)
    ptr: dict[int, int] = defaultdict(int)
    cache: set[int] = set()
    hits = 0
    import heapq
    for i, s in enumerate(flat):
        s = int(s)
        ptr[s] += 1  # advance past current position
        if s in cache:
            hits += 1
            continue
        if len(cache) < cache_slots:
            cache.add(s)
            continue
        # evict the cached slot whose next use is farthest (or never)
        victim, far = None, -1
        for c in cache:
            nu = next_use[c][ptr[c]] if ptr[c] < len(next_use[c]) else np.inf
            if nu > far:
                far, victim = nu, c
        cache.discard(victim)
        cache.add(s)
    return hits / len(flat) if len(flat) else 0.0


def hitrate_vs_vram(trace: AccessTrace, cache_fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                    optimal=False) -> list[dict[str, float]]:
    """The headline curve: hit-rate as a function of hot-cache size (= VRAM budget)."""
    fn = belady_hit_rate if optimal else lru_hit_rate
    rows = []
    for f in cache_fracs:
        cslots = max(1, int(f * trace.n_slots))
        hr = fn(trace, cslots)
        rows.append({
            "cache_frac": f,
            "cache_slots": cslots,
            "cache_bytes": cslots * trace.slot_bytes,
            "hit_rate": hr,
        })
    return rows


# --------------------------------------------------------------------------- #
# Bandwidth-viability frontier                                                #
# --------------------------------------------------------------------------- #
# Per-random-fetch latency over the cold link. At batch-1 decode, cache misses serialize, so
# LATENCY (not aggregate throughput) is usually the binding wall — a single ~us-scale random read
# per miss. Default ~1 us is a rough DDR5-random / small-PCIe-read figure; RE-MEASURE on target HW.
DEFAULT_FETCH_LATENCY_S = 1e-6


def viable_tok_per_s(hit_rate: float, slot_bytes: int, k: int,
                     bandwidth_gbps: float = DEFAULT_BW_GBPS,
                     fetch_latency_s: float = DEFAULT_FETCH_LATENCY_S) -> float:
    """Max decode tok/s sustainable for the cold-fetch path (batch-1, misses serialized).

    Per token: (1-hit)·k misses; each costs latency + bytes/bandwidth; they serialize at decode.
    This is the *cold-fetch ceiling only* — real tok/s is min(this, compute-bound rate). With small
    top-k rows the throughput term is tiny, so the LATENCY term dominates — which is the honest wall
    the throughput-only model missed.
    """
    misses = (1.0 - hit_rate) * k
    if misses <= 0:
        return float("inf")
    per_token_s = misses * (fetch_latency_s + slot_bytes / (bandwidth_gbps * 1e9))
    return 1.0 / per_token_s if per_token_s > 0 else float("inf")


def viability_frontier(trace: AccessTrace, target_tok_per_s: float = 30.0,
                       bandwidth_gbps: float = DEFAULT_BW_GBPS,
                       fetch_latency_s: float = DEFAULT_FETCH_LATENCY_S,
                       hitrate_target: float = 0.9,
                       cache_fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5)) -> list[dict[str, float]]:
    """For each VRAM budget: hit-rate, cold-fetch ceiling tok/s, miss traffic, and viability.

    Viability uses TWO honest criteria (both must hold): (a) the cold-fetch ceiling clears the
    target tok/s, and (b) the hit-rate clears `hitrate_target` — the robust, hardware-independent
    signal that the access pattern is actually cacheable rather than the absolute tok/s (which is
    dominated by guessed latency/bandwidth constants).
    """
    rows = hitrate_vs_vram(trace, cache_fracs, optimal=False)
    for r in rows:
        tps = viable_tok_per_s(r["hit_rate"], trace.slot_bytes, trace.k,
                               bandwidth_gbps, fetch_latency_s)
        r["max_tok_per_s"] = tps
        r["miss_GBps_at_target"] = ((1.0 - r["hit_rate"]) * trace.k * trace.slot_bytes
                                    * target_tok_per_s) / 1e9
        r["viable_at_target"] = (tps >= target_tok_per_s) and (r["hit_rate"] >= hitrate_target)
    return rows


# --------------------------------------------------------------------------- #
# Synthetic trace generators (bracket the three outcomes; validate the analysis) #
# --------------------------------------------------------------------------- #
def synth_zipfian(n_tokens=8000, n_slots=50_000, k=8, p_hot=0.9, hot_frac=0.01,
                  seed=0) -> AccessTrace:
    """Cache-LOCAL outcome: a small hot pool absorbs most accesses (genuinely cacheable).

    NOTE (a real finding the probe surfaced): pure *iid* Zipfian *skew* does NOT yield cacheability
    at LLM-memory scale — you need a concentrated hot set actually re-hit within the eviction window.
    This generator models that (fraction p_hot of accesses from a hot_frac pool), which is the honest
    "cache-local" representative; skew alone (see access_skew) is necessary but not sufficient.
    """
    rng = np.random.default_rng(seed)
    n_hot = max(1, int(hot_frac * n_slots))
    hot = rng.random((n_tokens, k)) < p_hot
    acc = np.where(hot,
                   rng.integers(0, n_hot, size=(n_tokens, k)),
                   rng.integers(0, n_slots, size=(n_tokens, k)))
    return AccessTrace(acc.astype(np.int64), n_slots)


def synth_uniform(n_tokens=4000, n_slots=100_000, k=8, seed=0) -> AccessTrace:
    """Cache-HOSTILE outcome: flat/load-balanced -> no working set, paging infeasible."""
    rng = np.random.default_rng(seed)
    acc = rng.integers(0, n_slots, size=(n_tokens, k))
    return AccessTrace(acc, n_slots)


def synth_bursty(n_tokens=4000, n_slots=100_000, k=8, block=64, hot=2000, seed=0) -> AccessTrace:
    """MIXED outcome: temporally-local bursts over a rotating hot-set (realistic-ish)."""
    rng = np.random.default_rng(seed)
    # Hot-set must be smaller than the universe and leave room to slide the window.
    hot = max(k, min(hot, n_slots // 2))
    acc = np.empty((n_tokens, k), dtype=np.int64)
    for start in range(0, n_tokens, block):
        base = int(rng.integers(0, n_slots - hot + 1))
        for t in range(start, min(start + block, n_tokens)):
            acc[t] = rng.integers(base, base + hot, size=k)
    return AccessTrace(acc, n_slots)


def load_trace(path: str, layer: int = 0, slot_bytes: int = 4096 * 2) -> AccessTrace:
    """Load a captured trace .npz (arrays 'L{layer}_idx' shape (T,k); 'n_slots' scalar)."""
    data = np.load(path)
    acc = data[f"L{layer}_idx"]
    n_slots = int(data["n_slots"]) if "n_slots" in data else int(acc.max() + 1)
    return AccessTrace(acc, n_slots, slot_bytes=slot_bytes)


def list_trace_layers(path: str) -> list[int]:
    """Return the sorted layer indices present in a captured trace .npz ('L{idx}_idx' keys)."""
    import re
    data = np.load(path)
    layers = []
    for key in data.files:
        m = re.fullmatch(r"L(\d+)_idx", key)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


# --------------------------------------------------------------------------- #
# Static-frequency residency curve (the DEPLOYABLE policy, measured per-layer) #
# --------------------------------------------------------------------------- #
def static_frequency_curve(path: str, n_points: int = 240) -> dict[str, object]:
    """Aggregate hit-rate vs per-layer resident fraction for STATIC-FREQUENCY residency, measured.

    THE deployable policy (Tier-2 in the paper): per layer, keep the top-f fraction of THAT layer's
    experts ranked by measured access frequency. This is a per-layer placement (each MoE layer has
    its own routed-expert tensor; a residency map is naturally per-layer), NOT a single global
    top-f across the union of all layers' experts. We therefore aggregate per-layer.

    --- per-layer aggregation (the choice, stated honestly) ----------------------------------------
    For a per-layer resident fraction f, each layer L keeps its hottest m_L = round(f * n_slots)
    experts. The accesses that layer L's resident set captures = (sum of the m_L largest expert
    access-counts in L). The AGGREGATE hit-rate at fraction f is:

        agg_hit(f) = sum_L (captured accesses in L) / sum_L (total accesses in L)

    i.e. an access-WEIGHTED average of per-layer hit-rates (layers with more tokens/accesses count
    proportionally more). All MoE layers see the same token stream here, so this is just the global
    captured/total ratio under a per-layer top-m_L mask. We use a SHARED per-layer fraction f across
    layers (one VRAM knob) rather than a per-layer optimized budget — that matches a simple
    deployable "keep top-f% of every layer" placement and is the conservative choice.

    This is a STATIC-frequency (counting) policy, not LRU and not Belady: residency is fixed offline
    from the frequency ranking, exactly the deployable artifact the paper describes.

    Returns a dict with arrays over a per-layer-fraction grid f in [0, 1]:
        frac          : (P,) per-layer resident fraction f
        resident_per_layer : (P,) int m_L = round(f * n_slots) experts kept PER layer
        resident_total     : (P,) int total resident experts = n_layers * m_L (the VRAM count)
        agg_hit       : (P,) measured aggregate (access-weighted) hit-rate of static-frequency
        blind_hit     : (P,) = m_L / n_slots  (frequency-blind control: random f-fraction per layer)
    and scalars:
        n_slots, n_layers, k, total_accesses,
        s_fit         : Zipf exponent fit on the pooled per-layer-normalized counts (see note),
        hit_at_8pct   : measured aggregate hit when the hottest 8% of each layer is resident,
        per_layer_counts : (n_layers, n_slots) sorted-descending access counts (for re-use/plots).
    """
    layers = list_trace_layers(path)
    if not layers:
        raise ValueError(f"no 'L{{idx}}_idx' arrays in {path}")
    data = np.load(path)
    n_slots = int(data["n_slots"]) if "n_slots" in data.files else None
    k = None
    # Held-out calib/eval token split. If the trace carries a doc_id (multi-document capture), split by
    # DOCUMENT (calib = even-indexed docs, eval = the disjoint odd-indexed docs) -> a CROSS-DOCUMENT
    # generalization test. Else fall back to a within-sequence first-half/second-half split.
    doc_id = data["doc_id"] if "doc_id" in data.files else None
    holdout_kind = "within-window 50/50"
    calib_mask = None
    if doc_id is not None and np.unique(doc_id).size >= 2:
        docs = np.unique(doc_id)
        calib_docs = set(docs[::2].tolist())              # interleave so both halves span the topic range
        calib_mask = np.isin(doc_id, list(calib_docs))
        holdout_kind = f"CROSS-DOCUMENT ({docs.size} docs: {len(calib_docs)} calib / {docs.size-len(calib_docs)} eval)"

    # per-layer access-count vectors, length n_slots; plus a calib/eval token split for OUT-OF-SAMPLE.
    counts_rows, calib_rows, eval_rows = [], [], []
    for L in layers:
        idx = data[f"L{L}_idx"]                           # (T, k) per-token selected experts
        if k is None:
            k = int(idx.shape[1])
        if n_slots is None:
            n_slots = int(idx.max() + 1)
        counts_rows.append(np.bincount(idx.reshape(-1), minlength=n_slots)[:n_slots].astype(np.float64))
        cmask = calib_mask if calib_mask is not None else (np.arange(idx.shape[0]) < max(idx.shape[0] // 2, 1))
        calib_rows.append(np.bincount(idx[cmask].reshape(-1), minlength=n_slots)[:n_slots].astype(np.float64))
        eval_rows.append(np.bincount(idx[~cmask].reshape(-1), minlength=n_slots)[:n_slots].astype(np.float64))
    counts = np.vstack(counts_rows)                       # (n_layers, n_slots), per-layer counts
    calib = np.vstack(calib_rows)
    evalc = np.vstack(eval_rows)
    n_layers = counts.shape[0]
    # sort each layer's counts descending -> cumulative capture as we add the next-hottest expert
    sorted_desc = np.sort(counts, axis=1)[:, ::-1]        # (n_layers, n_slots)
    cum = np.cumsum(sorted_desc, axis=1)                  # cum[L, m-1] = accesses captured by top-m in L
    layer_totals = counts.sum(axis=1)                     # (n_layers,)
    total_accesses = float(layer_totals.sum())

    def agg_hit_for_m(m: int) -> float:
        m = int(min(max(m, 0), n_slots))
        if m == 0:
            return 0.0
        captured = cum[:, m - 1].sum()                    # sum over layers of top-m captured
        return float(captured / total_accesses) if total_accesses else 0.0

    # per-layer-fraction grid (m_L over 0..n_slots), mapped to fractions
    m_grid = np.unique(np.clip(np.round(np.linspace(0, n_slots, n_points)).astype(int), 0, n_slots))
    frac = m_grid / n_slots
    agg_hit = np.array([agg_hit_for_m(m) for m in m_grid])
    blind_hit = m_grid / n_slots                          # random f-fraction per layer captures f
    resident_total = m_grid * n_layers

    # measured "hottest 8% of each layer" aggregate hit (the anchor the analytic fit targeted)
    m8 = int(round(0.08 * n_slots))
    hit_at_8pct = agg_hit_for_m(m8)

    # --- OUT-OF-SAMPLE (deployable) curve --------------------------------------------------------
    # Choose each layer's resident set from CALIB frequency, score captured accesses on HELD-OUT eval.
    # If the in-sample agg_hit is real per-layer skew it survives here; if it is selection bias /
    # short-window topical concentration it collapses toward the blind line (= f).
    calib_order = np.argsort(calib, axis=1)[:, ::-1]              # experts by calib freq, per layer
    eval_by_calib_rank = np.take_along_axis(evalc, calib_order, axis=1)
    cum_eval = np.cumsum(eval_by_calib_rank, axis=1)             # held-out accesses on calib-top-m
    eval_total = float(evalc.sum())

    def agg_hit_oos_for_m(m: int) -> float:
        m = int(min(max(m, 0), n_slots))
        if m == 0 or not eval_total:
            return 0.0
        return float(cum_eval[:, m - 1].sum() / eval_total)

    agg_hit_oos = np.array([agg_hit_oos_for_m(m) for m in m_grid])
    hit_at_8pct_oos = agg_hit_oos_for_m(m8)

    # Zipf s_fit: pool per-layer-normalized frequencies (so layers with different token counts are
    # comparable), then log-log slope of the sorted rank-frequency tail. Same estimator the capture
    # script reports as a global convenience; recomputed here from the per-layer-normalized counts.
    p_rows = sorted_desc / np.maximum(layer_totals[:, None], 1.0)   # per-layer normalized, sorted
    mean_p = p_rows.mean(axis=0)                           # average rank-frequency profile
    nz = mean_p[mean_p > 0]
    if nz.size >= 8:
        ranks = np.arange(1, nz.size + 1)
        s_fit = float(-np.polyfit(np.log(ranks), np.log(nz / nz.sum()), 1)[0])
    else:
        s_fit = float("nan")

    return {
        "frac": frac,
        "resident_per_layer": m_grid,
        "resident_total": resident_total,
        "agg_hit": agg_hit,
        "blind_hit": blind_hit,
        "n_slots": int(n_slots),
        "n_layers": int(n_layers),
        "k": int(k),
        "total_accesses": total_accesses,
        "s_fit": s_fit,
        "hit_at_8pct": float(hit_at_8pct),
        "agg_hit_oos": agg_hit_oos,
        "hit_at_8pct_oos": float(hit_at_8pct_oos),
        "holdout_kind": holdout_kind,
        "per_layer_counts": sorted_desc,
        "_agg_hit_for_m": agg_hit_for_m,
        "_agg_hit_oos_for_m": agg_hit_oos_for_m,
    }
