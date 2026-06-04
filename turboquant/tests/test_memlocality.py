"""Tests for memory-access locality analysis (memlocality.py).

Validates the cache simulators and metrics against known-answer cases and confirms the three
synthetic regimes bracket the outcomes (Zipfian cacheable, uniform hostile, bursty in-between).
"""

import numpy as np
import pytest

from turboquant.core.memlocality import (
    AccessTrace,
    access_skew,
    belady_hit_rate,
    hitrate_vs_vram,
    lru_hit_rate,
    synth_bursty,
    synth_uniform,
    synth_zipfian,
    temporal_locality,
    viability_frontier,
    viable_tok_per_s,
)


class TestCacheKnownAnswers:
    def test_lru_all_hits_when_cache_covers_universe(self):
        # 3 slots, cache of 3 -> after first touch everything hits
        acc = np.array([[0], [1], [2], [0], [1], [2]])
        tr = AccessTrace(acc, n_slots=3)
        # 3 cold + 3 hits = 0.5
        assert lru_hit_rate(tr, 3) == pytest.approx(0.5)

    def test_lru_zero_when_no_cache(self):
        tr = AccessTrace(np.array([[0], [1], [0]]), n_slots=2)
        assert lru_hit_rate(tr, 0) == 0.0

    def test_lru_thrash_pattern(self):
        # cache size 1, alternating 0,1,0,1 -> every access after first is a miss
        acc = np.array([[0], [1], [0], [1]])
        tr = AccessTrace(acc, n_slots=2)
        assert lru_hit_rate(tr, 1) == 0.0

    def test_belady_at_least_lru(self):
        tr = synth_bursty(n_tokens=500, n_slots=2000, seed=1)
        for cslots in (50, 200):
            assert belady_hit_rate(tr, cslots) >= lru_hit_rate(tr, cslots) - 1e-9

    def test_hitrate_monotonic_in_cache(self):
        tr = synth_zipfian(n_tokens=1000, n_slots=5000, seed=2)
        rows = hitrate_vs_vram(tr, cache_fracs=(0.01, 0.05, 0.2, 0.5))
        hrs = [r["hit_rate"] for r in rows]
        assert all(hrs[i] <= hrs[i + 1] + 1e-9 for i in range(len(hrs) - 1))

    @pytest.mark.parametrize("gen", [synth_zipfian, synth_uniform, synth_bursty])
    def test_lru_monotonic_at_default_params(self, gen):
        # LRU is a stack algorithm -> hit-rate must be non-decreasing in cache size, at the
        # ACTUAL default generator params used by the probe (regression guard for the curve).
        tr = gen()
        hrs = [lru_hit_rate(tr, int(f * tr.n_slots)) for f in (0.01, 0.05, 0.1, 0.2, 0.5)]
        assert all(hrs[i] <= hrs[i + 1] + 1e-9 for i in range(len(hrs) - 1)), hrs

    def test_generator_values_in_range(self):
        # Guard against out-of-range/garbage slot ids (caught a stale-bytecode corruption).
        for gen in (synth_zipfian, synth_uniform, synth_bursty):
            tr = gen()
            assert tr.accesses.min() >= 0
            assert tr.accesses.max() < tr.n_slots


class TestBracketsOutcomes:
    """The three synthetic regimes must produce the three distinct verdicts."""

    def test_zipfian_more_cacheable_than_uniform(self):
        z = synth_zipfian(n_tokens=3000, n_slots=50_000, seed=3)
        u = synth_uniform(n_tokens=3000, n_slots=50_000, seed=3)
        # at a 5% cache, zipfian hit-rate should dominate uniform
        zr = lru_hit_rate(z, int(0.05 * z.n_slots))
        ur = lru_hit_rate(u, int(0.05 * u.n_slots))
        assert zr > ur + 0.1

    def test_uniform_is_hostile(self):
        u = synth_uniform(n_tokens=3000, n_slots=50_000, seed=4)
        # uniform over a large universe -> low hit-rate even at 20% cache
        assert lru_hit_rate(u, int(0.2 * u.n_slots)) < 0.5

    def test_bursty_has_temporal_locality(self):
        b = synth_bursty(n_tokens=3000, n_slots=50_000, seed=5)
        u = synth_uniform(n_tokens=3000, n_slots=50_000, seed=5)
        lb = temporal_locality(b)["reuse_within_64tok"]
        lu = temporal_locality(u)["reuse_within_64tok"]
        assert lb > lu


class TestSkewMetrics:
    def test_uniform_high_entropy_low_gini(self):
        u = synth_uniform(n_tokens=2000, n_slots=10_000, seed=6)
        s = access_skew(u)
        assert s["entropy_ratio"] > 0.9   # near-flat
        assert s["top10pct_coverage"] < 0.3

    def test_zipfian_low_entropy_high_coverage(self):
        z = synth_zipfian(n_tokens=2000, n_slots=10_000, seed=7)
        s = access_skew(z)
        assert s["top10pct_coverage"] > 0.5   # hot slots dominate
        assert s["gini"] > access_skew(synth_uniform(2000, 10_000, seed=7))["gini"]


class TestBandwidthFrontier:
    def test_viable_tok_per_s_scales_with_hitrate(self):
        # higher hit-rate -> fewer miss bytes -> more tok/s
        lo = viable_tok_per_s(0.5, slot_bytes=8192, k=8, bandwidth_gbps=64)
        hi = viable_tok_per_s(0.9, slot_bytes=8192, k=8, bandwidth_gbps=64)
        assert hi > lo

    def test_perfect_hitrate_is_infinite(self):
        assert viable_tok_per_s(1.0, 8192, 8) == float("inf")

    def test_frontier_marks_viability(self):
        z = synth_zipfian(n_tokens=2000, n_slots=20_000, seed=8)
        rows = viability_frontier(z, target_tok_per_s=10.0, bandwidth_gbps=64.0)
        assert all("viable_at_target" in r and "max_tok_per_s" in r for r in rows)
        # viability is monotonic: once viable at a cache size, viable at larger ones
        viables = [r["viable_at_target"] for r in rows]
        first = next((i for i, v in enumerate(viables) if v), None)
        if first is not None:
            assert all(viables[first:])


class TestLayerListing:
    def test_list_trace_layers(self, tmp_path):
        from turboquant.core.memlocality import list_trace_layers
        p = str(tmp_path / "t.npz")
        np.savez_compressed(p, L0_idx=np.zeros((4, 2), dtype=np.int32),
                            L3_idx=np.zeros((4, 2), dtype=np.int32),
                            L11_idx=np.zeros((4, 2), dtype=np.int32),
                            n_slots=np.asarray(64))
        assert list_trace_layers(p) == [0, 3, 11]   # sorted, ignores n_slots

    def test_empty_trace(self, tmp_path):
        from turboquant.core.memlocality import list_trace_layers
        p = str(tmp_path / "e.npz")
        np.savez_compressed(p, n_slots=np.asarray(8))
        assert list_trace_layers(p) == []


def test_resident_fraction_for_hit_skew_vs_uniform():
    """The universe-size-law metric: a skewed trace must reach a target hit at a SMALLER resident
    fraction than a uniform one (pages better). Reachable target so neither is capped by cold misses."""
    import numpy as np
    from turboquant.core.memlocality import AccessTrace, resident_fraction_for_hit
    rng = np.random.default_rng(0)
    n_slots, T, k = 2000, 20000, 4
    # skewed: ~80% of mass on the first 5% of slots (Zipf-ish hot set)
    hot = int(0.05 * n_slots)
    sk = np.where(rng.random((T, k)) < 0.8, rng.integers(0, hot, (T, k)),
                  rng.integers(0, n_slots, (T, k))).astype(np.int32)
    un = rng.integers(0, n_slots, (T, k)).astype(np.int32)
    rf_sk = resident_fraction_for_hit(AccessTrace(sk, n_slots), 0.70)
    rf_un = resident_fraction_for_hit(AccessTrace(un, n_slots), 0.70)
    assert 0.0 < rf_sk["fraction"] <= 1.0 and 0.0 < rf_un["fraction"] <= 1.0
    assert rf_sk["fraction"] < rf_un["fraction"], (rf_sk["fraction"], rf_un["fraction"])
    assert rf_sk["n_slots"] == n_slots and rf_sk["k"] == k
