#!/usr/bin/env python
"""Generate the paper's result figures from the measured Track-A / Gate-1 numbers.

Writes figures/paper_*.png (and they're copied into figures/ by the caller):
  paper_hitrate_curve.png    — free locality: hit-rate vs VRAM cache, three bank sizes (universe effect).
  paper_deltalm_arch.png     — load-bearing prerequisite: ΔLM by architecture (all-memory makes it used).
  paper_vram_isoquality.png  — Tier-1 headline: iso-quality VRAM, frequency-aware vs frequency-BLIND
                               expert residency on Qwen3-235B (the VRAM-axis discovery; see the paper).
  paper_negative.png         — the negative: vanilla vs locality-loss (hit@5% and ΔLM both drop).

Numbers are the measured values from the runs (see the paper). Edit here if a run
updates. Style matches make_fundamentals_partII_figures.py.

    uv pip install matplotlib
    uv run python scripts/make_paper_figures.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "figure.autolayout": True})
INK, HOT, COLD, ACC, GOLD = "#1b2838", "#d1495b", "#9fb3c8", "#2e86ab", "#e09f3e"


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# Free locality — hit-rate vs cache fraction, three bank sizes (the universe effect, real data)
def hitrate_curve():
    frac = [1, 5, 10, 20, 50]
    curves = [
        ("65k slots (262M)",  [0.507, 0.804, 0.920, 0.982, 0.985], COLD, "o"),
        ("262k slots (296M)", [0.559, 0.795, 0.904, 0.977, 0.983], ACC, "s"),
        ("1.05M slots (1.1B)", [0.878, 0.990, 0.992, 0.992, 0.992], HOT, "D"),  # 8x-confirmed
    ]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, h, c, m in curves:
        ax.plot(frac, h, color=c, lw=2.2, marker=m, ms=7, mec="white", label=name)
    ax.axhline(0.95, ls=":", color="#666"); ax.text(22, 0.96, "95% target", color="#666", fontsize=9)
    ax.scatter([5], [0.990], s=160, facecolor="none", edgecolor=HOT, lw=2, zorder=5)
    ax.annotate("5% cache → 99% hit\n(rf@0.95 = 3%)", (5, 0.990), textcoords="offset points",
                xytext=(14, -34), fontsize=9, weight="bold", color=HOT)
    ax.set_xscale("log"); ax.set_xlabel("VRAM cache size (% of bank)")
    ax.set_ylabel("LRU hit-rate"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Free locality: a load-bearing memory is cache-local (larger bank → pages better)")
    ax.set_xticks(frac); ax.set_xticklabels([f"{f}%" for f in frac]); ax.legend(loc="lower right")
    save(fig, "paper_hitrate_curve.png")


# Load-bearing prerequisite — ΔLM by architecture
def deltalm_arch():
    labels = ["4L, FFN×4\n1 mem layer", "4L, FFN=0\n1 mem layer", "2L, FFN=0\n1 mem layer",
              "6L all-memory\n(no FFN)"]
    vals = [0.001, 0.117, 0.166, 0.75]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=[COLD, COLD, COLD, HOT], edgecolor="white")
    ax.axhline(0.5, ls="--", color="#b00020", lw=1.5)
    ax.text(2.4, 0.53, "load-bearing (ΔLM>0.5)", color="#b00020", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}" if v < 0.5 else f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("ΔLM  (LM-loss increase when memory is zeroed)")
    ax.set_title("Memory is load-bearing only when it is the ONLY non-attention path")
    ax.set_ylim(0, 0.9)
    save(fig, "paper_deltalm_arch.png")


# ---------------------------------------------------------------------------
# Tier-1 headline: iso-quality VRAM — frequency-aware vs frequency-BLIND residency
# ---------------------------------------------------------------------------
# Measurement substrate: Qwen3-235B-A22B routed-expert universe (the 235B preset in
# scripts/paging_tok_s_model.py: total=235e9, expert_share=0.92, n_moe_layers=94, n_experts=128).
#
#   --- byte-budget arithmetic (shown so the GB axis is auditable) -----------------
#   U  = n_moe_layers * n_experts          = 94 * 128            = 12,032 expert slots (the universe)
#   routed-expert params = total * share   = 235e9 * 0.92        = 216.2 B params  (the pageable mass)
#   per-expert params = routed / U         = 216.2e9 / 12,032    = 17.97 M params/expert
#   Q3_K_M ≈ 3.5 bits/param  -> per-expert bytes = 17.97e6 * 3.5/8 = 7.86 MB/expert
#   full expert pool         = U * per-expert bytes              = 94.6 GB  (matches the MEASURED
#                              UD-Q3_K_XL 96.59 GiB static baseline once non-expert weights are added —
#                              cross-check that the per-expert byte size is right).
#   resident VRAM(GB) for m resident experts = m * 7.86 MB.
#
# THE ANCHOR (measured, real, from the 235B-scale locality analysis / the paper):
#   at ~8% of experts resident (m=962 -> 7.56 GB) the frequency ranking captures hit ≈ 0.47, while a
#   frequency-BLIND random m-subset of the SAME size captures only m/U ≈ 0.08  (~6× hit-rate at equal VRAM).
#   No 235B *trace* file exists in this checkout, so the static-frequency curve is an ANALYTIC Zipf fit:
#   we choose the Zipf exponent s so the cumulative-frequency hit passes through that measured anchor.
#   Fitted s = 0.73 (cum_hit(top-8%-of-U, s=0.73) = 0.47). Labeled "analytic (Zipf s=0.73)" on the plot.
def vram_isoquality(trace_path: str | None = None):
    """Dispatch: MEASURED mode if a real routing trace is given (CLI arg or $TQ_MOE_TRACE),
    else the analytic Zipf s=0.73 fallback (unchanged). See the paper + the per-layer aggregation
    note in turboquant.core.memlocality.static_frequency_curve."""
    trace = trace_path or os.environ.get("TQ_MOE_TRACE")
    if trace:
        p = Path(trace)
        if not p.exists():
            print(f"  ! TQ_MOE_TRACE / --moe-trace = {p} not found; falling back to analytic Zipf")
        else:
            return _vram_isoquality_measured(str(p))
    return _vram_isoquality_analytic()


def _vram_isoquality_analytic():
    # --- preset constants (mirror scripts/paging_tok_s_model.py 'qwen3-235b') ---
    TOTAL, SHARE, N_MOE, N_EXP = 235e9, 0.92, 94, 128
    BITS_PER_PARAM = 3.5                                   # Q3_K_M
    U = N_MOE * N_EXP                                      # 12,032 expert slots
    per_expert_bytes = (TOTAL * SHARE / U) * BITS_PER_PARAM / 8.0   # 7.86 MB
    gb = lambda m: m * per_expert_bytes / 1e9             # resident-expert-count -> GB
    pool_gb = gb(U)                                        # 94.6 GB full pool

    # --- static-frequency cumulative hit (analytic Zipf, anchored to the measured 8%->0.47 point) ---
    ZIPF_S = 0.73                                          # fitted so cum_hit(0.08*U)=0.47 (the anchor)
    ranks = np.arange(1, U + 1)
    w = 1.0 / ranks ** ZIPF_S
    w /= w.sum()
    cum = np.cumsum(w)                                     # cum[m-1] = hit of top-m frequent experts
    cum_hit = lambda m: float(cum[min(max(m, 1), U) - 1])

    # --- the three residency policies over a VRAM-budget sweep ---
    m_grid = np.unique(np.clip(np.round(np.linspace(1, U, 240)).astype(int), 1, U))
    x_gb = np.array([gb(m) for m in m_grid])
    freq_hit  = np.array([cum_hit(m) for m in m_grid])            # the DISCOVERY (locality-aware)
    blind_hit = m_grid / U                                        # the CONTROL: random m-subset = m/U
    # Belady/oracle: NO 235B trace exists to compute true future-optimal, so this is an APPROXIMATION —
    # a modest envelope a little above static-frequency (closes the residual gap to 1.0 by ~25%).
    oracle_hit = np.minimum(1.0, freq_hit + 0.25 * (1.0 - freq_hit))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x_gb, blind_hit,  color=COLD, lw=2.2, ls="--",
            label="frequency-BLIND residency (random m-subset, hit = m/U)  — the control")
    ax.plot(x_gb, freq_hit,   color=HOT,  lw=2.6,
            label="static-frequency residency (the discovery)  — analytic Zipf s=0.73")
    ax.plot(x_gb, oracle_hit, color=GOLD, lw=1.6, ls=":",
            label="Belady/oracle upper bound (APPROX — no 235B trace; envelope)")

    # measured anchor point (8% resident -> 0.47 freq / 0.08 blind)
    m_a = int(round(0.08 * U)); xa = gb(m_a)
    ax.scatter([xa], [cum_hit(m_a)], s=120, color=HOT, zorder=6, edgecolor="white")
    ax.scatter([xa], [m_a / U],     s=120, color=COLD, zorder=6, edgecolor="white")
    ax.annotate(f"MEASURED anchor: 8% resident ({xa:.1f} GB)\n0.47 freq vs 0.08 blind  =  5.9× hit @ equal VRAM",
                (xa, cum_hit(m_a)), textcoords="offset points", xytext=(18, -6),
                fontsize=8.6, weight="bold", color=HOT)

    # --- VRAM-gap annotations at iso-hit {0.90, 0.95, 0.99} (horizontal "VRAM saved" arrows) ---
    def m_for_hit(target):                                 # smallest top-m frequent reaching target
        return int(np.searchsorted(cum, target) + 1)
    for h in (0.90, 0.95, 0.99):
        gf = gb(m_for_hit(h))                              # frequency GB at this hit
        gbl = gb(int(np.ceil(h * U)))                      # blind GB at this hit (hit=m/U -> m=h*U)
        saved, mult = gbl - gf, gbl / gf
        ax.annotate("", xy=(gf, h), xytext=(gbl, h),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
        ax.text((gf + gbl) / 2, h + 0.012, f"hit {h:.2f}:  save {saved:.0f} GB ({mult:.2f}×)",
                ha="center", fontsize=8.0, color="#333")

    # --- bandwidth-saturation knee: below it you save VRAM but NOT tok/s; above it you buy both ---
    # The roofline (paging_tok_s_model.py) only climbs once the miss term stops dominating — i.e. once
    # hit is high enough that resident reads, not PCIe misses, set the time. That crossover is ~0.85 hit.
    KNEE_HIT = 0.85
    xk = gb(m_for_hit(KNEE_HIT))
    ax.axvline(xk, color="#5a8f7b", lw=1.5, ls="-.")
    ax.text(xk + 0.6, 0.30, f"bandwidth-saturation knee (~hit {KNEE_HIT:.2f})\n← save VRAM only  |  buy VRAM+tok/s →",
            color="#3d6b58", fontsize=8.2, rotation=90, va="bottom")

    # --- 24 GB 4090 VRAM reference (expert-pool budget after non-expert weights is smaller, but mark the box) ---
    ax.axvline(24.0, color="#999", lw=1.4, ls=":")
    ax.text(24.3, 0.05, "24 GB (RTX 4090)", color="#666", fontsize=8.2, rotation=90, va="bottom")

    ax.axhline(0.95, color="#bbb", lw=0.8, ls=":")
    ax.set_xlabel("resident expert VRAM (GB)   [m experts × 7.86 MB; full pool 94.6 GB]")
    ax.set_ylabel("expert hit-rate")
    ax.set_xlim(0, pool_gb * 1.02); ax.set_ylim(0, 1.04)
    ax.set_title("Iso-quality VRAM: frequency-aware vs frequency-BLIND expert residency (Qwen3-235B)")
    ax.legend(loc="lower right", fontsize=8.0)
    # ΔLM twin-note: the framework maps hit-rate to ΔLM (loss-increase when missed experts are dropped);
    # the load-bearing figure (paper_deltalm_arch) uses ΔLM>0.5 as "memory is used". We do NOT have a
    # clean per-hit-rate ΔLM mapping at 235B, so per spec we add a caption note rather than invent one.
    fig.text(0.012, -0.02,
             "Control is frequency-BLIND residency (random m-subset), NOT LRU. Static-frequency curve is "
             "analytic (Zipf s=0.73) anchored to the MEASURED 8%→0.47 hit point; no 235B trace exists, so "
             "the 235B projection is Zipf-derived, not a captured trace. Belady curve is an APPROXIMATION. "
             "iso-quality must ultimately be verified as real PPL/ΔLM, not hit-rate alone (ΔLM>0.5 = memory used).",
             fontsize=6.6, color="#555", wrap=True)
    save(fig, "paper_vram_isoquality.png")


# ---------------------------------------------------------------------------
# MEASURED mode: rebuild the headline from a REAL MoE expert-routing trace.
# Gated behind $TQ_MOE_TRACE (or --moe-trace). The static-frequency curve is the per-layer
# top-f aggregate hit measured on the trace (see static_frequency_curve's per-layer note); no
# analytic Zipf anchor is used — s_fit, the 8%->hit number, and the iso-hit GB gaps all come
# from the captured counts.
# ---------------------------------------------------------------------------
def _vram_isoquality_measured(trace_path: str):
    from turboquant.core.memlocality import static_frequency_curve

    c = static_frequency_curve(trace_path)
    n_slots, n_layers, k = c["n_slots"], c["n_layers"], c["k"]
    frac = c["frac"]                     # per-layer resident fraction
    m_per_layer = c["resident_per_layer"]
    agg_hit = c["agg_hit"]               # MEASURED static-frequency aggregate hit (IN-SAMPLE)
    agg_hit_oos = c["agg_hit_oos"]       # DEPLOYABLE: residency from calib, hit on held-out tokens
    blind_hit = c["blind_hit"]           # control: per-layer random f-fraction -> hit = f

    # --- byte budget (honest choice) -------------------------------------------------------------
    # We reuse the SAME 235B per-expert byte budget the analytic figure uses (7.86 MB @ Q3_K_M) so
    # the GB axis is comparable across modes, and PROJECT the per-layer skew measured on a smaller
    # MoE (e.g. 30B-A3B) onto the 235B universe geometry (94 MoE layers). The 30B-A3B per-expert is
    # smaller, so a GB axis using the 30B per-expert size would not be the 235B deployment number;
    # we therefore keep the 235B byte budget and LABEL the axis "235B-projected GB". x-GB uses the
    # 235B layer count (94) scaled from the measured per-layer fraction, so the full pool == 94.6 GB
    # exactly as in the analytic figure (apples-to-apples). If the trace itself is a 235B trace
    # (n_layers==94) this projection is identity.
    TOTAL, SHARE, N_MOE_235B, N_EXP_235B, BITS = 235e9, 0.92, 94, 128, 3.5
    U235 = N_MOE_235B * N_EXP_235B
    per_expert_bytes = (TOTAL * SHARE / U235) * BITS / 8.0     # 7.86 MB (same as analytic)
    pool_gb = U235 * per_expert_bytes / 1e9                    # 94.6 GB full 235B pool
    # x-GB at per-layer fraction f: keep f of every one of the 94 projected layers' 128 experts.
    proj_resident = lambda f: f * N_MOE_235B * N_EXP_235B      # projected resident expert count
    x_gb = proj_resident(frac) * per_expert_bytes / 1e9
    gb_for_frac = lambda f: float(proj_resident(f) * per_expert_bytes / 1e9)

    measured_30b = (n_layers != N_MOE_235B) or (n_slots != N_EXP_235B)
    proj_note = (f"235B-projected GB (per-layer skew MEASURED on a {n_layers}L×{n_slots}e MoE trace)"
                 if measured_30b else f"235B GB (MEASURED {n_layers}L×{n_slots}e trace)")

    # oracle: APPROX envelope above static-frequency (same construction as analytic; no per-fraction
    # Belady on the full trace here — labeled APPROX).
    oracle_hit = np.minimum(1.0, agg_hit + 0.25 * (1.0 - agg_hit))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x_gb, blind_hit, color=COLD, lw=2.2, ls="--",
            label="frequency-BLIND residency (random per-layer f-subset, hit = f)  — the control")
    ax.plot(x_gb, agg_hit, color=HOT, lw=2.0, ls=(0, (4, 2)),
            label="static-frequency residency — MEASURED, IN-SAMPLE (optimistic)")
    ax.plot(x_gb, agg_hit_oos, color=HOT, lw=2.8,
            label=f"static-frequency residency — HELD-OUT (deployable: {c['holdout_kind']})")
    ax.plot(x_gb, oracle_hit, color=GOLD, lw=1.6, ls=":",
            label="Belady/oracle upper bound (APPROX envelope, not per-fraction OPT)")

    # measured anchor: hottest 8% of each layer resident -> measured aggregate hit
    fa = 0.08
    xa = gb_for_frac(fa)
    ya = c["hit_at_8pct"]
    ax.scatter([xa], [ya], s=120, color=HOT, zorder=6, edgecolor="white")
    ax.scatter([xa], [fa], s=120, color=COLD, zorder=6, edgecolor="white")
    mult_a = ya / fa if fa else float("nan")
    ax.annotate(f"MEASURED anchor: top-8%/layer ({xa:.1f} GB)\n"
                f"{ya:.2f} freq vs {fa:.2f} blind  =  {mult_a:.1f}× hit @ equal VRAM",
                (xa, ya), textcoords="offset points", xytext=(18, -6),
                fontsize=8.6, weight="bold", color=HOT)

    # iso-hit VRAM-gap arrows at {0.90, 0.95, 0.99}, all from MEASURED curves.
    agg_for_m = c["_agg_hit_for_m"]
    def frac_for_hit_freq(target):                # smallest per-layer fraction reaching target (static-freq)
        m = int(np.searchsorted(agg_hit, target))
        m = min(max(m, 0), len(frac) - 1)
        return float(frac[m])
    for h in (0.90, 0.95, 0.99):
        gf = gb_for_frac(frac_for_hit_freq(h))    # static-frequency GB at this hit
        gbl = gb_for_frac(h)                      # blind GB at this hit (blind hit = f -> f = h)
        if gf <= 0:
            continue
        saved, mult = gbl - gf, gbl / gf
        ax.annotate("", xy=(gf, h), xytext=(gbl, h),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
        ax.text((gf + gbl) / 2, h + 0.012, f"hit {h:.2f}:  save {saved:.0f} GB ({mult:.2f}×)",
                ha="center", fontsize=8.0, color="#333")

    # bandwidth-saturation knee (~0.85 hit), same regime note as analytic.
    KNEE_HIT = 0.85
    xk = gb_for_frac(frac_for_hit_freq(KNEE_HIT))
    ax.axvline(xk, color="#5a8f7b", lw=1.5, ls="-.")
    ax.text(xk + 0.6, 0.30, f"bandwidth-saturation knee (~hit {KNEE_HIT:.2f})\n← save VRAM only  |  buy VRAM+tok/s →",
            color="#3d6b58", fontsize=8.2, rotation=90, va="bottom")

    ax.axvline(24.0, color="#999", lw=1.4, ls=":")
    ax.text(24.3, 0.05, "24 GB (RTX 4090)", color="#666", fontsize=8.2, rotation=90, va="bottom")
    ax.axhline(0.95, color="#bbb", lw=0.8, ls=":")
    ax.set_xlabel(f"resident expert VRAM (GB)   [{proj_note}; per-expert 7.86 MB; full pool {pool_gb:.0f} GB]")
    ax.set_ylabel("expert hit-rate")
    ax.set_xlim(0, pool_gb * 1.02); ax.set_ylim(0, 1.04)
    src = f"Qwen3 routing trace, {n_layers}L×{n_slots}e top-{k}"
    ax.set_title(f"Iso-quality VRAM: frequency-aware vs frequency-BLIND residency  —  MEASURED ({src})")
    ax.legend(loc="lower right", fontsize=8.0)
    fig.text(0.012, -0.02,
             f"MEASURED from {Path(trace_path).name} ({c['total_accesses']:.0f} accesses across {n_layers} MoE "
             f"layers). Static-frequency curve is the per-layer top-f AGGREGATE hit (access-weighted over "
             f"layers); control is frequency-BLIND per-layer random f-subset (hit=f), NOT LRU. s_fit and the "
             f"top-8%/layer->{ya:.2f} anchor are computed from the captured counts. "
             + ("GB axis PROJECTS this per-layer skew onto the 235B universe (94L×128e, 7.86 MB/expert); the "
                "skew is measured at smaller scale. " if measured_30b else "")
             + "Belady curve is an APPROX envelope. iso-quality must ultimately be real PPL/ΔLM (ΔLM>0.5).",
             fontsize=6.6, color="#555", wrap=True)
    save(fig, "paper_vram_isoquality.png")
    # console summary (auditable; printed in measured mode)
    def frac_for_hit_oos(target):
        m = int(np.searchsorted(agg_hit_oos, target)); m = min(max(m, 0), len(frac) - 1)
        return float(frac[m])
    print(f"  [measured] trace={Path(trace_path).name}  n_layers={n_layers}  n_slots={n_slots}  k={k}")
    print(f"  [measured] held-out kind: {c['holdout_kind']}")
    print(f"  [measured] top-8%/layer hit:  IN-SAMPLE={ya:.3f}   HELD-OUT={c['hit_at_8pct_oos']:.3f}  "
          f"(blind=0.08; analytic anchor was 0.47)")
    print(f"  [measured] (s_fit={c['s_fit']:.3f} is the averaged-sorted-profile slope; the honest single-"
          f"distribution exponent is the capture's global s_fit. The DEPLOYABLE numbers are HELD-OUT.)")
    for h in (0.90, 0.95, 0.99):
        gf = gb_for_frac(frac_for_hit_freq(h)); gbl = gb_for_frac(h)
        go = gb_for_frac(frac_for_hit_oos(h))
        print(f"  [measured] iso-hit {h:.2f}: blind {gbl:.1f} GB | in-sample {gf:.1f} GB "
              f"({gbl/max(gf,1e-9):.2f}x) | HELD-OUT {go:.1f} GB ({gbl/max(go,1e-9):.2f}x)  <- deployable")
    return c


# The negative — vanilla vs locality-loss (hit@5% and ΔLM both drop)
def negative():
    cond = ["vanilla\n(no loss)", "locality loss\nfrom step 0", "locality loss\n+ 50% warmup"]
    hit5 = [0.94, 0.82, 0.90]; dlm = [0.72, 0.16, 0.62]
    x = np.arange(len(cond)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x-w/2, hit5, w, color=ACC, edgecolor="white", label="hit@5%-cache")
    b2 = ax.bar(x+w/2, dlm, w, color=HOT, edgecolor="white", label="ΔLM (memory used)")
    for bars, vs in [(b1, hit5), (b2, dlm)]:
        for b, v in zip(bars, vs):
            ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(cond)
    ax.set_ylabel("metric value"); ax.set_ylim(0, 1.05)
    ax.set_title("Training for locality BACKFIRES: it lowers hit-rate AND memory usage")
    ax.legend(loc="upper right")
    save(fig, "paper_negative.png")


# Live KTransformers deployment — decode tok/s vs GPU-expert ratio, frequency vs random (the knee).
# Reads the MEASURED ratio sweep (H100, Qwen3-30B-A3B, sglang-kt BF16); see
# the paper and
# data/eval/ktransformers_live_h100_ratio_sweep.csv.
def livedemo_knee():
    import csv
    src = Path(__file__).resolve().parent.parent / "data" / "eval" / "ktransformers_live_h100_ratio_sweep.csv"
    rows = list(csv.DictReader(open(src)))
    def series(place):
        r = sorted((x for x in rows if x["placement"] == place), key=lambda x: float(x["ratio"]))
        return ([float(x["ratio"]) for x in r],
                [float(x["decode_tok_s_median"]) for x in r],
                [float(x["analytic_hit"]) for x in r])
    fr, ff, fh = series("frequency")
    rr, rf, rh = series("random")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(fr, ff, color=HOT, lw=2.4, marker="D", ms=8, mec="white",
            label="frequency (our per-layer trace)")
    ax.plot(rr, rf, color=COLD, lw=2.2, marker="o", ms=8, mec="white",
            label="random (frequency-blind control)")
    ax.set_yscale("log")
    # speedup annotations at each ratio
    for x, yf, yr in zip(fr, ff, rf):
        ax.annotate(f"{yf/yr:.2f}×", (x, yf), textcoords="offset points", xytext=(2, 9),
                    fontsize=9, weight="bold", color=HOT)
    # hit-rate labels on each marker
    for x, y, h in zip(fr, ff, fh):
        ax.annotate(f"hit {h:.2f}", (x, y), textcoords="offset points", xytext=(6, -13),
                    fontsize=7.5, color=HOT)
    for x, y, h in zip(rr, rf, rh):
        ax.annotate(f"hit {h:.2f}", (x, y), textcoords="offset points", xytext=(6, -13),
                    fontsize=7.5, color=ACC)
    ax.set_xlabel("GPU-expert ratio  (--kt-gpu-experts-ratio)")
    ax.set_ylabel("decode tok/s  (median of n; log scale)")
    ax.set_xticks(fr); ax.set_xticklabels([f"{x:.2f}" for x in fr])
    ax.set_title("Live KTransformers (H100, Qwen3-30B-A3B): per-layer frequency placement\n"
                 "wins more as GPU residency rises — the bandwidth knee, measured at iso-VRAM")
    ax.legend(loc="upper left")
    fig.text(0.012, -0.02,
             "Same VRAM (~68.5 GB) at every point; only WHICH experts are resident differs. Below the knee "
             "both arms are CPU-bound (~2.4 tok/s); at ratio 0.90 frequency reaches hit 1.00 (112.7 tok/s, "
             "pure GPU) while the control's 10% miss rate (hit 0.90) holds it at 31.3 tok/s. Output is exact "
             "(lossless paging). Absolute tok/s is hardware/tuning-bound; the contrast is the result.",
             fontsize=6.6, color="#555", wrap=True)
    save(fig, "paper_livedemo_knee.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the paper figures.")
    ap.add_argument("--moe-trace", default=None,
                    help="real MoE routing .npz -> rebuild paper_vram_isoquality.png in MEASURED mode "
                         "(else $TQ_MOE_TRACE; else analytic Zipf s=0.73 fallback)")
    ap.add_argument("--only", default=None,
                    help="run only one figure: hitrate|deltalm|vram|negative|livedemo")
    args = ap.parse_args()
    only = args.only
    if only in (None, "hitrate"): hitrate_curve()
    if only in (None, "deltalm"): deltalm_arch()
    if only in (None, "vram"): vram_isoquality(trace_path=args.moe_trace)
    if only in (None, "negative"): negative()
    if only in (None, "livedemo"): livedemo_knee()
    print("done.")
