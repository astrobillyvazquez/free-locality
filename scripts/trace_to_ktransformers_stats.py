#!/usr/bin/env python3
"""Convert a captured MoE routing trace (.npz) -> KTransformers/SGLang ``--init-expert-location`` stats (.pt).

This is the P1 (KTransformers) bridge from the paper. It takes our captured routing trace
(``data/traces/moe_Qwen3-30B-A3B_xdoc.npz``) and emits the per-layer expert activation-count
tensor that the kvcache-ai SGLang fork's ``--kt-expert-placement-strategy frequency`` path consumes
via ``--init-expert-location <stats>.pt``. The ``frequency`` strategy then pins the most-frequently
activated experts per layer onto the GPU -- i.e. the exact the paper static-frequency residency policy.

------------------------------------------------------------------------------------------------
OUTPUT SCHEMA (pinned against the kvcache-ai/sglang loader, fetched 2026-06-03)
------------------------------------------------------------------------------------------------
The loader is ``sglang/srt/eplb/expert_location.py::compute_initial_expert_location_metadata``:

    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)   # -> must be a dict of TENSORS only
    ...
    elif "logical_count" in data_dict:
        # KT frequency path: "logical_count will be read directly by KT layers"
        ... init_by_eplb(..., logical_count=data_dict["logical_count"])

and ``init_by_eplb`` does ``if len(logical_count.shape) == 2: logical_count.unsqueeze(0)``.

The native producer (the ExpertDistributionRecorder ``stat`` dump,
``--record-kt-gpu-expert-distribution``) writes:

    output = dict(rank=..., logical_count=<Tensor (dim_extra, num_layers, num_logical_experts)>,
                  average_utilization_rate_over_window=...)
    torch.save(output, "expert_distribution_recorder_<ts>.pt")

We therefore write a ``torch.save``'d dict:

    {
      "logical_count": int64 Tensor of shape (1, n_layers, n_experts)   # (dim_extra=1, 48, 128)
    }

  * ``logical_count[0, L, e]`` = number of times expert ``e`` was selected in MoE layer ``L``,
    summed over all (token, top-k-slot) routing events in the trace
    (== ``np.bincount(L{L}_idx.reshape(-1), minlength=n_experts)``).
  * Shape is 3D ``(1, n_layers, n_experts)`` to mirror the recorder's NATIVE output exactly; the
    leading dim is the recorder's ``dim_extra`` (EP-rank/window) axis, which is 1 for our single
    offline capture. (``--with-rank-key`` additionally writes the recorder's ``rank``/utilization
    fields; off by default because the loader only reads ``logical_count``.)
  * dtype int64: the recorder preserves the gather dtype (integer counts) and ``rebalance_experts``
    treats this as ``tokens_per_expert``; integer counts are the safe, lossless representation.
  * ``weights_only=True`` constraint: the dict must contain ONLY tensors / plain scalars -- no numpy
    arrays, no custom objects. We cast every value to a torch tensor before saving.

------------------------------------------------------------------------------------------------
VERIFY against the live KTransformers loader on the 4090 (NOT validated on CPU here):
------------------------------------------------------------------------------------------------
  [VERIFY-1] LAYER ORDERING / INDEX BASE. We emit layers in ascending trace-layer index
             (L0..L47) packed into rows 0..47. the paper's model is Qwen3-30B-A3B: 48 MoE layers,
             all-MoE, so trace-layer-i == model-MoE-layer-i with a 0-based contiguous map. If the
             target model has DENSE prefix layers (some Qwen variants do) the KT loader may expect
             rows indexed by GLOBAL decoder-layer id (with dense layers present as zero/te rows).
             Use --n-model-layers + --moe-layer-offset to left-pad if so. VERIFY which indexing
             ``num_layers`` in ModelConfigForExpertLocation uses for the exact checkpoint.
  [VERIFY-2] EXPERT-COUNT MATCH. n_experts (=n_slots=128 for 30B-A3B) must equal the model's
             ``num_logical_experts``. For 122B-A10B this differs; pass the trace for THAT model.
             The loader does not pad/truncate -- a mismatch will error in init_by_eplb.
  [VERIFY-3] 2D vs 3D. The loader accepts 2D (it unsqueezes) and the recorder emits 3D. We emit 3D
             to match the recorder byte-for-byte. If a given fork build rejects 3D in the KT
             frequency branch, re-run with --squeeze to emit 2D (n_layers, n_experts).
  [VERIFY-4] COUNTS vs FREQUENCIES vs RANKS. We emit raw COUNTS (matching the recorder). The
             frequency strategy only needs the per-layer RANK order, which counts preserve. If a
             build expects normalized frequencies, --normalize emits per-layer probabilities
             (float64); rank order is identical so placement is unchanged.
  [VERIFY-5] weights_only=True. Confirmed in the fetched loader. If an older fork build uses
             ``weights_only=False`` it still loads our dict; the constraint is one-directional.

Dependencies: numpy (always) + torch (only when actually writing/self-testing a .pt). ``--help`` and
import work WITHOUT torch. No GPU required. No KTransformers required.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

LAYER_KEY_RE = re.compile(r"L(\d+)_idx")


# --------------------------------------------------------------------------- #
# Core (numpy-only) -- importable and testable without torch                   #
# --------------------------------------------------------------------------- #
def list_trace_layers_npz(npz) -> list[int]:
    """Sorted layer indices present as ``L{idx}_idx`` keys in an opened npz."""
    layers = []
    for key in npz.files:
        m = LAYER_KEY_RE.fullmatch(key)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def trace_to_logical_count(
    npz,
    *,
    n_experts: int | None = None,
    n_model_layers: int | None = None,
    moe_layer_offset: int = 0,
) -> np.ndarray:
    """Compute the (n_layers, n_experts) per-layer expert activation-count matrix from a trace npz.

    counts[i, e] = sum over all tokens & top-k slots of [layer-i selected expert e]
                 = np.bincount(L{i}_idx.reshape(-1), minlength=n_experts).

    Layer rows are packed in ascending trace-layer order starting at ``moe_layer_offset``. If
    ``n_model_layers`` is given the matrix is left-/zero-padded to that many rows (for models whose
    MoE layers start after a dense prefix; see VERIFY-1). The returned dtype is int64.
    """
    layers = list_trace_layers_npz(npz)
    if not layers:
        raise ValueError("no 'L{idx}_idx' arrays found in the trace npz")

    if n_experts is None:
        n_experts = int(npz["n_slots"]) if "n_slots" in npz.files else None

    rows: dict[int, np.ndarray] = {}
    inferred_max = 0
    for L in layers:
        idx = npz[f"L{L}_idx"]
        flat = np.asarray(idx).reshape(-1)
        inferred_max = max(inferred_max, int(flat.max()) + 1)
        ne = n_experts if n_experts is not None else inferred_max
        rows[L] = np.bincount(flat, minlength=ne).astype(np.int64)

    if n_experts is None:
        n_experts = inferred_max
    # re-bincount any rows that were sized to a smaller inferred width
    for L in layers:
        if rows[L].shape[0] != n_experts:
            flat = np.asarray(npz[f"L{L}_idx"]).reshape(-1)
            rows[L] = np.bincount(flat, minlength=n_experts).astype(np.int64)

    contiguous_from_zero = layers == list(range(layers[0], layers[-1] + 1))
    if not contiguous_from_zero:
        # Non-contiguous trace layers: place each at its own (offset) row index, zero elsewhere.
        max_row = moe_layer_offset + max(layers) + 1
        total_rows = max(max_row, n_model_layers or 0)
        counts = np.zeros((total_rows, n_experts), dtype=np.int64)
        for L in layers:
            counts[moe_layer_offset + L] = rows[L]
        return counts

    body = np.vstack([rows[L] for L in layers])  # (len(layers), n_experts)
    if moe_layer_offset == 0 and (n_model_layers is None or n_model_layers == body.shape[0]):
        return body
    total_rows = n_model_layers if n_model_layers is not None else moe_layer_offset + body.shape[0]
    counts = np.zeros((total_rows, n_experts), dtype=np.int64)
    counts[moe_layer_offset:moe_layer_offset + body.shape[0]] = body
    return counts


def summarize(counts: np.ndarray) -> str:
    """Human-readable shape/skew summary of the count matrix (no torch needed)."""
    n_layers, n_experts = counts.shape
    totals = counts.sum(axis=1)
    nz_layers = int((totals > 0).sum())
    # per-layer top-8% capture (the ADR anchor)
    m8 = max(1, int(round(0.08 * n_experts)))
    sorted_desc = np.sort(counts, axis=1)[:, ::-1]
    cap8 = sorted_desc[:, :m8].sum()
    grand = counts.sum()
    hit8 = float(cap8 / grand) if grand else 0.0
    # uniform baseline for that fraction == m8/n_experts
    blind8 = m8 / n_experts
    return (
        f"logical_count shape = ({n_layers}, {n_experts})  [layers x experts]\n"
        f"  non-empty layers      : {nz_layers}/{n_layers}\n"
        f"  total routing events  : {int(grand):,}\n"
        f"  per-layer events      : min={int(totals.min()):,} max={int(totals.max()):,} "
        f"mean={totals.mean():.0f}\n"
        f"  static-freq hit @ top-{m8}/{n_experts} experts ({100*m8/n_experts:.1f}%): "
        f"{hit8:.3f}   (frequency-blind/uniform baseline = {blind8:.3f})"
    )


# --------------------------------------------------------------------------- #
# IO (torch) -- only imported when we actually write a .pt                      #
# --------------------------------------------------------------------------- #
def write_stats_pt(
    counts: np.ndarray,
    out_path: str,
    *,
    squeeze: bool = False,
    normalize: bool = False,
    with_rank_key: bool = False,
) -> dict:
    """Write the KTransformers/SGLang ``logical_count`` stats .pt (torch.save'd dict of tensors).

    See module docstring for the schema. Returns the dict that was saved (tensors), for inspection.
    """
    import torch  # local import: keep --help / import torch-free

    if normalize:
        totals = counts.sum(axis=1, keepdims=True)
        vals = counts.astype(np.float64) / np.maximum(totals, 1.0)
        lc = torch.from_numpy(vals)  # float64
    else:
        lc = torch.from_numpy(np.ascontiguousarray(counts.astype(np.int64)))

    if not squeeze:
        lc = lc.unsqueeze(0)  # (1, n_layers, n_experts) -> mirror recorder's dim_extra axis

    payload: dict = {"logical_count": lc}
    if with_rank_key:
        # Mirror the recorder's full dict (loader ignores these extra keys).
        payload["rank"] = torch.tensor(0, dtype=torch.int64)
        payload["average_utilization_rate_over_window"] = torch.tensor(
            float("nan"), dtype=torch.float64
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload


# --------------------------------------------------------------------------- #
# Self-test (synthetic trace; numpy + torch round-trip)                         #
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    rng = np.random.default_rng(0)
    n_layers, n_experts, T, k = 4, 16, 500, 3
    # Build a synthetic trace with a known per-layer hot expert so we can assert argmax.
    expected_hot = []
    npz_dict = {}
    for L in range(n_layers):
        hot = (L * 3 + 1) % n_experts
        expected_hot.append(hot)
        base = rng.integers(0, n_experts, size=(T, k))
        # force ~60% of slots in this layer to the hot expert
        mask = rng.random((T, k)) < 0.6
        base[mask] = hot
        npz_dict[f"L{L}_idx"] = base.astype(np.int32)
    npz_dict["n_slots"] = np.int64(n_experts)

    with tempfile.TemporaryDirectory() as d:
        npz_path = Path(d) / "synthetic_trace.npz"
        np.savez(npz_path, **npz_dict)
        with np.load(npz_path) as npz:
            counts = trace_to_logical_count(npz, n_experts=n_experts)

        # 1. shape
        assert counts.shape == (n_layers, n_experts), counts.shape
        # 2. argmax per layer == injected hot expert
        got_hot = counts.argmax(axis=1).tolist()
        assert got_hot == expected_hot, (got_hot, expected_hot)
        # 3. row sums == T*k (every routing event counted exactly once)
        assert (counts.sum(axis=1) == T * k).all(), counts.sum(axis=1)
        # 4. matches the documented bincount contract exactly
        with np.load(npz_path) as npz:
            for L in range(n_layers):
                ref = np.bincount(npz[f"L{L}_idx"].reshape(-1), minlength=n_experts)
                assert np.array_equal(counts[L], ref), L

        print("[self-test] numpy core OK: shape, per-layer argmax, row-sum, bincount contract")

        # 5. torch round-trip mirrors the loader's torch.load(weights_only=True) contract
        try:
            import torch
        except ImportError:
            print("[self-test] torch not installed -> skipped .pt round-trip (numpy core passed)")
            return 0

        out = Path(d) / "stats.pt"
        write_stats_pt(counts, str(out))
        loaded = torch.load(str(out), weights_only=True)  # the exact loader call
        assert set(loaded.keys()) == {"logical_count"}, loaded.keys()
        lc = loaded["logical_count"]
        assert lc.shape == (1, n_layers, n_experts), lc.shape
        assert lc.dtype == torch.int64, lc.dtype
        assert np.array_equal(lc[0].cpu().numpy(), counts)
        # 2D (--squeeze) variant
        write_stats_pt(counts, str(out), squeeze=True)
        lc2 = torch.load(str(out), weights_only=True)["logical_count"]
        assert lc2.shape == (n_layers, n_experts), lc2.shape
        # normalized variant preserves per-layer rank order
        write_stats_pt(counts, str(out), normalize=True)
        lcf = torch.load(str(out), weights_only=True)["logical_count"]
        assert lcf.dtype == torch.float64
        assert (lcf[0].argmax(dim=1).cpu().numpy() == counts.argmax(axis=1)).all()
        print("[self-test] torch .pt round-trip OK: weights_only load, 3D/2D/normalize variants")

    print("[self-test] PASS")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a captured MoE routing trace (.npz) to the KTransformers/SGLang "
            "--init-expert-location activation-stats file (.pt). See module docstring for the "
            "pinned output schema and the VERIFY-against-loader checklist."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("trace", nargs="?", help="input trace .npz (e.g. data/traces/moe_Qwen3-30B-A3B_xdoc.npz)")
    p.add_argument("-o", "--out", help="output stats .pt path (default: alongside trace, *.expert_stats.pt)")
    p.add_argument("--n-experts", type=int, default=None,
                   help="override num_logical_experts (default: read 'n_slots' from npz). "
                        "Must equal the model's num_logical_experts (VERIFY-2).")
    p.add_argument("--n-model-layers", type=int, default=None,
                   help="pad the layer dim to this many rows (for models with a dense prefix; VERIFY-1)")
    p.add_argument("--moe-layer-offset", type=int, default=0,
                   help="row index of the first MoE layer when left-padding (VERIFY-1)")
    p.add_argument("--squeeze", action="store_true",
                   help="emit 2D (n_layers, n_experts) instead of 3D (1, ...) (VERIFY-3)")
    p.add_argument("--normalize", action="store_true",
                   help="emit per-layer probabilities (float64) instead of raw counts (VERIFY-4)")
    p.add_argument("--with-rank-key", action="store_true",
                   help="also write the recorder's rank/utilization keys (loader ignores them)")
    p.add_argument("--dry-run", action="store_true",
                   help="compute & print the summary but do NOT write the .pt (no torch needed)")
    p.add_argument("--self-test", action="store_true",
                   help="run the synthetic-trace self-test and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.trace:
        build_parser().error("a trace .npz is required (or use --self-test)")

    trace_path = Path(args.trace)
    if not trace_path.exists():
        build_parser().error(f"trace not found: {trace_path}")

    with np.load(trace_path) as npz:
        counts = trace_to_logical_count(
            npz,
            n_experts=args.n_experts,
            n_model_layers=args.n_model_layers,
            moe_layer_offset=args.moe_layer_offset,
        )

    print(f"trace: {trace_path}")
    print(summarize(counts))

    if args.dry_run:
        print("[dry-run] not writing .pt")
        return 0

    out = args.out or str(trace_path.with_suffix("")) + ".expert_stats.pt"
    write_stats_pt(
        counts, out,
        squeeze=args.squeeze, normalize=args.normalize, with_rank_key=args.with_rank_key,
    )
    shape = counts.shape if args.squeeze else (1,) + counts.shape
    dt = "float64" if args.normalize else "int64"
    print(f"wrote {out}")
    print(f"  dict key 'logical_count': torch {dt} tensor of shape {tuple(shape)}")
    print("  load with: torch.load(path, weights_only=True)['logical_count']")
    print("  server flag: --kt-expert-placement-strategy frequency "
          f"--init-expert-location {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
