# Committed traces

- `moe_Qwen3-30B-A3B_xdoc.npz` — the real cross-document MoE routing trace (Qwen3-30B-A3B, 48 layers x
  128 experts, top-8, 8 held-out WikiText documents). Arrays `L{0..47}_idx` are `(n_tokens, 8)` selected
  expert ids per token; `doc_id` is `(n_tokens,)`; `n_slots=128`. This is the artifact behind the
  iso-quality-VRAM result, the confidence intervals, and the dynamic-cache baseline.
- `moe_Qwen3-30B-A3B_xdoc.expert_stats.pt` — the per-layer frequency map (`logical_count`, shape
  `(1,48,128)`) derived from the trace; the KTransformers `--init-expert-location` placement map.
  Regenerate with `python scripts/trace_to_ktransformers_stats.py <trace.npz> -o <out.pt>` (needs torch).
- `synthetic_moe_demo.npz` — a small synthetic trace for smoke tests / the probe's `--synthetic` regimes.

Provenance: routing was captured by running the (Apache-2.0) Qwen3-30B-A3B model over WikiText-2
(CC-BY-SA) documents and recording per-token top-k expert selections. The committed `.npz` contains only
aggregated routing indices (no model weights, no document text).

- `moe_Qwen3-235B-A22B_xdoc.npz` — the directly-captured Qwen3-235B-A22B routing trace (94 MoE layers x 128
  experts, top-8, 32 cross-document WikiText documents; arrays `L{0..93}_idx` = `(n_tokens, 8)`, `doc_id`,
  `n_slots=128`). The scale-confirmation artifact: per-layer locality holds at ~8x the parameters of the 30B
  trace, so the iso-quality-VRAM GB axis is measured at 235B rather than projected. Captured in 4-bit NF4 on
  one 80GB-class GPU (transformers 4.57.1; see the paper's reproducibility notes).
