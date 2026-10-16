# Offline Dense-trajectory sensing, NOT fixed-10% inference

This directory captures sampled coarse logits while executing Dense global
attention outputs (`sensing_only=true`). It is an offline calibration diagnostic.
Its pose accuracy and latency MUST NOT be reported as fixed-10% sparse results.

The old diagnostic's per-layer `actual_density`, total_keys and A/V counts
describe the nominal counterfactual sparse budget, not the Dense work actually
executed. The authoritative execution flag is `config.sensing_only=true`.
The report script rejects sensing-only results as a sparse-policy evaluation.
