# Inference efficiency experiments

See report.md for acceptance failures, paired original-method timings, and historical-only Dense limitations. Candidate configurations are opt-in; neither candidate meets all requirements. All quality percentages in Markdown are explicitly labelled, raw JSON uses fractions.

Each variant includes raw per-scene six-metric results, protocol audit, three-sample full-forward timing, an independent compressed exclusive profile, GPU lease monitors, a source SHA256 manifest, and a source delta archive against 31b32ea. External checkpoints and datasets are not included. v6 contained an Exact-mask layout error; v7 fixed it; v8 restores serial Attention arithmetic. No failed result was overwritten.
