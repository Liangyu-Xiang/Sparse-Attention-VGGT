# heads128: completed old-scheme stage ablation snapshot

This is a completed **quality-only** diagnostic, not a runtime comparison or a
seven-scene result. Same128 uniformly sampled heads frames, VGGT checkpoint and
preprocessing as the paired Dense run. Stage interventions use one inference
forward, no warmup/repeated timing. Other stages return original Dense output,
while maintaining the sparse scheme's own Value/history state.

Old scheme: manual Attention fractions `[0]+[.2]*9+[.8]*7+[.2]*7`, target20%
support, history80%, whole-Parent selection and original executor. It is **not**
the new independent Attention-first/residual-Value policy.

| Sparse-output layers | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup |
|---|---:|---:|---:|---:|---:|---:|---|
| None: VGGT Dense |51.6281|94.0071|0.0342|0.0308|0.0325|85.1444|N/A: accuracy-only comparison|
|0–9 only|51.2467|94.0248|0.0334|0.0309|0.0321|85.2124|N/A: concurrent diagnosis|
|10–16 only|46.5428|93.0450|0.0400|0.0370|0.0385|78.1859|N/A: concurrent diagnosis|
|17–23 only|44.4841|92.8375|0.0401|0.0423|0.0412|74.2212|N/A: concurrent diagnosis|
|0–23: full old scheme|37.7543|91.4920|0.0492|0.0509|0.0501|61.8153|N/A: not contemporaneous timing|

CD=(ACC+Comp)/2 using unsquared nearest-neighbour distances. F1 threshold0.05m,
downsampling4/512m; unposed reconstruction. Six raw values and input lists are
stored locally under
`outputs/attention_first_residual_value/heads_priority_quality/`; these large
running experiment trees are deliberately not part of the source commit.

Evidence so far: shallow sparsification alone is approximately lossless here;
middle and deep sparsification each degrade quality, with deep worse in AUC3
and F1. This does not identify a specific failing kernel, prove an intrinsic
branch defect, or establish additive stage contributions. Full-layer error
accumulation/interactions remain possible. Follow-up interventions10–12,
13–16,17–20,21–23 and new-scheme ablations are separate pending experiments in
this snapshot; their absent values are not claimed as successes.

The new128-frame fire run hit OOM. Heads/chess repeated-timing runs were
interrupted to prioritize accuracy diagnosis. None is presented as a completed
new-scheme dataset result, and no Dense fallback or quality threshold relaxation
was applied to conceal the failures.
