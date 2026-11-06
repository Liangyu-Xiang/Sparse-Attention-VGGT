# Teacher-calibrated temporal probability estimation

This is an independent experimental backend (`teacher_guided_history`). Original
backends/configuration defaults are unchanged. The target is the head/probe mean
of four actual Query full-Key softmaxes on the **same sparse trajectory QKV**.
It is not the exact probability for every Query, nor a full Dense model teacher.
No model training or geometric test truth is used for estimator selection.

## Fixed state model

Value head-sketch4, source-local unanchored clustering (one update), 10% cluster
count, History A / prior-residual execution, full Exact eligibility, 16 budget
iterations, 256-Query groups, 10/18/10% layer budgets and all five special Query
corrections match the preceding fast14 and streamed-history configurations.
Only probability production/transfer and mathematically equivalent AV scheduling
change. Key blocks are contiguous64 logical Patch Keys, not output-mask blocks.

## Persistent fine density

At L0 only, four full-Key probes initialize p_0. There is no historical input at
L0; this declared bootstrap is charged to runtime/QK counts. Layers1–23 do not
compute a current full-Key Query probability bank.

Each layer computes a rough block mass C_l(r) from four actual probes against
mean pooled Key blocks, with log(population) and individual special Keys in the
softmax. Eight uniformly spaced actual Keys/block additionally supply a within-
anchor softmax A_l(r,j). All per-head softmaxes are averaged **before transfer**.
No adjacent-layer head matching is used.

For previous fine p, retain M(r)=sum_{e in r}p(e). Transport block mass by
M(r)*C_l(r)/C_{l-1}(r), then normalize together with the analogous aggregate
special mass. Confidence beta(r)=2sqrt(C_l(r)C_{l-1}(r))/(C_l(r)+C_{l-1}(r)).

Within each64-Key block, assign every current Key to the nearest of eight current
anchors in fixed head-sketch4 Key feature space. This assignment is shared across
Query groups; it contains no all-Key Query dot products. Tilt the previous fine
conditional by exp(gamma*(log A_l - log A_{l-1})) at that anchor, and renormalize
within the block. Tilt is clamped to [-20,20]. Mix this tilted conditional with
block-uniform using beta, then mix transported history with current coarse prior
using alpha=.75. Gamma=1. This is a change of probability shape, not a uniform
rescaling. Anchor identity is fixed by token position; the current nearest-anchor
assignment can change each layer. It is a feature-based interpolation heuristic,
not a guaranteed bound on exact per-Key changes.

## Produce next-layer evidence during formal AV

Formal Attention keeps original per-head Q/K/V and normal compensated logits.
Export four probe rows of already computed raw QK logits in FP16 relative to a
per-tile raw max. Reuse formal online softmax quantities and inverse compensation
to accumulate the **uncompensated selected partition**. No second selected QK.

After AV, recover head/probe-mean selected conditional probabilities. Multiply by
the predicted total selected mass (including special mass). Replace the selected
entries of the transient fine prior; leave unselected fine entries intact. A
block normalization restores the predicted block masses (IPF with one fixed block
constraint). This avoids estimating an absolute missing-Key Dense normalizer
from a block mean. It remains approximate: selected head masses are replaced by
one head-mean mass. Failed odds-calibration alternatives are retained as RGB-only
diagnostics rather than adopted after target scoring.

L0 keeps its full bootstrap density; L23 skips evidence generation because no
consumer exists. All caches reset between sequences. Cross-layer data has no
head dimension. Two FP32 G×Patch buffers avoid erasing unselected fine history;
a4GiB explicit limit guards their combined allocation. This uses more memory than
sparse-only conditional caches and is reported, not called free storage.

## Selection and validation protocol

Record one ScanNet64 RGB trajectory and one TUM64 RGB trajectory without GT.
Replay the same QKV with fast4 probability targets, compare estimator L1/L2/JSD,
cosine and retained teacher mass. Refine finite confidence/update candidates;
compare actual masks and projected AV on the same history IDs. Validate the
chosen method on different TUM128 RGB, then freeze before once-only complete
7Scenes7 / ETH3D11 pose+recon_unposed. Target GT is never used for calibration.

Kernel scheduling changes use only microbenchmarks and numerical tests. Dense,
SAF3R, fast4 and old history full controls are not rerun; prior quality/times are
explicit historical references. Missing paired Dense speedup is N/A.
