# Compact implementation of the complete combined revision

Design reference: chatgpt-conversation://6ae08705-de38-83e8-b077-997f1627b1c6,
complexity discussion and subsequent nearest-history clarification.
The measured algorithm is unchanged: Drop=pi, Rep-relative weighted cosine
distortion, all executed patch edges observed, current-grid calibrated relative
logits, four non-Rep history candidates, all eleven states, sixteen dual steps,
20% cap, original exact eligibility, original within-state assignments and
estimated-mass transfer. No quality truth chooses performance settings.

## Query-independent sufficient geometry

The current prior is uniform within CURRENT canonical grids except observed
keys, not necessarily uniform over a fine cluster. A naive single uniform
cluster moment would therefore change the method. Instead cache each layer's
local pairwise distances d(j,k)=clamp(1-dot(z_j,z_k),0,2), with self distance
exactly zero, for members of the SAME cluster only. No global P-by-P table is
formed. For typical 128-frame clusters, the maximum population fits 32 slots.
All query groups reuse this cache: the O(G*P*H*d) repeated Value products become
O(P*M*d) one-time preparation plus O(G*P*H) scalar contractions. Padding size M
comes only from maximum cluster population. If M exceeds 256, the original
loss/top-history/transfer implementation is used, without dropping members or
states. This memory guard is independent of truth.

This preserves arbitrary cross-grid priors, original cosine clipping, zero
vectors and all multi-history nearest-neighbour states. It is an alternative
exact sufficient-statistics implementation of the chat's cluster-moment idea;
the nearest-neighbour minimum cannot be recovered from a first moment alone.

## Other equivalent operations

- Remove unused representative fine-probe QK/pi_hat in the compact cosine path;
  retain the SAME residual Top-25% eligibility mask for Exact.
- Build a previous-bank-position lookup and gather by current cluster members.
  Per-cluster fixed Top-4 uses probability first, then previous bank position
  for exact stable tie breaking. All observations remain in the prior.
- Fuse state argmin/count and dual-bound updates. Keep sixteen iterations,
  the same lambda interval, first-state tie breaking, and FP32 operations
  without fused multiply-add for solver equivalence; no greedy allocator.
- Count/scan/scatter support directly in the original order (history slots,
  representatives, exact members); no full concatenated support tensor.
- Analytic mass transfer for Rep, single History, H+Rep and Exact. Multi-H
  uses the exact cached nearest assignment. When subtracting history mass
  would leave less than 1% of cluster mass, directly sum unselected priors to
  avoid cancellation amplification; this is a numerical guard, not a gate.
- Increase processing window from 8 to 64 query groups to reduce launches.
  Query groups themselves remain 32 queries, with the same membership.
  Original method defaults remain unchanged. All allocations and diagnostics
  count in full-forward timings.

The existing fused raw relative-logit evidence and second selected-QK pass for
exact probability/grid summaries are retained; no claim of eliminating that
pass. Dense and existing quality references are reused as instructed by the
user; reused timing provenance must be explicit. All quality evaluations
include AUC@3/AUC@30 and recon_unposed ACC/Comp/CD/F1, seven-scene macro only
after all seven finish. Diagnostic-only checks have six unavailable fields.

Unit/real-shape numerical comparisons precede quality evaluation. New dedicated
profiles must not substitute old output-proxy stage times for compact timings.
