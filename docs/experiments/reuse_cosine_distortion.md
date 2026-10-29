# History-conditioned cosine-distortion objective and mass-transfer states

Source: user-referenced chat 6ae08705-de38-83e8-b077-997f1627b1c6, latest response. Exact reference text: reuse_cosine_distortion_reference.md. Earlier contribution-mean/output-norm suggestions in that chat are superseded. User explicitly requested implementing both new losses and corresponding mass transfer, rather than a loss-only ablation.

## Fixed settings

7Scenes all seven scenes, uniform128frames, same checkpoint/preprocessing/pose/recon_unposed protocol; original2x2canonical grid,32-query groups,20%upper edge budget,4 historical candidates per current fine cluster,25%fine-probe/Exact eligibility,16dual iterations, original second-QK evidence kernel. No shift8/fine-grid/single-QK/max-error combination. Fresh same-GPU Dense/original controls; SAF3R DINO per-frame64 is a separately labelled comparison. Original profile/configuration remains available unchanged under defaults.

## Loss

Features remain the original64-D arithmetic head-mean of current raw V; z is the per-token L2-normalized feature, not a normalized cluster mean. Distortion is sum_j a_hat_j (1 - dot(z_j,z_representative(j))). Self distortion is explicitly zero. Clamp numerical cosine distortion to[0,2]. Normalization epsilon1e-12 handles finite zero inputs; zero-vector angular similarity has no geometric guarantee. Cosine loss ignores vector magnitude and is not an exact Dense output-error oracle.

- Drop:2*pi, cost0.
- Rep: all cluster members represented by the original centroid-nearest REAL member; cost1.
- History(t): same descending-utility candidate prefix as original, t=1..4; each member assigned to its nearest selected historical direction. Selected Keys own themselves (self wins exact-direction ties); deterministic earlier candidate wins other ties. Cost t.
- History(t)+Rep: selected historical members own themselves; remaining members assigned to Rep. Cost t+1; original n>t feasibility retained.
- Exact: all real cluster members, zero distortion, cost n; original residual fine-probe eligibility retained.

The budget solver still minimizes total loss subject to <=B; it remains the original separable discrete Lagrange approximation, not an exact global optimizer. Layer0 remains the original representative-only seed.

## Common probability prior

pi is the original coarse-canonical-to-fine cluster mass, not pi_hat for the probed Rep. All valid positive beta-scaled genuine previous-Key observations inform the prior, including those excluded from the selectable history prefix. Overfull observed mass is scaled down to pi; if every member is observed, their masses are normalized to pi. Unknown members share the residual uniformly. The same prior is used for every state. Per-cluster prior sums equal pi.

A numerical full-support mix is fixed before evaluation: a=(1-1e-6)*a_raw +1e-6*pi/n. This permits finite mass transfer to a representative with zero raw prior. This is a declared engineering completion of the chat's zero-prior case, not a test-truth-tuned parameter. pi=0 clusters have zero loss and prefer free Drop.

The original fine-probe computation remains, including pi_hat, but pi_hat does not enter the new loss. Its cost remains included; no unreported removal of original estimator costs.

## Execution and genuine evidence

For emitted Key k, M_k=sum of estimated member masses assigned to k. Use multiplier w_k=M_k/a_k >=1 in the existing unified sparse softmax via log(w_k). Thus a_k*w_k=M_k; represented cluster mass is preserved under the estimated prior, and Exact has unit multipliers. Drop transfers no mass. Values and current QK are recomputed as usual; current QK can differ from the estimated prior, so final probabilities are not claimed equal to true Dense probabilities or forcibly equal to the proxy masses.

The final selected-Key probability p_k includes transferred mass. Only p_k/w_k is genuine unit-Key evidence. Save this for selected genuine history/Exact Keys; pure representative states still create no genuine history. Full executed canonical mass uses p_k, genuine canonical mass uses p_k/w_k. This prevents transferred mass from being presented as a freshly observed edge. Second selected-QK evidence overhead is retained.

## Implementation and validation

Exact tokenwise state losses use a token-parallel FP32 Triton kernel, avoiding giant group*cluster*member*feature intermediates. A second token pass computes the selected state's mass assignment. No full NxN attention or test truth is used. CPU explicit reference covers all states; GPU tests compare losses/assignments to it, check mass conservation, self ownership, invalid states, zero cluster mass, fully observed evidence, the90%-history example, mean-cancellation counterexample, transfer under prior logits, and24-layer support/evidence invariants.

Report AUC@3/AUC@30 and ACC/Comp/CD/F1 for every run, per-scene and equal-scene macro. Reconstruction units metres, threshold0.05m, downsample0.0078125m, CD=(ACC+Comp)/2 unsquared. Record raw timings/peak memory, exact inputs/config/source snapshots, history-selection counts and paired Dense speedup. OOM/diagnostics/interrupted runs retain six unavailable fields with explicit reasons; never label them complete quality evaluations.

## Layout regression found and corrected

The actual top_history return excludes a sentinel column, so group rows have a one-slot stride gap. Custom cosine kernels now explicitly take contiguous copies of hids. Regression tests use row-padded views rather than only compact fixtures. Initial cosine runs and early diagnostics using the incorrect layout are excluded and archived; Dense, SAF3R and original controls never execute this new helper and remain valid. A full128-frame diagnostic checks actual intermediate losses against an explicit CPU member reference and executed transfer mass conservation on layers2/12/14/23 before quality reruns.
