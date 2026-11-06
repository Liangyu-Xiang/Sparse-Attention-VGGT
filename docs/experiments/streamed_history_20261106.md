# Historical Dense-probability reuse with contiguous Key blocks

Status:experimental,not final. This restores the user's defining principle:
previous-layer estimated Dense probabilities participate in next-layer decisions.
The full-current-Key probe backend is retained only as an information diagnostic.

Start from fast14 b1f12ba8;keep cluster_fraction=.10,head_sketch4,Query groups256,
10/18/10% layer budgets,History A,prior-residual execution,16-round allocator,
all-Exact eligibility and all5 special-query corrections. Only probability
acquisition/transfer changes. The failed global10 budget is not reused.

User suggested groups of64 or more Keys. Use contiguous64-Patch-Key blocks;
128/256 and old per-frame2x2 grids are independent RGB diagnostics. Partitions
are shared but each Query group has its own distribution. Key blocks do not
force blockwise final masks. Contiguous mean pooling replaces scatter pooling;
tail block sizes are counted correctly. One actual current Query/group observes
pooled blocks and Special Keys. No current full-Key probe.

The SAME formal QK tiles supply raw historical logits. Accumulate unweighted
selected logsumexp inline,combine with pooled missing-member logsumexp before
Head averaging,then immediately form head-mean FP16 conditional caches and
calibrated block mass. Do not use compensated sparse probabilities as Dense
probabilities. Missing-member estimates are still approximate,not a Dense oracle.
No second full selected-edge QK scan. Coarse Special-Key observations overlap a
small subset of formal QK and are counted in coarse work,not hidden as free.

Per-head raw encoding is temporarily needed inside one window;there is no
cross-layer head correspondence or headwise persistent probability cache. FP16
was faster than8-bit after including cache completion and avoids8-bit error.
Neither zero extra global memory nor zero runtime overhead is claimed. The old
coarse-proxy direct headmean prototype is preserved for reproducibility and
rejected by actual Dense-probability diagnostics. Some helper module headers
still describe that retained prototype;the selected production branch is
streamed_history_normalizer='selected_d2',not'coarse_proxy'.

Transfer calibrated previous block mass by current/previous head-mean coarse
ratios,normalize with a Special-mass bucket,then transfer block conditionals with
smooth symmetric confidence. Blend the migrated probability distribution with
current coarse distribution. A positive global history weight.25 is chosen on
independent ScanNet RGB exact sampled-QKV probability error,not7Scenes/ETH3D
geometry truth. This is distribution shrinkage,not uniform probability scaling.
Even at.25,the only within-block nonuniform evidence is historical probability.
Changing probability caches with fixed current observations/IDs changes the next
prior;tests explicitly enforce this. First layer uses coarse prior without history.

Frozen independent TUM128 validation:history-key relative L1=.536156,coarse-only
=.601592,unchanged history=.928030. Runtime5.85707s,only new method measured.
No target geometric GT used in selection;no retuning after target scores.

24 regression tests pass,including real selected partitions,strong compensation
removal,FP16/8-bit error,headless cache,history dependency,tail Key blocks,region
transport and convex mixing. Same-QKV/same-Mask128-frame kernel diagnostic has
max relative output L2=0:the attention kernel itself agrees with bare AV on
that trajectory. Its event medians sum to bareAV1.26004s/export1.51526s,increment
.25522s(20.26%). This is a separate instrumented diagnostic,not full-model time.
Independent disjoint profile:AV+export1.47253s,cache finalize.01678s,transfer
.01851s,missing normalization.01372s,coarse observation.04859s. Do NOT add the
separate.25522s to this profile or label AV+export as bare AV.

18/18 quality runs complete:7Scenes7x128,ETH3D11x14–76,same checkpoint SHA,
frame lists,preprocessing,pose,unposed alignment/fusion and reconstruction
thresholds. AUC/F1 percentages;distancesm;CD=(ACC+Comp)/2 unsquared.
7Scenes:AUC3/30=22.74794/84.28155%,ACC/Comp/CD=.14904/.12050/.13477m,
F1=48.21156%,5.85238s,warm peak14.954GiB.
ETH3D:21.22566/77.03620%,.33423/.92011/.62717m,56.57975%,1.18305s,
warm peak11.023GiB. Relative to full-Key fast14,ETH pose/reconstruction and
7Scenes ACC regress;cannot call this final. Compared to historical A12,most
point-cloud metrics improve but ETH AUC3 drops. Geometry quality is not guaranteed
by improved sampled probability error. No further budget/parameter tuning follows
these test scores.

Old A12/full-Key/Dense/SAF3R results are reused,not rerun. Old target runtimes are
preceding-batch references,not paired speed claims. Formal Dense speedupN/A.
Full six-metric tables,per-scene/layer CSV,raw results,hashes,source snapshots,
monitors,failed prototypes and diagnostic scopes are in
outputs/streamed_history_20261106/REPORT.md.
