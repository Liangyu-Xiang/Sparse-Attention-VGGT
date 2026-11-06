# Global 10% final-output retention assessment

User requested 10% across all 24 global layers, not 10% independently at every
layer. Start from the frozen fast14 current-probe backend, cluster_fraction=.10.
No hyperparameter search: scale the existing10/18/10% schedule by5/7 before any
target quality, giving7.142857/12.857143/7.142857%. All existing probability,
Value representation,History A,compensation,selection and16-round solver settings
stay fixed. All5 special Query outputs per frame retain their full-key correction.

The new opt-in accounting includes those corrected rows in the SAME budget.
For N tokens,D corrected queries,S special keys and layer output targetr,
patch cap=floor((r*N*N-D*N)/(N-D))-S. Thus the final unique output support is
bounded by(N-D)*(patchcap+S)+D*N<=r*N*N. State choices can underspend;actual
measured density is reported rather than claiming exactly10%. Do not fill with
unneeded keys to make a nominal count. The old grouping budget excluded this
special-query expansion;old fast14 naming is retained but actual density shown.

Probe QK and repeated execution for corrected queries are separate physical
work:time and total logical QK include them. Neither is hidden as10% output
support. Kernel padding FLOPs are not measured by logical edge counts.

Quality scope:7Scenes all7scenes128frames,ETH3D same11scenes14–76frames as the
existing complete inputs. Same checkpoint SHA,frame lists,preprocessing,pose,
fusion/alignment and reconstruction thresholds. Save all six metrics,per-scene
rows,equal-scene macro means,runtime,memory,budget and GPU isolation reports.
No unchanged Dense,SAF3R,A12 or fast14 reruns;reuse historical complete results.
Formal paired Dense speedup N/A. Original fast14 times are previous-batch
references,not contemporaneous paired measurements. No target-GT retuning.

22 regression tests pass,covering the new complete-output budget bound,tail
geometry,default behavior,FP32/BF16 probability and sparse attention. This is
budget-accounting/config work,not a new attention kernel. Defaults keep the
old budget rule;the new field requires the current-probe backend.

## Completed result

18/18 pose+unposed reconstruction evaluations and source/config/input/protocol,
isolation and budget audits passed. Actual final unique retention macro:7Scenes
9.96487%,ETH3D9.97744%;total logical QK including probes/repeated corrected
queries13.13114%/13.16128% of Dense QK,not10%.

7Scenes:AUC3/30=22.31147/84.42503%,ACC/Comp/CD=.14243/.12513/.13378m,
F1=47.10222%,5.34518s warm median scene mean,14.865GiB warm allocated peak.
ETH3D:24.59474/78.32003%,.35891/.90942/.63416m,54.31966%,1.12759s,11.023GiB.
Compared with fast14 reference,F1 drops2.456%/5.336% relative;ETH3D ACC worsens
12.430% and CD5.630%.7Scenes Comp worsens2.524%,despite improved ACC. Thus this
fixed schedule does not maintain quality;do not deploy it as a lossless
replacement. No budget retuning or old-control reruns follow these test scores.
Reference time is preceding-batch5.74509s/1.15039s,not paired. Formal Dense
speedup N/A. See outputs/current_probe_global10_20261106/REPORT.md for all six
metrics,per-scene rows,budget/state counts,protocol and runtime/memory details.
