# Camera token diagnosis: A + all-special correction + cluster_fraction=0.20

Scope: ETH3D filtered11, 377 exact matched frames; 24 frame + 24 global blocks, all16 heads and camera rows. Layer numbers below are 1-based. Dense is the feature reference (cosine1, relativeL2=0). No production code/config change.

Attention metrics compare effective Value-source token probability distributions. SAF3R broadcast_first uses first-frame scoring Q/K but target-frame Values; score ownership is retained separately in raw tensor summaries. SAF3R is local Top-4 lazy first-correspondence-head, not paper DINO Top-K.

Cosine = <x,d>/(||x||||d||); relative L2 = ||x-d||/||d||. Feature rows flatten all frame camera vectors per scene, then average scenes equally. Attention rows average head/query metrics within scene, then scenes equally. JS uses natural logarithms. Similarity is diagnostic, not pose accuracy.

| Global layer | Ours block cos | Ours block relL2 | SAF block cos | SAF block relL2 | Ours attention JS | SAF attention JS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 0.000937 | 0.992600 | 0.121340 | 0.000000 | 0.596870 |
| 2 | 0.999998 | 0.002171 | 0.993403 | 0.116005 | 0.000000 | 0.617651 |
| 3 | 0.999780 | 0.020799 | 0.997431 | 0.072366 | 0.000121 | 0.615110 |
| 4 | 0.999948 | 0.010330 | 0.999328 | 0.037861 | 0.000355 | 0.599724 |
| 5 | 0.999910 | 0.013551 | 0.999304 | 0.038080 | 0.000258 | 0.601078 |
| 6 | 0.999904 | 0.013957 | 0.999237 | 0.040718 | 0.000428 | 0.544546 |
| 7 | 0.999878 | 0.015757 | 0.999108 | 0.044160 | 0.000488 | 0.566487 |
| 8 | 0.999872 | 0.016113 | 0.999051 | 0.045132 | 0.000453 | 0.588235 |
| 9 | 0.999858 | 0.016930 | 0.999105 | 0.043717 | 0.000125 | 0.610869 |
| 10 | 0.999847 | 0.017555 | 0.999163 | 0.042341 | 0.000286 | 0.281391 |
| 11 | 0.999856 | 0.017055 | 0.999140 | 0.043039 | 0.000208 | 0.268225 |
| 12 | 0.999836 | 0.018147 | 0.998589 | 0.054100 | 0.000620 | 0.236708 |
| 13 | 0.999785 | 0.020896 | 0.997821 | 0.066836 | 0.002300 | 0.254639 |
| 14 | 0.999473 | 0.032152 | 0.996896 | 0.078903 | 0.003870 | 0.200793 |
| 15 | 0.998904 | 0.045709 | 0.996258 | 0.086416 | 0.003894 | 0.113256 |
| 16 | 0.998282 | 0.056940 | 0.994338 | 0.105755 | 0.010217 | 0.172631 |
| 17 | 0.995967 | 0.086536 | 0.990853 | 0.133786 | 0.026589 | 0.218584 |
| 18 | 0.984194 | 0.163657 | 0.987761 | 0.154504 | 0.014542 | 0.311623 |
| 19 | 0.986287 | 0.152912 | 0.985726 | 0.166294 | 0.000559 | 0.355709 |
| 20 | 0.985344 | 0.159897 | 0.982864 | 0.181906 | 0.006355 | 0.217141 |
| 21 | 0.985999 | 0.157583 | 0.983459 | 0.178523 | 0.002734 | 0.340821 |
| 22 | 0.986879 | 0.152710 | 0.985625 | 0.164115 | 0.004897 | 0.268163 |
| 23 | 0.985130 | 0.165028 | 0.985933 | 0.161231 | 0.000905 | 0.353696 |
| 24 | 0.989181 | 0.139711 | 0.986372 | 0.157486 | 0.001079 | 0.340937 |

## Camera head input, per scene

| Scene | Ours cosine | Ours relL2 | SAF cosine | SAF relL2 |
|---|---:|---:|---:|---:|
| facade | 0.996329 | 0.085999 | 0.992085 | 0.125573 |
| terrains | 0.984354 | 0.176270 | 0.953541 | 0.301264 |
| delivery_area | 0.980118 | 0.198544 | 0.986734 | 0.162349 |
| electro | 0.991842 | 0.127628 | 0.989907 | 0.141802 |
| courtyard | 0.994209 | 0.107599 | 0.988613 | 0.150675 |
| playground | 0.991597 | 0.129535 | 0.984059 | 0.178322 |
| kicker | 0.995078 | 0.099248 | 0.993863 | 0.110706 |
| office | 0.995922 | 0.090251 | 0.993635 | 0.112696 |
| relief_2 | 0.977800 | 0.210337 | 0.983148 | 0.184067 |
| relief | 0.977766 | 0.209925 | 0.988538 | 0.151142 |
| pipes | 0.981249 | 0.193603 | 0.992152 | 0.125827 |
| Equal-scene macro | 0.987842 | 0.148085 | 0.986025 | 0.158584 |

## Six required quality metrics and speed

| Method | AUC@3 (fraction) | AUC@30 (fraction) | ACC | Comp | CD | F1 (fraction) | Dense full-forward speedup |
|---|---|---|---|---|---|---|---|
| Dense | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ours | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SAF3R | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

All six unavailable: this is a camera attention/features diagnostic, without GT pose scoring or unposed reconstruction. Per-run quality.json/quality.md record that reason. Instrumented runtime/peak memory are in instrumented_resources.csv; production speedup unavailable because counterfactuals, CPU transfer and full attention reconstruction change execution. Dense=1.00x only defines the production speed reference; no paired warmed production timings were measured here. No accuracy claim follows from cosine alone.

Raw per-scene JSON and camera_tensors.pt are retained locally. Tracked CSVs cover frame/global layers, all captured stages, Q/K/V, attention heads, same-QKV operator checks and counterfactuals. Direct ours-vs-SAF feature relativeL2 uses SAF as denominator. Counterfactual effects are nonlinear conditional sensitivities, not additive causal percentages.

Postprocessing recovery: all33 model diagnostics succeeded; initial coordinator ended with missing report module. This CPU reporter was then added and run without rerunning model forwards.
