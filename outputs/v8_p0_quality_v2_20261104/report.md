# Rejected initial P0: partial quality archive

Evaluation incomplete: stopped remaining queued scenes after routing equivalence failed at L4. All completed scores retained; no repeated successful scene was selected. Completed7Scenes all7 and ETH3D courtyard1/11. ETH3D macro unavailable because10scenes were not run.

|dataset|scene|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|
|---|---|---|---|---|---|---|---|
|7scenes|office|11.8643|75.7792|0.048008|0.051874|0.049941|62.4256|
|7scenes|pumpkin|10.2813|77.2478|0.413964|0.086284|0.250124|28.3565|
|7scenes|fire|52.8174|94.2926|0.054339|0.051789|0.053064|59.7669|
|7scenes|redkitchen|7.5869|79.7843|0.080690|0.078237|0.079464|46.0905|
|7scenes|stairs|5.9260|80.7968|0.225817|0.460345|0.343081|15.4215|
|7scenes|chess|16.6339|87.3577|0.136143|0.087147|0.111645|37.5149|
|7scenes|heads|52.2925|93.9505|0.035607|0.033030|0.034319|83.1412|
|eth3d|courtyard|29.5875|88.3594|0.225801|0.145082|0.185441|70.7163|
|7scenes|equal-scene macro,7/7|22.4861|84.1727|0.142081|0.121244|0.131662|47.5310|
|eth3d|macro N/A,1/11|N/A|N/A|N/A|N/A|N/A|N/A|

AUC/F1 percentages; distances metres,unsquared CD=(ACC+Comp)/2. Same pose+recon_unposed protocol and thresholds as final campaign. Full-forward timing belongs to initial p0 row in dense_sparse_comparison_20261104,not the corrected version. Per-scene single-forward quality timings are not paired speedups. Dense relative dataset speedup N/A: no paired dataset timing protocol.
