# Stage0 validation

Upstream HEAD and remote branch matched at0856426c0537195255946b356f027416d349be5b.
Both handoff and analysis template were read in full before execution.
Original six diagnostic scripts passed py_compile. Original production tests:
9passed,1failed due to unsupported BF16 sparse CUDA addmm (upstream_tests.log).
Diagnostic-only compatibility plugin:10passed (compat_tests.log).
No production selector, Value clustering, scoring, population, grouping, or
Rep/Exact implementation is modified. The diagnostic process only disables
autocast around sparse.mm and loads the previously validated archived padded
four-effective-probe kernel. No16-probe approximation is introduced.

Diagnostic corrections: QK transpose (pipeline previously emitted64x64 instead
of Npatch x Npatch); explicit repo import path; tie-aware Spearman ranks.
Additional diagnostic tests check QK support accounting and average tie ranks.
Additional required scatter plots/reporting are generated from unchanged raw CSV.
All diagnostic/microbenchmark runs report AUC@3,AUC@30,ACC,Comp,CD,F1 and paired
Dense speedup as N/A with explicit diagnostic-only reasons. These are not quality
evaluations and their instrumented runtime is not deployed full-forward speed.
