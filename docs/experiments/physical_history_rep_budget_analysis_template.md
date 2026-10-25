# Physical History–Rep Budget Ablation Analysis Template

Copy to:

`outputs/physical_overlap_accounting/analysis.md`

and fill only from committed outputs.

## 1. Evaluated revision

- Commit SHA:
- Stage-0 py_compile:
- Stage-0 pytest:
- Config:
- Hardware:

## 2. Matched two-frame accounting sanity

| Stage | Mode | Optimizer cost | Physical keys | Utilization | Optimizer-physical gap | Unused budget | History/Rep candidate overlap | Merged H+Rep | H/Exact residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0-L8 | control | | | | | | | | |
| L0-L8 | physical | | | | | | | | |
| L9-L16 | control | | | | | | | | |
| L9-L16 | physical | | | | | | | | |
| L17-L23 | control | | | | | | | | |
| L17-L23 | physical | | | | | | | | |

Conclusion: did the accounting fix remove the large duplicate-cost gap?

## 3. History filtering after accounting correction

| Layer | Candidate density | Selected density | Recall | Precision | Enrichment | Mass |
|---|---:|---:|---:|---:|---:|---:|
| L9 | | | | | | |
| L10 | | | | | | |
| L11 | | | | | | |
| L12 | | | | | | |
| L13 | | | | | | |
| L14 | | | | | | |
| L15 | | | | | | |
| L16 | | | | | | |

State clearly whether the previous conclusion about non-monotonic sharpening
changes or remains.

## 4. Seven-scene macro quality

| Method | AUC@3 | AUC@30 | ACC | Comp | CD | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | | | | | | |
| SAF3R | | | | | | |
| Old reuse+residual | | | | | | |
| 57bdb78 | | | | | | |
| All-edge/no-coverage | | | | | | |
| Physical H/Rep accounting | | | | | | |

New minus all-edge/no-coverage:

- AUC@3:
- AUC@30:
- ACC:
- Comp:
- CD:
- F1:

## 5. Seven-scene layer behavior

| Stage | Physical utilization | Optimizer gap | Unused budget | History fraction | Merged H+Rep | Population Rep clusters | Exact | H/Exact residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0-L8 | | | | | | | | |
| L9-L16 | | | | | | | | |
| L17-L23 | | | | | | | | |

## 6. Runtime and memory

| Method | Mean scene median s | Peak allocated GiB | Peak reserved GiB | Actual density |
|---|---:|---:|---:|---:|
| All-edge/no-coverage | | | | |
| Physical H/Rep accounting | | | | |

Do not interpret this as the final efficiency result: Exploration work is still
not reduced as History grows.

## 7. Causal conclusions

### Budget accounting

- Supported / not supported:
- Evidence:

### Quality effect of restoring physical budget

- AUC/pose:
- Reconstruction:
- Per-scene anomalies:

### Remaining accounting issue

- H/Exact residual negligible / material:
- Evidence:

### History filtering

- unchanged / improved / degraded:
- Evidence:

## 8. Exactly one next recommendation

Recommendation:

Why this is now the highest-priority next experiment:

What must remain frozen in that experiment:
