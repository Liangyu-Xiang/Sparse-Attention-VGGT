# Preserved initial numerical diagnostic failure

Initial numerical oracle compared FP16 cached P to unquantized FP32 with an inappropriate3e-5 absolute tolerance; observed4.042685031890869e-5. v2 compares FP16 reference. Model implementation unchanged.

| AUC@3 | AUC@30 | ACC | Comp | CD | F1 | Dense speedup |
|---|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Failed small numerical diagnostic before any geometric quality evaluation
