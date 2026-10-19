# Fused full-coverage Value / evolving Attention

Shared32-query support, manual0/20/80/20%Attention schedule, all24global layers; actual per-layer retention is recorded in configs. No Dense seed/fallback. Primary warmed timings include all state/reset/selection costs; independent profiles are not primary samples.

Full acceptance: **False**. Complete seven-scene128-frame scope: False. Quality gate: False; speed gate: False; memory gate: False.

| Scene | Dense(s) | Fused(s) | Dense GiB | Fused GiB |
|---|---:|---:|---:|---:|
| chess | 4.3134 | 4.0387 | 10.035 | 10.047 |

Paired mean-time ratio Dense/fused: 1.0680x. Missing: [].
An unavailable quality gate is NOT a pass. Small-input pose checks are not dataset accuracy. Terminal selection performs extra probe QK; its winning final QK is currently repeated and counted. Value local-source groups and spatial probe pooling are a new explicit approximation, not bitwise replay of the old branch.

## Independent profile (seconds; inclusive selection includes terminal)

```json
{
  "chess": {
    "global_module": 1.8489907188415526,
    "value": 0.01952559992671013,
    "pack": 0.0014919039998203519,
    "execution": 0.5658352957963944,
    "history": 0.10848188801109793,
    "pool": 0.011512928158044814,
    "terminal": 0.30074409532547003,
    "selection_including_terminal": 0.787859873175621
  }
}
```
