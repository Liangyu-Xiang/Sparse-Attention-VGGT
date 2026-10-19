# Fused full-coverage Value / evolving Attention

Shared32-query support, manual0/20/80/20%Attention schedule, all24global layers; actual per-layer retention is recorded in configs. No Dense seed/fallback. Primary warmed timings include all state/reset/selection costs; independent profiles are not primary samples.

Full acceptance: **False**. Complete seven-scene128-frame scope: False. Quality gate: False; speed gate: False; memory gate: False.

| Scene | Dense(s) | Fused(s) | Dense GiB | Fused GiB |
|---|---:|---:|---:|---:|
| chess | 16.6402 | 13.9401 | 14.359 | 14.927 |

Paired mean-time ratio Dense/fused: 1.1937x. Missing: [].
An unavailable quality gate is NOT a pass. Small-input pose checks are not dataset accuracy. Terminal selection performs extra probe QK; its winning final QK is currently repeated and counted. Value local-source groups and spatial probe pooling are a new explicit approximation, not bitwise replay of the old branch.

## Independent profile (seconds; inclusive selection includes terminal)

```json
{
  "chess": {
    "global_module": 9.06688296508789,
    "value": 0.04837180864810943,
    "pack": 0.004154304005205631,
    "execution": 3.806263075590133,
    "history": 0.5592918083071708,
    "pool": 0.03397532784938813,
    "terminal": 1.8065961911082267,
    "selection_including_terminal": 3.6292727344036106
  }
}
```
