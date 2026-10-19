# Fused full-coverage Value / evolving Attention

Shared32-query support, manual0/20/80/20%Attention schedule, all24global layers; actual per-layer retention is recorded in configs. No Dense seed/fallback. Primary warmed timings include all state/reset/selection costs; independent profiles are not primary samples.

Full acceptance: **False**. Complete seven-scene128-frame scope: False. Quality gate: False; speed gate: False; memory gate: False.

| Scene | Dense(s) | Fused(s) | Dense GiB | Fused GiB |
|---|---:|---:|---:|---:|
| chess | 0.2025 | 0.3780 | 6.998 | 6.999 |

Paired mean-time ratio Dense/fused: 0.5358x. Missing: [].
An unavailable quality gate is NOT a pass. Small-input pose checks are not dataset accuracy. Terminal selection performs extra probe QK; its winning final QK is currently repeated and counted. Value local-source groups and spatial probe pooling are a new explicit approximation, not bitwise replay of the old branch.

## Independent profile (seconds; inclusive selection includes terminal)

```json
{
  "chess": {
    "global_module": 0.16774243092536922,
    "value": 0.04573404771089554,
    "pack": 0.0012124160043895246,
    "execution": 0.006157375946640969,
    "history": 0.014280575990676881,
    "pool": 0.009669536054134368,
    "terminal": 0.004865056008100509,
    "selection_including_terminal": 0.05034790444374085
  }
}
```
