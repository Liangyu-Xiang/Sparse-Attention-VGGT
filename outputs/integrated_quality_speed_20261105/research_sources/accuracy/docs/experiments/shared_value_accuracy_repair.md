# Shared Value Loss · Query-256 accuracy repair

The opt-in candidate is `configs/evaluation/reuse_shared_value_loss_256_repaired.json`.
It retains the 20% total selected-Key budget, original nominal cluster count,
Query-256 sampling, four states, head-mean Value features and 16 dual iterations.
The original configuration and defaults are preserved.

Two changes address measured attention error: compensate each covered cluster
by population / emitted members, and allow the Representative state when its
representative was observed in history. History now carries approximate cluster
population; its original dropped-member loss remains a proxy and is not an exact
model of this execution. No nearest-history owner search or extra QK is added.

The CUDA weight kernel reuses allocator counts and a ~1 MiB immutable lookup
table. Every table entry is bitwise equal to the corresponding GPU FP32 quotient.
The table is initialized within the first forward, outside graph capture, and
is reused during warmed forwards.

The final heads128 paired complete-forward medians were 17.381526 s original
versus 17.793197 s repaired: **2.37% more runtime**, or 0.9769x relative to the
original Shared Value method. This is not a speedup versus VGGT Dense; that is
N/A because no contemporaneous Dense timing was rerun. The first valid,
pre-lookup implementation measured 4.15% overhead and is also retained.

Seven 7Scenes scenes and an ETH3D three-scene subset completed pose and unposed
reconstruction evaluation. Quality improves unevenly, so this is **not a
quality-equivalent replacement for Dense**, and it is not promoted to a default.
Pumpkin's local sequence has missing source frames; its reused Dense control
was selected by the exact same input list and protocol, not by its scores.

Full six-metric tables, scene rows, scope-matched macro averages, per-layer
attention diagnostics, timing, memory, limitations and reference provenance:

[Experiment report](../../outputs/shared_value_accuracy_20261105/report.md)

Run from the repository root with an available GPU:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_shared_value_repair.py -v

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_shared_value_accuracy.py \
  --frames 31 --dense-trajectory \
  --config configs/evaluation/reuse_shared_value_loss_256_repaired.json \
  --output outputs/new_repair_attention_audit

python scripts/run_gpu_exclusive.py \
  --gpu 0 --monitor outputs/new_repair_monitor.json --log outputs/new_repair.log -- \
  python scripts/evaluate_zdrop_history_states.py \
  --dataset eth3d --scene kicker --frames 128 --method zdrop_history_states \
  --config configs/evaluation/reuse_shared_value_loss_256_repaired.json \
  --output-dir outputs/new_repair_quality --warmup 1 --repeats 3
```

`rep_m5c_relaxed` is not substituted with another reuse method: its specified
configuration and implementation were absent from all fetched visible remote
branch trees. Consequently its mask alignment is still unavailable.
