# Previous-layer logit residual transfer (centered050)

This is the evaluated research baseline, not a quality improvement over the fast version. Its ETH3D 11-scene macro AUC@3/AUC@30/F1 are 20.9442% / 77.3498% / 50.8996%; ACC/Comp/CD are 0.3818 / 0.9567 / 0.6693 metres. All six are worse than the matched historical fast control. Raw per-scene results, equal-scene averages, state distributions, timing, input lists, source hashes and protocol are in `outputs/logit_eth3d_20261108/`. Dense speedup is N/A because no contemporary paired Dense timing was run. Historical controls were reused, not rerun.

For each head and ordinary Query probe, cache only the immediately previous layer's actually executed sparse-mask Key IDs and delta = raw logit minus same-layer block-mean logit. For an observed Key, the next estimate is current block-mean logit + 0.5 * (previous delta - mean previous delta over observed Keys in that block). For an unobserved Key, use current block-mean logit. Reconstruct softmax per head and average all 16 heads and valid probes for the state weights. Each layer replaces the cache with its actual native logits; predicted or unexecuted values never become history.

Query groups have 256 tokens and four probes. Target Key block size is 1024; ETH3D actually uses 925 ordinary Keys per frame/block. Special tokens are excluded from ordinary estimation; formal attention keeps the existing special-token handling. There is no current-layer fine QK before mask selection, no rolling cache, no current representative observation and no extra anchor. Formal sparse AV exports its already computed logits for the next layer.

Drop/Rep/History/Exact, Value clustering, 10%/18%/10% layer retention and the geometric-mean population/prior-residual compensation are retained. `previous_logit_delta_forward.install('centered050')` explicitly opts into this variant; repository defaults are not changed. `one_layer_edge_history.py` supplies the valid-probe mean and evidence decoding helpers; its alternative forward is not installed by this method.

## Run

Use the repository evaluation environment with PyTorch, Triton, addict and Open3D. Provide the dataset under `workspace/benchmark_dataset/eth3d` and the checkpoint recorded in the frozen expected-input manifests (or an identical-SHA checkpoint at an updated local path). GPU 4/5/6 are the supported campaign allocation, one CUDA application per GPU.

```bash
python experiments/logit_eth3d/run_campaign.py --output outputs/logit_centered050_rerun/eth3d
```

The output directory must be new. This runs only centered050, with one warmup and three full-forward timing samples per scene, then pose and recon_unposed. The exact input lists and tensor shape, checkpoint SHA, 24-layer cache legality and no-current-fine-QK invariants are checked. The original result archive is never overwritten. The GPU-exclusive wrapper is bundled with this entry point; checkpoint and dataset files are not committed.

For a single scene:

```bash
python experiments/logit_eth3d/evaluate.py --variant centered050 --scene pipes --expected-input outputs/logit_eth3d_20261108/eth3d/expected_inputs/pipes.json --config configs/evaluation/logit_centered050_eth3d.json --output outputs/logit_centered050_pipes_rerun
```

Set CUDA_VISIBLE_DEVICES to an idle allocated GPU for the single-scene command. The campaign handles GPU allocation and monitoring automatically. Distances are unsquared metres, CD=(ACC+Comp)/2, AUC/F1 in raw JSON are fractions. ETH3D reconstruction threshold is 0.25m, downsampling/TSDF voxel 0.0390625m, sdf_trunc 0.2m, mesh sampling 1,000,000. This method is not SAF3R; Top-K/profile is N/A.
