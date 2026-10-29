"""Layer-shared Value representatives: dense seed at L2, sparse L3--L9."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .value_clustering import distortion, local_refine, source_neighborhoods
from .value_medoids import blocked_distances, eager_pam, host_distances


def token_layout(frames, patches_per_frame, special_per_frame, device):
    ids = torch.arange(frames*(patches_per_frame+special_per_frame), device=device)
    special = ids.remainder(patches_per_frame+special_per_frame) < special_per_frame
    return ids[~special], ids[special]


def representative_attention(q, k, v, patch_indices, special_indices, medoids, scale=None):
    """Keep ALL queries; gather the same representative K/V rows for every head.

    Special tokens are retained as keys. Special queries use this SAME selected
    key set, so no dense special-query branch or full NxN mask is evaluated.
    """
    medoids = torch.as_tensor(medoids, dtype=torch.long, device=q.device)
    selected = torch.cat((special_indices, patch_indices[medoids])).sort().values
    return F.scaled_dot_product_attention(q, k.index_select(2, selected), v.index_select(2, selected),
                                          dropout_p=0.0, scale=scale)


class LayerValueState:
    def __init__(self, retention=.10, neighbors=8, iterations=2, sparse_start=3, sparse_end=9,
                 seed_backend="host", max_seed_passes=100, output_dir=None, save_values=False,
                 local_backend="cpu", seed_cache=None, profile_cuda=False,
                 representative_update="exact", sampled_candidates=4, capture_diagnostics=False):
        if not 0 < retention <= 1 or neighbors < 0 or iterations < 1:
            raise ValueError("Invalid retention/neighbors/iterations")
        if not 2 <= sparse_start <= sparse_end <= 24:
            raise ValueError("Sparse layers must lie in 2..24 with a preceding dense seed layer")
        if seed_backend not in {"device", "host"}:
            raise ValueError("seed_backend must be 'device' or 'host'")
        if local_backend not in {"cpu", "gpu"}:
            raise ValueError("local_backend must be cpu or gpu")
        if representative_update not in {"exact", "previous", "centroid", "sampled"}:
            raise ValueError("Unknown representative update")
        if representative_update != "exact" and local_backend != "gpu":
            raise ValueError("Approximate representative updates require the GPU backend")
        if sampled_candidates < 2:
            raise ValueError("sampled_candidates must be at least two")
        self.representative_update, self.sampled_candidates = representative_update, sampled_candidates
        self.capture_diagnostics = capture_diagnostics
        if local_backend == "gpu" and output_dir is not None:
            raise ValueError("GPU timing path exports diagnostics after forward, not inside it")
        self.local_backend, self.seed_cache, self.profile_cuda = local_backend, seed_cache, profile_cuda
        self.retention, self.neighbors, self.iterations = retention, neighbors, iterations
        self.sparse_start, self.sparse_end = sparse_start, sparse_end
        self.seed_layer = sparse_start-1
        self.seed_backend, self.max_seed_passes = seed_backend, max_seed_passes
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.save_values = save_values
        self.forward_id = 0
        self.reset(1, 1, 1)
        self.forward_id = 0

    def reset(self, frames, patch_height, patch_width):
        self.frames, self.patch_height, self.patch_width = frames, patch_height, patch_width
        self.previous_values = self.labels = self.medoids = self.previous_layer = None
        self.records = []
        self.selection_events, self.attention_events = [], []
        self.seed_values = None
        self.forward_id += 1

    def _seed(self, mean_values):
        n = len(mean_values)
        k = max(1, round(self.retention*n))
        if self.neighbors >= k:
            raise ValueError(f"q={self.neighbors} requires K>q; got N={n}, K={k}")
        if k == n:
            return np.arange(n), np.arange(n), {"algorithm": "all patch tokens (exact zero distortion)", "converged": True}
        if k < 2:
            costs = torch.cdist(mean_values.double(), mean_values.double()).sum(1)
            return np.zeros(n, dtype=np.int64), np.array([int(costs.argmin())]), {"algorithm": "exact one medoid", "converged": True}
        if mean_values.is_cuda and self.seed_backend == "device":
            free, _ = torch.cuda.mem_get_info(mean_values.device)
            required = 4*n*n + 1024**3
            if free < required:
                raise RuntimeError(f"Global L{self.seed_layer} needs at least {required/2**30:.2f} GiB free; use seed_backend='host' for the exact host-distance path")
        hybrid = mean_values.is_cuda and self.seed_backend == "host"
        distance = host_distances(mean_values, mean_values.device) if hybrid else blocked_distances(mean_values)
        starts = [eager_pam(distance, k, seed=20261015+100*self.seed_layer+10000*trial,
                            max_passes=self.max_seed_passes, candidate_batch=128,
                            compute_device=mean_values.device if hybrid else None) for trial in range(2)]
        fit = min(starts, key=lambda r: r["objective"])
        if not fit["converged"]:
            raise RuntimeError("Dense seed PAM did not converge; not using an incomplete seed")
        stats = {key: value for key, value in fit.items() if key not in {"medoids", "assignments", "distances"}}
        stats["start_objectives"] = [r["objective"] for r in starts]
        return fit["assignments"], fit["medoids"], stats

    def select(self, mean_values, layer_1based):
        if self.local_backend == "gpu":
            return self._select_gpu(mean_values, layer_1based)
        started = time.perf_counter()
        current = mean_values.detach().float().cpu().numpy().copy()
        source_layer = self.previous_layer
        if layer_1based == self.seed_layer:
            if source_layer is not None:
                raise RuntimeError("Seed reused without resetting the input sequence")
            labels, medoids, details = self._seed(mean_values)
            kind = "global_seed_dense_attention"
            anchors, snapshots = None, {}
        else:
            if source_layer != layer_1based-1 or self.previous_values.shape != current.shape:
                raise RuntimeError("Local selection must inherit the immediately previous online layer")
            anchors = self.medoids.copy()
            neighborhoods = source_neighborhoods(self.previous_values, anchors, self.neighbors)
            fit = local_refine(current, self.labels, anchors, neighborhoods,
                               max_iterations=self.iterations, anchors=anchors,
                               save_iterations=range(1, self.iterations+1), stop_when_converged=False)
            labels, medoids = fit["assignments"], fit["medoids"]
            details = {"history": fit["history"], "iterations": fit["iterations"],
                       "migration_fraction": fit["migration_fraction"], "fixed_point": fit["converged"]}
            snapshots = fit["snapshots"]
            np.testing.assert_array_equal(labels[anchors], np.arange(len(anchors)))
            kind = "recursive_neighbor_local_sparse_attention"
        objective, distances = distortion(current, labels, medoids)
        counts = np.bincount(labels, minlength=len(medoids))
        if not (counts > 0).all() or len(np.unique(medoids)) != len(medoids):
            raise RuntimeError("Representative selection changed K or produced an empty cluster")
        record = {"global_layer_1based": layer_1based, "source_layer_1based": source_layer,
                  "kind": kind, "n": len(current), "k": len(medoids), "retention": len(medoids)/len(current),
                  "heads": "mean raw V; one partition and shared token indices for all heads",
                  "q": self.neighbors, "distortion": objective/len(current),
                  "selection_seconds": time.perf_counter()-started, **details}
        self.previous_values, self.labels, self.medoids = current, labels, medoids
        self.previous_layer = layer_1based
        self.records.append(record)
        if self.output_dir is not None:
            directory = self.output_dir / f"forward_{self.forward_id:04d}" / f"layer_{layer_1based:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / "clusters.npz", assignments=labels, medoids=medoids,
                                distances=distances, cluster_sizes=counts,
                                anchors=np.array([], dtype=np.int64) if anchors is None else anchors)
            for iteration, snapshot in snapshots.items():
                np.savez_compressed(directory / f"iteration_{iteration}.npz", assignments=snapshot["assignments"], medoids=snapshot["medoids"])
            if self.save_values:
                np.save(directory / "mean_values.npy", current)
            (directory / "clustering.json").write_text(json.dumps(record, indent=2)+"\n")
        return medoids

    def _select_gpu(self, mean_values, layer_1based):
        from .value_clustering_gpu import local_refine_gpu, source_neighborhoods_gpu
        current = mean_values.detach().float()
        source_layer = self.previous_layer
        details = {}
        if layer_1based == self.seed_layer:
            if source_layer is not None:
                raise RuntimeError("Seed reused without resetting the input sequence")
            self.seed_values = current
            if self.seed_cache is None:
                labels, medoids, _ = self._seed(current)
                labels = torch.as_tensor(labels, device=current.device, dtype=torch.long)
                medoids = torch.as_tensor(medoids, device=current.device, dtype=torch.long)
            else:
                labels, medoids = self.seed_cache["assignments"], self.seed_cache["medoids"]
                if labels.device != current.device or len(labels) != len(current):
                    raise ValueError("Seed cache device or input token count mismatch")
            kind = "cached_global_seed_dense_attention" if self.seed_cache is not None else "global_seed_dense_attention"
        else:
            if source_layer != layer_1based-1 or self.previous_values.shape != current.shape:
                raise RuntimeError("Local selection must inherit the immediately previous online layer")
            neighbors = source_neighborhoods_gpu(self.previous_values, self.medoids, self.neighbors)
            fit = local_refine_gpu(current, self.labels, self.medoids, neighbors, self.iterations,
                                   representative_update=self.representative_update,
                                   sampled_candidates=self.sampled_candidates)
            labels, medoids = fit["assignments"], fit["medoids"]
            details = {"objective_tensor": fit["objective"], "member_pair_counts": fit["member_pair_counts"],
                       "representative_distance_counts": fit["representative_distance_counts"]}
            kind = "gpu_recursive_neighbor_local_sparse_attention"
        self.previous_values, self.labels, self.medoids = current, labels, medoids
        self.previous_layer = layer_1based
        self.records.append({"global_layer_1based": layer_1based, "source_layer_1based": source_layer,
                             "kind": kind, "n": len(current), "k": len(medoids),
                             "q": self.neighbors, "local_backend": "gpu", **details})
        self.records[-1].update(representative_update=self.representative_update,
                                sampled_candidates=self.sampled_candidates)
        if self.capture_diagnostics:
            self.records[-1].update(values_tensor=current, labels_tensor=labels, medoids_tensor=medoids)
        return medoids


class LayerValueAttention(nn.Module):
    def __init__(self, original, *, state, layer_idx, special_per_frame=5):
        super().__init__()
        self.original, self.state, self.layer_idx = original, state, layer_idx
        self.special_per_frame = special_per_frame

    def forward(self, x, pos=None):
        if self.training or x.shape[0] != 1:
            raise ValueError("Layer Value attention supports inference with batch size 1")
        original, state = self.original, self.state
        b, n, c = x.shape
        patches_per_frame = state.patch_height*state.patch_width
        if n != state.frames*(patches_per_frame+self.special_per_frame):
            raise ValueError("Token layout changed without resetting the sequence geometry")
        qkv = original.qkv(x).reshape(b, n, 3, original.num_heads, original.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        patches, specials = token_layout(state.frames, patches_per_frame, self.special_per_frame, x.device)
        selection_event = None
        if state.profile_cuda and x.is_cuda and self.layer_idx+1 != state.seed_layer:
            selection_event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            selection_event[0].record()
        with torch.autocast(device_type=x.device.type, enabled=False):
            mean_values = v[0].index_select(1, patches).float().mean(0)
            medoids = state.select(mean_values, self.layer_idx+1)
        if selection_event is not None:
            selection_event[1].record()
            state.selection_events.append(selection_event)
        q, k = original.q_norm(q), original.k_norm(k)
        if original.rope is not None:
            q, k = original.rope(q, pos), original.rope(k, pos)
        attention_event = None
        if state.profile_cuda and x.is_cuda and self.layer_idx+1 != state.seed_layer:
            attention_event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            attention_event[0].record()
        if self.layer_idx+1 == state.seed_layer:
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=original.scale)
        else:
            attended = representative_attention(q, k, v, patches, specials, medoids, scale=original.scale)
        if attention_event is not None:
            attention_event[1].record()
            state.attention_events.append(attention_event)
        state.records[-1]["attention_patch_keys"] = len(patches) if self.layer_idx+1 == state.seed_layer else len(medoids)
        state.records[-1]["special_keys"] = len(specials)
        if state.output_dir is not None:
            path = state.output_dir / f"forward_{state.forward_id:04d}" / f"layer_{self.layer_idx+1:02d}" / "clustering.json"
            path.write_text(json.dumps(state.records[-1], indent=2)+"\n")
        return original.proj_drop(original.proj(attended.transpose(1, 2).reshape(b, n, c)))


def install_layer_value_attention(model, **kwargs):
    """Patch the dense seed and requested sparse layers; reset on every input."""
    if hasattr(model, "layer_value_state"):
        raise ValueError("Layer Value attention is already installed")
    state = LayerValueState(**kwargs)
    for layer in range(state.seed_layer-1, state.sparse_end):
        block = model.aggregator.global_blocks[layer]
        block.attn = LayerValueAttention(block.attn, state=state, layer_idx=layer).eval()

    def reset_on_input(module, inputs):
        images = inputs[0]
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected [1, frames, 3, height, width]")
        state.reset(images.shape[1], images.shape[-2]//14, images.shape[-1]//14)

    state.reset_hook_handle = model.aggregator.register_forward_pre_hook(reset_on_input)
    model.layer_value_state = state
    return state
