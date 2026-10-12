"""Progressive token-level sparse attention with 2x2 spatial de-duplication.

The selector in this module is intentionally token-level: each patch query keeps
an explicit list of patch-key tokens.  A 2x2 grid is only used as a *hard
spatial de-duplication constraint* (one key per grid), not as the unit of the
attention computation.  The selected ragged token lists are consumed by the
existing per-query sparse-attention kernel.

The state is shared by every head in a functional group.  At a global layer,
all heads in one group read the same previous state, use staggered uniform
probes, and then jointly write the next state.  Consequently the update is
independent of head execution order.  State remains live for the entire model
forward; there is deliberately no maximum-number-of-layers reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..triton.sparse_topk_attention import fused_sparse_topk_attention


@dataclass
class _GroupMemory:
    """Per-query token history for one functional group."""

    token_idx: Tensor  # [num_patch_tokens, budget], local patch-token indices
    evidence: Tensor   # [num_patch_tokens, budget], unnormalised logit evidence


class ProgressiveGridState:
    """Keeps progressive key-token state for all functional groups in one scene.

    Evidence is the standardized, *pre-softmax* QK compatibility observed on
    an actually evaluated edge.  This avoids treating a sparse-subset softmax
    weight as a reliable importance target.  Old evidence is exponentially
    decayed when new observations are merged, so the state evolves from the
    initial uniform coverage toward repeatedly supported key grids.
    """

    def __init__(
        self,
        patch_key_budget: int,
        exploitation_ratio: float = 0.75,
        evidence_decay: float = 0.60,
        score_chunk_size: int = 1024,
    ) -> None:
        if patch_key_budget < 1:
            raise ValueError("patch_key_budget must be positive")
        if not 0.0 <= exploitation_ratio <= 1.0:
            raise ValueError("exploitation_ratio must be in [0, 1]")
        if not 0.0 <= evidence_decay <= 1.0:
            raise ValueError("evidence_decay must be in [0, 1]")

        self.patch_key_budget = patch_key_budget
        self.exploitation_ratio = exploitation_ratio
        self.evidence_decay = evidence_decay
        self.score_chunk_size = score_chunk_size
        self._memory: Dict[str, _GroupMemory] = {}
        self._pending: Dict[str, _GroupMemory] = {}
        self._geometry: Tuple[int, int, int] | None = None

    def reset(self, patch_height: int, patch_width: int, num_images: int | None = None) -> None:
        """Start a new scene.  ``num_images`` is checked lazily by ``select``."""
        self._memory.clear()
        self._pending.clear()
        self._geometry = (patch_height, patch_width, -1 if num_images is None else num_images)

    @staticmethod
    def _grid_ids(token_idx: Tensor, patch_height: int, patch_width: int) -> Tensor:
        """Map flattened patch-token ids to per-frame 2x2 grid ids."""
        patches_per_image = patch_height * patch_width
        grids_per_row = (patch_width + 1) // 2
        grids_per_image = ((patch_height + 1) // 2) * grids_per_row

        frame = torch.div(token_idx, patches_per_image, rounding_mode="floor")
        local = token_idx.remainder(patches_per_image)
        row = torch.div(local, patch_width, rounding_mode="floor")
        col = local.remainder(patch_width)
        return frame * grids_per_image + (row // 2) * grids_per_row + col // 2

    @staticmethod
    def _grid_tokens(
        grid_idx: Tensor,
        patch_height: int,
        patch_width: int,
        head_rank: int,
    ) -> Tensor:
        """Choose a staggered token corner for every selected 2x2 grid."""
        patches_per_image = patch_height * patch_width
        grids_per_row = (patch_width + 1) // 2
        grids_per_image = ((patch_height + 1) // 2) * grids_per_row

        frame = torch.div(grid_idx, grids_per_image, rounding_mode="floor")
        local_grid = grid_idx.remainder(grids_per_image)
        grid_row = torch.div(local_grid, grids_per_row, rounding_mode="floor")
        grid_col = local_grid.remainder(grids_per_row)

        # Heads 0--3 choose distinct corners.  More heads repeat the corner
        # assignment but use a shifted grid phase in ``_uniform_tokens``.
        corner = head_rank % 4
        row = torch.clamp(2 * grid_row + corner // 2, max=patch_height - 1)
        col = torch.clamp(2 * grid_col + corner % 2, max=patch_width - 1)
        return frame * patches_per_image + row * patch_width + col

    def _uniform_tokens(
        self,
        num_patch_tokens: int,
        patch_height: int,
        patch_width: int,
        head_rank: int,
        device: torch.device,
    ) -> Tensor:
        """Uniformly probe the entire flattened key-token space for every query."""
        patches_per_image = patch_height * patch_width
        if num_patch_tokens % patches_per_image:
            raise ValueError("patch tokens must contain an integer number of images")
        num_images = num_patch_tokens // patches_per_image
        grids_per_image = ((patch_height + 1) // 2) * ((patch_width + 1) // 2)
        num_grids = num_images * grids_per_image
        budget = min(self.patch_key_budget, num_grids)

        # The base positions are exactly uniform over all frames and all patch
        # grids.  All queries of one head start from this same global lattice,
        # as required by the uniform initialisation; only same-layer heads are
        # staggered.  Query-specific concentration appears only after actual
        # sparse edges have been observed and written to history.
        base = torch.div(
            torch.arange(budget, device=device, dtype=torch.long) * num_grids,
            budget,
            rounding_mode="floor",
        )
        head_phase = (head_rank // 4) * max(1, num_grids // 7)
        grid_idx = ((base + head_phase) % num_grids).unsqueeze(0).expand(num_patch_tokens, -1)
        return self._grid_tokens(grid_idx, patch_height, patch_width, head_rank)

    @staticmethod
    def _deduplicate_and_take(
        token_idx: Tensor,
        priority: Tensor,
        patch_height: int,
        patch_width: int,
        budget: int,
    ) -> Tuple[Tensor, Tensor]:
        """Keep the highest-priority token for each 2x2 grid, then take ``budget``.

        The two stable sorts implement a lexicographic ordering by
        ``(grid_id, -priority)`` without materialising a dense query-by-grid
        tensor.  Thus the hard grid constraint scales with selected edges, not
        with the full key space.
        """
        if token_idx.shape != priority.shape:
            raise ValueError("token_idx and priority must have the same shape")

        by_score = torch.argsort(priority, dim=1, descending=True, stable=True)
        idx_by_score = torch.gather(token_idx, 1, by_score)
        pri_by_score = torch.gather(priority, 1, by_score)
        grids_by_score = ProgressiveGridState._grid_ids(idx_by_score, patch_height, patch_width)

        by_grid = torch.argsort(grids_by_score, dim=1, stable=True)
        idx_sorted = torch.gather(idx_by_score, 1, by_grid)
        pri_sorted = torch.gather(pri_by_score, 1, by_grid)
        grid_sorted = torch.gather(grids_by_score, 1, by_grid)

        keep = torch.ones_like(grid_sorted, dtype=torch.bool)
        keep[:, 1:] = grid_sorted[:, 1:] != grid_sorted[:, :-1]
        dedup_priority = pri_sorted.masked_fill(~keep, -torch.inf)
        best = torch.topk(dedup_priority, k=budget, dim=1, sorted=False).indices
        return torch.gather(idx_sorted, 1, best), torch.gather(pri_sorted, 1, best)

    def select(
        self,
        group_id: str,
        num_patch_tokens: int,
        patch_height: int,
        patch_width: int,
        head_rank: int,
        device: torch.device,
    ) -> Tensor:
        """Select token keys for one head without mutating the shared state."""
        uniform = self._uniform_tokens(
            num_patch_tokens, patch_height, patch_width, head_rank, device
        )
        memory = self._memory.get(group_id)
        if memory is None or memory.token_idx.shape != uniform.shape:
            return uniform

        budget = uniform.shape[1]
        exploit_count = min(budget, int(round(budget * self.exploitation_ratio)))
        if exploit_count == 0:
            return uniform

        top_pos = torch.topk(memory.evidence, k=exploit_count, dim=1, sorted=False).indices
        exploit_idx = torch.gather(memory.token_idx, 1, top_pos)
        exploit_evidence = torch.gather(memory.evidence, 1, top_pos)

        # Uniform probes provide the exploration quota.  Because they are
        # included as full candidates before token selection, an explored grid
        # is still retained whenever it collides with a historical grid.
        candidates = torch.cat((exploit_idx, uniform), dim=1)
        # Exploration must rank below every finite historical observation, but
        # above a duplicate that has been explicitly masked to ``-inf`` by the
        # grid de-duplication step.
        explore_priority = torch.full_like(uniform, -1e9, dtype=exploit_evidence.dtype)
        priority = torch.cat((exploit_evidence, explore_priority), dim=1)
        selected, _ = self._deduplicate_and_take(
            candidates, priority, patch_height, patch_width, budget
        )
        return selected

    @staticmethod
    def compatibility_scores(q_img: Tensor, k_img: Tensor, token_idx: Tensor, chunk_size: int) -> Tensor:
        """Return actual QK logits only for evaluated sparse edges.

        Chunking bounds temporary ``[query, key, dim]`` storage and deliberately
        avoids constructing any dense query-key score matrix.
        """
        if q_img.ndim != 2 or k_img.ndim != 2:
            raise ValueError("q_img and k_img must be [num_patch_tokens, head_dim]")
        output: List[Tensor] = []
        scale = q_img.shape[-1] ** -0.5
        for start in range(0, q_img.shape[0], chunk_size):
            end = min(start + chunk_size, q_img.shape[0])
            selected_k = k_img[token_idx[start:end].long()]
            score = (q_img[start:end].unsqueeze(1).float() * selected_k.float()).sum(dim=-1)
            output.append(score.mul_(scale))
        return torch.cat(output, dim=0)

    @staticmethod
    def _standardize(scores: Tensor) -> Tensor:
        scores = scores.float()
        centered = scores - scores.mean(dim=1, keepdim=True)
        return centered / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)

    def begin_group(self, group_id: str) -> None:
        """Open a same-layer group update while preserving its read snapshot."""
        if group_id in self._pending:
            raise RuntimeError(f"functional group {group_id!r} is already being updated")
        previous = self._memory.get(group_id)
        if previous is not None:
            self._pending[group_id] = _GroupMemory(
                previous.token_idx,
                previous.evidence * self.evidence_decay,
            )

    def accumulate(
        self,
        group_id: str,
        token_idx: Tensor,
        scores: Tensor,
        patch_height: int,
        patch_width: int,
    ) -> None:
        """Merge one observed head into a bounded pending group state.

        Each head still selected from the immutable ``_memory`` snapshot.  The
        bounded ``_pending`` accumulator makes same-layer coordination scale as
        O(num_queries * budget), rather than O(num_queries * budget * heads).
        """
        current_evidence = self._standardize(scores)
        previous = self._pending.get(group_id)
        if previous is not None and previous.token_idx.shape == token_idx.shape:
            candidate_idx = torch.cat((previous.token_idx, token_idx), dim=1)
            candidate_evidence = torch.cat((previous.evidence, current_evidence), dim=1)
        else:
            candidate_idx = token_idx
            candidate_evidence = current_evidence

        next_idx, next_evidence = self._deduplicate_and_take(
            candidate_idx, candidate_evidence, patch_height, patch_width, token_idx.shape[1]
        )
        self._pending[group_id] = _GroupMemory(next_idx.to(torch.int32), next_evidence)

    def end_group(self, group_id: str) -> None:
        """Commit all coordinated same-layer observations for one group."""
        pending = self._pending.pop(group_id, None)
        if pending is not None:
            self._memory[group_id] = pending

    def update(
        self,
        group_id: str,
        observations: Sequence[Tuple[Tensor, Tensor]],
        patch_height: int,
        patch_width: int,
    ) -> None:
        """Convenience API used by tests and single-step callers."""
        self.begin_group(group_id)
        for token_idx, scores in observations:
            self.accumulate(group_id, token_idx, scores, patch_height, patch_width)
        self.end_group(group_id)


def _sparse_patch_attention(
    q_patch: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    key_indices: Tensor,
    chunk_size: int,
) -> Tensor:
    """Sparse per-query attention over an arbitrary global K/V token subset."""
    if q_patch.device.type == "cuda":
        return fused_sparse_topk_attention(q_patch, k_all, v_all, key_indices.to(torch.int32))

    # CPU reference used by unit tests.  It is intentionally chunked for the
    # same memory behaviour as the CUDA path.
    out: List[Tensor] = []
    scale = q_patch.shape[-1] ** -0.5
    for start in range(0, q_patch.shape[0], chunk_size):
        end = min(start + chunk_size, q_patch.shape[0])
        k_selected = k_all[key_indices[start:end].long()]
        v_selected = v_all[key_indices[start:end].long()]
        logits = (q_patch[start:end].float().unsqueeze(1) * k_selected.float()).sum(-1) * scale
        weights = logits.softmax(dim=-1).to(v_selected.dtype)
        out.append((weights.unsqueeze(-1) * v_selected).sum(dim=1))
    return torch.cat(out, dim=0)


def progressive_grid_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    patch_token_indices: Tensor,
    patch_height: int,
    patch_width: int,
    num_special_tokens: int,
    chunk_size: int,
) -> Tensor:
    """Apply sparse attention while always retaining all special key tokens."""
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError("progressive sparse attention currently supports [B=1, H=1]")
    num_tokens = q.shape[2]
    tokens_per_image = patch_height * patch_width + num_special_tokens
    if num_tokens % tokens_per_image:
        raise ValueError("global token count is incompatible with patch geometry")
    num_images = num_tokens // tokens_per_image

    device = q.device
    base = torch.arange(num_images, device=device, dtype=torch.long) * tokens_per_image
    special_local = torch.arange(num_special_tokens, device=device, dtype=torch.long)
    special_idx = (base[:, None] + special_local[None, :]).reshape(-1)
    patch_local = torch.arange(num_special_tokens, tokens_per_image, device=device, dtype=torch.long)
    image_idx = (base[:, None] + patch_local[None, :]).reshape(-1)

    q_flat = q.squeeze(0).squeeze(0)
    k_flat = k.squeeze(0).squeeze(0)
    v_flat = v.squeeze(0).squeeze(0)
    q_patch = q_flat[image_idx]

    # The special tokens are present in every patch query's K/V set.  They are
    # not assigned to spatial grids and therefore never compete with patch keys.
    selected_global = image_idx[patch_token_indices.long()]
    special_per_query = special_idx.unsqueeze(0).expand(selected_global.shape[0], -1)
    selected_global = torch.cat((special_per_query, selected_global), dim=1).contiguous()
    out_patch = _sparse_patch_attention(q_patch, k_flat, v_flat, selected_global, chunk_size)

    # Special-token queries are few (5 per image), so retaining dense access is
    # inexpensive and avoids dropping camera/register communication.
    out_special = F.scaled_dot_product_attention(
        q[:, :, special_idx, :], k, v
    ).squeeze(0).squeeze(0)

    out = torch.empty_like(v_flat)
    out[image_idx] = out_patch
    out[special_idx] = out_special
    return out.unsqueeze(0).unsqueeze(0)


class ProgressiveGridAttentionWrapper(nn.Module):
    """VGGT attention replacement implementing the progressive grid selector."""

    def __init__(
        self,
        orig_module: nn.Module,
        *,
        layer_idx: int,
        layer_groups: Sequence[dict],
        state: ProgressiveGridState,
        num_special_tokens: int,
        **_: object,
    ) -> None:
        super().__init__()
        self.orig_module = orig_module
        self.layer_idx = layer_idx
        self.layer_groups = [
            {"id": str(group["id"]), "head_ids": tuple(group["head_ids"])}
            for group in layer_groups
        ]
        self.state = state
        self.num_special_tokens = num_special_tokens
        self.patch_height: int | None = None
        self.patch_width: int | None = None

        covered = sorted(h for group in self.layer_groups for h in group["head_ids"])
        expected = list(range(orig_module.num_heads))
        if covered != expected:
            raise ValueError(
                f"layer {layer_idx}: functional groups must partition heads {expected}; got {covered}"
            )

    def set_geometry(self, patch_height: int, patch_width: int) -> None:
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if self.patch_height is None or self.patch_width is None:
            raise RuntimeError("call update_args_per_forward before progressive sparse inference")

        batch, num_tokens, channels = x.shape
        qkv = self.orig_module.qkv(x).reshape(
            batch, num_tokens, 3, self.orig_module.num_heads, self.orig_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None:
            q = self.orig_module.rope(q, pos)
            k = self.orig_module.rope(k, pos)

        tokens_per_image = self.patch_height * self.patch_width + self.num_special_tokens
        if num_tokens % tokens_per_image:
            raise ValueError("input token count does not match current patch geometry")
        num_patch_tokens = (num_tokens // tokens_per_image) * self.patch_height * self.patch_width

        outputs: List[Tensor | None] = [None] * self.orig_module.num_heads
        image_indices = _image_token_indices(
            num_tokens, self.patch_height, self.patch_width, self.num_special_tokens, q.device
        )

        for group in self.layer_groups:
            head_ids: Iterable[int] = group["head_ids"]
            self.state.begin_group(group["id"])
            for rank, head_idx in enumerate(head_ids):
                selected = self.state.select(
                    group["id"], num_patch_tokens, self.patch_height, self.patch_width,
                    rank, q.device,
                )
                q_head = q[:, head_idx:head_idx + 1]
                k_head = k[:, head_idx:head_idx + 1]
                v_head = v[:, head_idx:head_idx + 1]
                outputs[head_idx] = progressive_grid_attention(
                    q_head, k_head, v_head, selected,
                    self.patch_height, self.patch_width, self.num_special_tokens,
                    self.state.score_chunk_size,
                )

                scores = self.state.compatibility_scores(
                    q_head.squeeze(0).squeeze(0)[image_indices],
                    k_head.squeeze(0).squeeze(0)[image_indices],
                    selected,
                    self.state.score_chunk_size,
                )
                self.state.accumulate(
                    group["id"], selected, scores, self.patch_height, self.patch_width
                )

            # All heads in this same-layer functional group read a common
            # snapshot and update the group only after all have executed.
            self.state.end_group(group["id"])

        if any(output is None for output in outputs):  # defensive; validated in __init__
            raise RuntimeError("some attention heads did not produce an output")
        attention = torch.cat([output for output in outputs if output is not None], dim=1)
        result = attention.transpose(1, 2).reshape(batch, num_tokens, channels)
        result = self.orig_module.proj(result)
        return self.orig_module.proj_drop(result)


def _image_token_indices(
    num_tokens: int,
    patch_height: int,
    patch_width: int,
    num_special_tokens: int,
    device: torch.device,
) -> Tensor:
    tokens_per_image = patch_height * patch_width + num_special_tokens
    num_images = num_tokens // tokens_per_image
    base = torch.arange(num_images, device=device, dtype=torch.long) * tokens_per_image
    patches = torch.arange(num_special_tokens, tokens_per_image, device=device, dtype=torch.long)
    return (base[:, None] + patches[None, :]).reshape(-1)
