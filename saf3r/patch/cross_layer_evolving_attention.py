"""Compact, cross-layer evolving sparse attention for VGGT.

Final attention is always over explicit, query-specific Q--K edges.  Between
layers we keep neither an explicit ``[query, key]`` mask nor attention
weights.  A compact recipe holds Top-M coarse relations, residual mass, and a
deterministic sampling program that regenerates the same exact mask on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CompactMaskRecipe:
    """CPU-resident program for one layer's exact sparse connectivity.

    The relation tensors are not a persisted block attention map.  They are
    the compact parameters of the canonical mask generator.  Given this node,
    its parent, a head, and query ids, :func:`generate_support` regenerates the
    same patch-key mask without storing ``[Q, K]`` indices.
    """

    layer: int
    mode: str
    patch_budget: int
    blocks: int
    top_block_ids: Tensor | None  # CPU uint16 [H, Q, M], None for Uniform
    top_block_probs: Tensor | None  # CPU bf16 [H, Q, M]
    residual_mass: Tensor | None  # CPU bf16 [H, Q]
    observation_strength: Tensor  # CPU fp32 [H]
    history_strength: Tensor  # CPU fp32 [H]
    source_head: Tensor  # CPU int64 [H], -1 means no inheritance
    reuse_count: Tensor  # CPU int64 [H]
    parent: "CompactMaskRecipe | None" = None
    uniform: bool = False

    @property
    def compact_bytes(self) -> int:
        return sum(
            0 if tensor is None else tensor.numel() * tensor.element_size()
            for tensor in (
                self.top_block_ids,
                self.top_block_probs,
                self.residual_mass,
                self.observation_strength,
                self.history_strength,
                self.source_head,
                self.reuse_count,
            )
        )


class CrossLayerEvolvingState:
    """Scene-local state shared by all 24 global-attention replacements."""

    def __init__(
        self,
        *,
        mode: str,
        block_size: int = 128,
        top_m: int = 8,
        match_query_fraction: float = 0.01,
        query_chunk_size: int = 128,
        rank_penalty: float = 0.01,
        dense_reference: bool = False,
    ) -> None:
        if mode not in {"uniform", "current", "cross"}:
            raise ValueError("mode must be 'uniform', 'current', or 'cross'")
        if block_size < 1 or top_m < 1 or query_chunk_size < 1:
            raise ValueError("block_size, top_m, and query_chunk_size must be positive")
        if not 0.0 < match_query_fraction <= 1.0:
            raise ValueError("match_query_fraction must lie in (0, 1]")
        if rank_penalty < 0.0:
            raise ValueError("rank_penalty must be non-negative")
        self.mode = mode
        self.block_size = block_size
        self.top_m = top_m
        self.match_query_fraction = match_query_fraction
        self.query_chunk_size = query_chunk_size
        self.rank_penalty = rank_penalty
        self.dense_reference = dense_reference
        self.history: CompactMaskRecipe | None = None
        self.patch_height: int | None = None
        self.patch_width: int | None = None
        self.frames: int | None = None
        self.diagnostics: list[dict[str, Any]] = []

    def reset(self, *, patch_height: int, patch_width: int) -> None:
        self.history = None
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frames = None
        self.diagnostics.clear()

    def configure_frames(self, frames: int) -> None:
        if self.frames is None:
            self.frames = frames
        elif self.frames != frames:
            raise ValueError("one sparse state cannot mix sequence lengths")


@dataclass(frozen=True)
class BlockLayout:
    starts: Tensor  # GPU int64 [B]
    sizes: Tensor  # GPU int64 [B]
    patch_to_block: Tensor  # GPU int64 [Npatch]


def patch_and_special_indices(
    frames: int,
    patch_height: int,
    patch_width: int,
    special_per_frame: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    patches_per_frame = patch_height * patch_width
    tokens_per_frame = special_per_frame + patches_per_frame
    base = torch.arange(frames, device=device, dtype=torch.long) * tokens_per_frame
    special = (base[:, None] + torch.arange(special_per_frame, device=device)).reshape(-1)
    patch = (base[:, None] + special_per_frame + torch.arange(patches_per_frame, device=device)).reshape(-1)
    return special, patch


def block_layout(
    frames: int,
    patch_height: int,
    patch_width: int,
    block_size: int,
    device: torch.device,
) -> BlockLayout:
    """Consecutive key blocks are independent for every frame."""
    patches_per_frame = patch_height * patch_width
    starts: list[int] = []
    sizes: list[int] = []
    patch_to_block: list[Tensor] = []
    block_id = 0
    for frame in range(frames):
        frame_start = frame * patches_per_frame
        for local_start in range(0, patches_per_frame, block_size):
            size = min(block_size, patches_per_frame - local_start)
            starts.append(frame_start + local_start)
            sizes.append(size)
            patch_to_block.append(torch.full((size,), block_id, device=device, dtype=torch.long))
            block_id += 1
    return BlockLayout(
        starts=torch.tensor(starts, device=device, dtype=torch.long),
        sizes=torch.tensor(sizes, device=device, dtype=torch.long),
        patch_to_block=torch.cat(patch_to_block),
    )


def block_key_means(k_patch: Tensor, layout: BlockLayout) -> Tensor:
    return torch.stack(
        [k_patch[:, start : start + size].float().mean(dim=1) for start, size in zip(layout.starts.tolist(), layout.sizes.tolist())],
        dim=1,
    )


def coarse_distribution(q_chunk: Tensor, key_means: Tensor, sizes: Tensor, scale: float) -> Tensor:
    logits = torch.einsum("hqd,hbd->hqb", q_chunk.float(), key_means)
    return torch.softmax(logits.mul(scale).add(sizes.float().log().view(1, 1, -1)), dim=-1)


def mutual_information_strength(marginal_sum: Tensor, conditional_entropy_sum: Tensor, queries: int) -> Tensor:
    eps = 1.0e-12
    marginal = (marginal_sum.float() / max(queries, 1)).clamp_min(eps)
    marginal = marginal / marginal.sum(dim=-1, keepdim=True)
    h_block = -(marginal * marginal.log()).sum(dim=-1)
    h_conditional = conditional_entropy_sum.float() / max(queries, 1)
    return torch.where(h_block > eps, (h_block - h_conditional) / h_block, torch.zeros_like(h_block)).clamp(0.0, 1.0)


def stratified_query_ids(frames: int, patches_per_frame: int, fraction: float, device: torch.device) -> Tensor:
    """Uniform frame coverage and uniform patch-lattice coverage within frames."""
    total = frames * patches_per_frame
    count = min(total, max(frames, math.ceil(total * fraction)))
    ids: list[Tensor] = []
    base, extra = divmod(count, frames)
    for frame in range(frames):
        take = base + int(frame < extra)
        if take <= 0:
            continue
        local = (torch.arange(take, device=device, dtype=torch.long) * patches_per_frame // take + frame * 97) % patches_per_frame
        ids.append(local + frame * patches_per_frame)
    return torch.cat(ids).sort().values


def _uniform_block_distribution(rows: int, layout: BlockLayout, device: torch.device) -> Tensor:
    return (layout.sizes.float() / layout.sizes.sum()).view(1, -1).expand(rows, -1).to(device)


def recipe_observation(recipe: CompactMaskRecipe, head: int, query_ids: Tensor, layout: BlockLayout, device: torch.device) -> Tensor:
    """Recover Top-M+residual coarse state only for the active query chunk."""
    rows = query_ids.numel()
    if recipe.uniform:
        return _uniform_block_distribution(rows, layout, device)
    if recipe.top_block_ids is None or recipe.top_block_probs is None or recipe.residual_mass is None:
        raise RuntimeError("non-uniform recipe is missing relation tensors")
    if recipe.blocks != layout.sizes.numel():
        raise ValueError("recipe block layout does not match sequence geometry")
    ids_cpu = query_ids.detach().to("cpu", dtype=torch.long)
    ids = recipe.top_block_ids[head].index_select(0, ids_cpu).to(device=device, dtype=torch.long)
    values = recipe.top_block_probs[head].index_select(0, ids_cpu).to(device=device, dtype=torch.float32)
    residual = recipe.residual_mass[head].index_select(0, ids_cpu).to(device=device, dtype=torch.float32)
    block_sizes = layout.sizes.float()
    selected_sizes = block_sizes[ids].sum(dim=-1)
    remaining_size = (block_sizes.sum() - selected_sizes).clamp_min(1.0)
    result = residual[:, None] * block_sizes.view(1, -1) / remaining_size[:, None]
    result.scatter_(1, ids, values)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def recipe_sampling_distribution(recipe: CompactMaskRecipe, head: int, query_ids: Tensor, layout: BlockLayout, device: torch.device) -> Tensor:
    observation = recipe_observation(recipe, head, query_ids, layout, device)
    strength = float(recipe.observation_strength[head].item())
    return strength * observation + (1.0 - strength) * _uniform_block_distribution(query_ids.numel(), layout, device)


def _current_edge_candidates(
    *,
    guidance: Tensor,
    inherited: Tensor,
    layout: BlockLayout,
    layer: int,
    head: int,
    query_ids: Tensor,
    rank_penalty: float,
    count: int,
) -> Tensor:
    """Rank procedural exact edges globally; blocks are never final masks."""
    rows, blocks = guidance.shape
    if count == 0:
        return torch.empty((rows, 0), device=guidance.device, dtype=torch.int32)
    max_size = int(layout.sizes.max().item())
    rank = torch.arange(max_size, device=guidance.device, dtype=torch.long).view(1, 1, -1)
    block = torch.arange(blocks, device=guidance.device, dtype=torch.long).view(1, -1, 1)
    sizes = layout.sizes.view(1, -1, 1)
    valid = rank < sizes
    # 37 is coprime to both VGGT's 128-key blocks and its 89-key remainder.
    phase = torch.remainder(
        query_ids.view(-1, 1, 1) * 19349663 + layer * 83492791 + head * 29791 + block * 911,
        sizes,
    )
    local = torch.remainder(37 * rank + phase, sizes)
    candidates = layout.starts.view(1, -1, 1) + local
    scores = guidance.clamp_min(1.0e-12).log().unsqueeze(-1) - rank.float() / sizes.float() * rank_penalty
    scores = scores.masked_fill(~valid, float("-inf"))
    if inherited.numel():
        # ``inherited_local`` is a physical token position, whereas the last
        # axis of ``scores`` is a rank in the procedural permutation.  Mark in
        # local coordinates first, then gather through the permutation.
        used_local = torch.zeros((rows, blocks, max_size), device=guidance.device, dtype=torch.bool)
        inherited_block = layout.patch_to_block[inherited.long()]
        inherited_local = inherited.long() - layout.starts[inherited_block]
        row = torch.arange(rows, device=guidance.device, dtype=torch.long).view(-1, 1).expand_as(inherited_block)
        used_local[row, inherited_block, inherited_local] = True
        used = torch.gather(used_local, 2, local)
        scores = scores.masked_fill(used, float("-inf"))
    flat_scores = scores.reshape(rows, -1)
    if count > flat_scores.shape[1] - inherited.shape[1]:
        raise RuntimeError("not enough unoccupied patch keys for new support")
    # Stable negligible tie break for residual-uniform replay.
    flat_scores = flat_scores - torch.arange(flat_scores.shape[1], device=guidance.device).view(1, -1) * 1.0e-8
    position = torch.topk(flat_scores, k=count, dim=-1, largest=True, sorted=False).indices
    return torch.gather(candidates.expand(rows, -1, -1).reshape(rows, -1), 1, position).to(torch.int32)


def generate_support(
    recipe: CompactMaskRecipe,
    *,
    head: int,
    query_ids: Tensor,
    layout: BlockLayout,
    rank_penalty: float,
) -> Tensor:
    """Regenerate one query chunk's exact support from compact history."""
    inherited = torch.empty((query_ids.numel(), 0), device=query_ids.device, dtype=torch.int32)
    reuse_count = int(recipe.reuse_count[head].item())
    if recipe.parent is not None and reuse_count:
        source = int(recipe.source_head[head].item())
        if source < 0:
            raise RuntimeError("reuse count requires a valid source head")
        previous = generate_support(recipe.parent, head=source, query_ids=query_ids, layout=layout, rank_penalty=rank_penalty)
        observation = recipe_observation(recipe, head, query_ids, layout, query_ids.device)
        score = torch.gather(observation, 1, layout.patch_to_block[previous.long()]) - previous.float() * 1.0e-8
        inherited = torch.gather(previous, 1, torch.topk(score, k=reuse_count, dim=-1, largest=True, sorted=False).indices).to(torch.int32)
    guidance = recipe_sampling_distribution(recipe, head, query_ids, layout, query_ids.device)
    new = _current_edge_candidates(
        guidance=guidance,
        inherited=inherited,
        layout=layout,
        layer=recipe.layer,
        head=head,
        query_ids=query_ids,
        rank_penalty=rank_penalty,
        count=recipe.patch_budget - inherited.shape[1],
    )
    support = torch.cat((inherited, new), dim=-1)
    if support.shape[1] != recipe.patch_budget:
        raise RuntimeError("generated support violates configured patch-key budget")
    return support


def block_distribution_from_mask(
    recipe: CompactMaskRecipe,
    *,
    head: int,
    query_ids: Tensor,
    layout: BlockLayout,
    rank_penalty: float,
) -> Tensor:
    """Temporary M -> R projection used only for head matching."""
    support = generate_support(recipe, head=head, query_ids=query_ids, layout=layout, rank_penalty=rank_penalty)
    result = torch.zeros((query_ids.numel(), layout.sizes.numel()), device=query_ids.device, dtype=torch.float32)
    result.scatter_add_(1, layout.patch_to_block[support.long()], torch.ones_like(support, dtype=torch.float32))
    return result / recipe.patch_budget


def cross_layer_match(
    history: CompactMaskRecipe,
    current_sample: Tensor,
    current_strength: Tensor,
    query_ids: Tensor,
    layout: BlockLayout,
    rank_penalty: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """All-to-all head matching using online binary-mask-to-block JSD."""
    heads = current_sample.shape[0]
    previous_block = torch.empty_like(current_sample, dtype=torch.float32)
    for source in range(heads):
        previous_block[source] = block_distribution_from_mask(
            history, head=source, query_ids=query_ids, layout=layout, rank_penalty=rank_penalty
        )
    eps = 1.0e-12
    agreement = torch.empty((heads, heads), device=current_sample.device, dtype=torch.float32)
    for source in range(heads):
        p = previous_block[source].clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True)
        for target in range(heads):
            q = current_sample[target].float().clamp_min(eps)
            q = q / q.sum(dim=-1, keepdim=True)
            midpoint = 0.5 * (p + q)
            jsd = 0.5 * ((p * (p.log() - midpoint.log())).sum(dim=-1) + (q * (q.log() - midpoint.log())).sum(dim=-1))
            agreement[source, target] = (1.0 - jsd.mean() / math.log(2.0)).clamp(0.0, 1.0)
    prior_strength = history.history_strength.to(current_sample.device, dtype=torch.float32)
    reuse_score = prior_strength[:, None] * current_strength[None, :] * agreement
    ratio, source = reuse_score.max(dim=0)
    return source, agreement.gather(0, source.view(1, -1)).squeeze(0), ratio


def sparse_patch_attention(
    q_patch: Tensor,
    k_patch: Tensor,
    v_patch: Tensor,
    k_special: Tensor,
    v_special: Tensor,
    support: Tensor,
    scale: float,
) -> Tensor:
    """Reference sparse softmax over selected patch edges and every special key."""
    selected_k = k_patch[support.long()].float()
    selected_v = v_patch[support.long()].float()
    q_float = q_patch.float()
    patch_logits = (q_float.unsqueeze(1) * selected_k).sum(dim=-1) * scale
    special_logits = q_float @ k_special.float().transpose(0, 1) * scale
    probabilities = torch.softmax(torch.cat((special_logits, patch_logits), dim=-1), dim=-1)
    special_out = probabilities[:, : k_special.shape[0]] @ v_special.float()
    patch_out = (probabilities[:, k_special.shape[0] :].unsqueeze(-1) * selected_v).sum(dim=1)
    return special_out + patch_out


class CrossLayerEvolvingAttentionWrapper(nn.Module):
    """VGGT attention replacement with procedural, compact cross-layer masks."""

    def __init__(
        self,
        orig_module: nn.Module,
        *,
        layer_idx: int,
        state: CrossLayerEvolvingState,
        num_special_tokens: int = 5,
        retention: float = 0.25,
        **_: object,
    ) -> None:
        super().__init__()
        if not 0.0 < retention < 1.0:
            raise ValueError("retention must lie in (0, 1)")
        self.orig_module = orig_module
        self.layer_idx = layer_idx
        self.state = state
        self.num_special_tokens = num_special_tokens
        self.retention = retention
        self.patch_height: int | None = None
        self.patch_width: int | None = None

    def set_geometry(self, patch_height: int, patch_width: int) -> None:
        self.patch_height = patch_height
        self.patch_width = patch_width

    def _make_recipe(self, q_patch: Tensor, key_means: Tensor, layout: BlockLayout, patch_budget: int) -> tuple[CompactMaskRecipe, Tensor]:
        """First coarse pass: save only Top-M values and residual mass on CPU."""
        heads, queries, _ = q_patch.shape
        blocks = layout.sizes.numel()
        device = q_patch.device
        if self.state.mode == "uniform":
            zeros = torch.zeros((heads,), dtype=torch.float32)
            return CompactMaskRecipe(
                layer=self.layer_idx, mode="uniform", patch_budget=patch_budget, blocks=blocks,
                top_block_ids=None, top_block_probs=None, residual_mass=None,
                observation_strength=zeros, history_strength=zeros.clone(),
                source_head=torch.full((heads,), -1, dtype=torch.long),
                reuse_count=torch.zeros((heads,), dtype=torch.long), uniform=True,
            ), torch.zeros((heads,), device=device, dtype=torch.float32)
        top_m = min(self.state.top_m, blocks)
        if blocks > torch.iinfo(torch.uint16).max:
            raise ValueError("compact uint16 block ids require at most 65,535 blocks")
        top_ids = torch.empty((heads, queries, top_m), device="cpu", dtype=torch.uint16)
        top_probs = torch.empty((heads, queries, top_m), device="cpu", dtype=torch.bfloat16)
        residual = torch.empty((heads, queries), device="cpu", dtype=torch.bfloat16)
        marginal_sum = torch.zeros((heads, blocks), device=device, dtype=torch.float64)
        conditional_sum = torch.zeros((heads,), device=device, dtype=torch.float64)
        for start in range(0, queries, self.state.query_chunk_size):
            end = min(queries, start + self.state.query_chunk_size)
            observation = coarse_distribution(q_patch[:, start:end], key_means, layout.sizes, self.orig_module.scale)
            values, indices = torch.topk(observation, k=top_m, dim=-1, largest=True, sorted=True)
            top_ids[:, start:end].copy_(indices.to(torch.uint16).cpu())
            top_probs[:, start:end].copy_(values.to(torch.bfloat16).cpu())
            residual[:, start:end].copy_((1.0 - values.sum(dim=-1)).clamp_min(0.0).to(torch.bfloat16).cpu())
            marginal_sum += observation.sum(dim=1, dtype=torch.float64)
            safe = observation.clamp_min(1.0e-12)
            conditional_sum += -(safe * safe.log()).sum(dim=-1).sum(dim=-1, dtype=torch.float64)
        strength = mutual_information_strength(marginal_sum, conditional_sum, queries)
        recipe = CompactMaskRecipe(
            layer=self.layer_idx, mode=self.state.mode, patch_budget=patch_budget, blocks=blocks,
            top_block_ids=top_ids, top_block_probs=top_probs, residual_mass=residual,
            observation_strength=strength.detach().cpu().float(),
            history_strength=torch.zeros((heads,), dtype=torch.float32),
            source_head=torch.full((heads,), -1, dtype=torch.long),
            reuse_count=torch.zeros((heads,), dtype=torch.long), uniform=False,
        )
        return recipe, strength

    def _dense_reference_nmse(self, q_patch: Tensor, k: Tensor, v: Tensor, sparse_output: Tensor) -> Tensor:
        numerator = torch.zeros((q_patch.shape[0],), device=q_patch.device, dtype=torch.float64)
        denominator = torch.zeros_like(numerator)
        for start in range(0, q_patch.shape[1], self.state.query_chunk_size):
            end = min(q_patch.shape[1], start + self.state.query_chunk_size)
            dense = F.scaled_dot_product_attention(q_patch[:, start:end].unsqueeze(0), k, v).squeeze(0).float()
            diff = sparse_output[:, start:end] - dense
            numerator += diff.square().sum(dim=(1, 2), dtype=torch.float64)
            denominator += dense.square().sum(dim=(1, 2), dtype=torch.float64)
        return (numerator / denominator.clamp_min(1.0e-12)).float()

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if self.patch_height is None or self.patch_width is None:
            raise RuntimeError("call update_args_per_forward before cross-layer sparse inference")
        if x.shape[0] != 1:
            raise ValueError("cross-layer sparse attention currently supports batch size one")
        batch, tokens, channels = x.shape
        tokens_per_frame = self.num_special_tokens + self.patch_height * self.patch_width
        if tokens % tokens_per_frame:
            raise ValueError("input token count does not match patch geometry")
        frames = tokens // tokens_per_frame
        self.state.configure_frames(frames)
        qkv = self.orig_module.qkv(x).reshape(batch, tokens, 3, self.orig_module.num_heads, self.orig_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None:
            q = self.orig_module.rope(q, pos)
            k = self.orig_module.rope(k, pos)
        special, patch = patch_and_special_indices(frames, self.patch_height, self.patch_width, self.num_special_tokens, x.device)
        layout = block_layout(frames, self.patch_height, self.patch_width, self.state.block_size, x.device)
        q_patch, k_patch, v_patch = q[0, :, patch], k[0, :, patch], v[0, :, patch]
        k_special, v_special = k[0, :, special], v[0, :, special]
        patch_budget = int(round(self.retention * patch.numel()))
        if patch_budget < 1:
            raise RuntimeError("retention produced an empty patch-key budget")
        key_means = block_key_means(k_patch, layout)
        recipe, observed_strength = self._make_recipe(q_patch, key_means, layout, patch_budget)
        heads, queries, dimension = q_patch.shape
        source_head = torch.full((heads,), -1, device=x.device, dtype=torch.long)
        agreement = torch.zeros((heads,), device=x.device, dtype=torch.float32)
        reuse_ratio = torch.zeros((heads,), device=x.device, dtype=torch.float32)
        if self.state.mode == "cross" and self.state.history is not None:
            sample_ids = stratified_query_ids(frames, self.patch_height * self.patch_width, self.state.match_query_fraction, x.device)
            sample_observation = coarse_distribution(q_patch[:, sample_ids], key_means, layout.sizes, self.orig_module.scale)
            source_head, agreement, reuse_ratio = cross_layer_match(
                self.state.history, sample_observation, observed_strength, sample_ids, layout, self.state.rank_penalty
            )
            recipe.parent = self.state.history
            recipe.source_head.copy_(source_head.detach().cpu())
            recipe.reuse_count.copy_(torch.round(reuse_ratio * patch_budget).long().detach().cpu())
        sparse_output = torch.empty((heads, queries, dimension), device=x.device, dtype=torch.float32)
        marginal_sum = torch.zeros((heads, layout.sizes.numel()), device=x.device, dtype=torch.float64)
        conditional_sum = torch.zeros((heads,), device=x.device, dtype=torch.float64)
        for head in range(heads):
            for start in range(0, queries, self.state.query_chunk_size):
                end = min(queries, start + self.state.query_chunk_size)
                query_ids = torch.arange(start, end, device=x.device, dtype=torch.long)
                support = generate_support(recipe, head=head, query_ids=query_ids, layout=layout, rank_penalty=self.state.rank_penalty)
                ordered = torch.sort(support.long(), dim=-1).values
                if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                    raise RuntimeError("procedural generator emitted duplicate QK edges")
                sparse_output[head, start:end] = sparse_patch_attention(
                    q_patch[head, start:end], k_patch[head], v_patch[head], k_special[head], v_special[head], support, self.orig_module.scale
                )
                counts = torch.zeros((end - start, layout.sizes.numel()), device=x.device, dtype=torch.float32)
                counts.scatter_add_(1, layout.patch_to_block[support.long()], torch.ones_like(support, dtype=torch.float32))
                relation = counts / patch_budget
                marginal_sum[head] += relation.sum(dim=0, dtype=torch.float64)
                safe = relation.clamp_min(1.0e-12)
                conditional_sum[head] += -(safe * safe.log()).sum(dtype=torch.float64)
        history_strength = mutual_information_strength(marginal_sum, conditional_sum, queries)
        recipe.history_strength.copy_(history_strength.detach().cpu().float())
        nmse = self._dense_reference_nmse(q_patch, k, v, sparse_output) if self.state.dense_reference else torch.full((heads,), float("nan"), device=x.device)
        actual_reuse = recipe.reuse_count.to(x.device, dtype=torch.float32) / patch_budget
        for head in range(heads):
            self.state.diagnostics.append({
                "layer": self.layer_idx, "head": head,
                "g_observation": float(observed_strength[head].item()),
                "g_history": float(history_strength[head].item()),
                "source_head": int(source_head[head].item()),
                "cross_layer_agreement": float(agreement[head].item()),
                "reuse_ratio": float(reuse_ratio[head].item()),
                "actual_reuse_ratio": float(actual_reuse[head].item()),
                "patch_budget": patch_budget, "patch_retention": patch_budget / patch.numel(),
                "compact_history_bytes": recipe.compact_bytes,
                "nmse_to_dense_same_input": float(nmse[head].item()),
            })
        self.state.history = recipe
        special_output = F.scaled_dot_product_attention(q[:, :, special], k, v)
        attention = torch.empty_like(v)
        attention[0, :, patch] = sparse_output.to(v.dtype)
        attention[:, :, special] = special_output
        result = attention.transpose(1, 2).reshape(batch, tokens, channels)
        return self.orig_module.proj_drop(self.orig_module.proj(result))
