"""Opt-in, separately synchronized profiling and same-input attention oracles.

Never enable this collector in production timing runs. Raw tensors and dense
oracles are exported AFTER closing each layer's measured pipeline interval.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F


class ValueDiagnostics:
    def __init__(self, output_dir, *, raw=False, dense_output=True, dense_oracle=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw, self.dense_output, self.dense_oracle = raw, dense_output, dense_oracle
        if dense_oracle and not raw:
            raise ValueError('Dense oracle is intended for small raw diagnostic inputs')
        self.rows, self.head_rows, self.runtime_rows = [], [], []
        self.previous_hierarchy = None
        self.stage_name = None

    @staticmethod
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin_layer(self, layer, history):
        self.sync()
        self.layer, self.incoming = layer, history
        self.stage_name = None
        self.timings, self.chunks = {}, []
        self.life = dict(incoming=0 if history is None else history.numel(), reused=0,
                         explored=0, validated=0, reused_validated=0, explored_validated=0,
                         promoted_value=0)
        self.layer_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def mark(self, name):
        # Diagnostic synchronized wall-clock, explicitly NOT formal timings.
        self.sync()
        now = time.perf_counter()
        if self.stage_name is not None:
            self.timings[self.stage_name] = self.timings.get(self.stage_name, 0.)+(now-self.stage_start)*1000
        self.stage_start, self.stage_name = now, name

    def context(self, hierarchy, patches, specials, p, budget, ka, uniform):
        self.hierarchy, self.patches, self.specials = hierarchy, patches, specials
        self.p, self.budget, self.ka, self.uniform = p, budget, ka, uniform

    def selected(self, indices, reused_count, quotas):
        self.mark('diagnostic_bookkeeping')
        slots = torch.arange(indices.shape[1], device=indices.device)[None]
        origin = torch.where(slots < reused_count[:, None], 1,
                             torch.where(slots < self.ka, 2, 0)).to(torch.uint8)
        self.current_origin = origin
        self.life['reused'] += reused_count.sum()
        self.life['explored'] += len(indices)*self.ka-reused_count.sum()
        if self.raw:
            self.chunks.append(dict(indices=indices.detach(), origin=origin,
                                    group_budget=quotas.detach()))

    def validated(self, winners):
        self.mark('diagnostic_bookkeeping')
        origin = self.current_origin.gather(1, winners)
        self.life['validated'] += winners.numel()
        self.life['reused_validated'] += (origin == 1).sum()
        self.life['explored_validated'] += (origin == 2).sum()
        self.life['promoted_value'] += (origin == 0).sum()
        if self.raw:
            self.chunks[-1]['validated_slots'] = winners.detach()

    def uniform_selected(self, indices, queries):
        if self.raw:
            nc = len(self.hierarchy[2].unique())
            self.chunks.append(dict(indices=indices[None].expand(queries, -1),
                origin=torch.zeros((queries, len(indices)), device=indices.device, dtype=torch.uint8),
                group_budget=torch.zeros((queries, nc), device=indices.device, dtype=torch.int32),
                validated_slots=torch.empty((queries, 0), device=indices.device, dtype=torch.long)))

    @staticmethod
    def number(x):
        return float(x) if torch.is_tensor(x) else x

    @staticmethod
    def distribution(x):
        x = x.float()
        return dict(min=float(x.min()), mean=float(x.mean()), median=float(x.median()),
                    p90=float(torch.quantile(x, .9)), max=float(x.max()))

    def finish_layer(self, record, q, k, v, sparse_output, history, scale):
        self.mark(None)
        pipeline_ms = (time.perf_counter()-self.layer_start)*1000
        peak = torch.cuda.max_memory_allocated()/2**30 if q.is_cuda else None
        row = {key: self.number(value) for key, value in record.items()}
        row.update(pipeline_wall_ms=pipeline_ms, peak_allocated_gib=peak)
        life = {key: self.number(value) for key, value in self.life.items()}
        life['history_filtered_before_execution'] = life['incoming']-life['reused']
        life['rejected_attention_candidates'] = life['reused']+life['explored']-life['reused_validated']-life['explored_validated']
        life['history_survival_rate'] = life['reused_validated']/life['reused'] if life['reused'] else None
        life['exploration_precision'] = life['explored_validated']/life['explored'] if life['explored'] else None
        life['value_only'] = len(q[0])*(self.budget-self.ka)-life['promoted_value']
        row.update(life)
        z, reps, mapping, count = self.hierarchy
        coarse_size = torch.zeros(len(mapping.unique()), device=q.device, dtype=torch.long)
        coarse_size.index_add_(0, mapping, count)
        row['fine_cluster_sizes'] = self.distribution(count)
        row['coarse_group_sizes'] = self.distribution(coarse_size)
        if self.previous_hierarchy is None:
            row['fine_migration_fraction'] = row['representative_change_fraction'] = None
            row['source_coarse_sibling_count'] = None
        else:
            old_z, old_reps, old_map, _ = self.previous_hierarchy
            row['fine_migration_fraction'] = float((old_z != z).float().mean())
            row['representative_change_fraction'] = float((old_reps != reps).float().mean())
            old_counts = torch.bincount(old_map, minlength=int(old_map.max())+1)
            row['source_coarse_sibling_count'] = self.distribution(old_counts[old_map[old_z]])
        row['tensor_bytes'] = dict(hierarchy=sum(t.numel()*t.element_size() for t in self.hierarchy),
            coarse_attention=0 if self.p is None else self.p.numel()*self.p.element_size(),
            history=history.numel()*history.element_size(),
            incoming_history=0 if self.incoming is None else self.incoming.numel()*self.incoming.element_size())
        for stage, ms in self.timings.items():
            self.runtime_rows.append(dict(layer_zero_based=self.layer, stage=stage, milliseconds=ms,
                                          percent_of_pipeline=100*ms/max(pipeline_ms, 1e-12)))
        raw = None
        if self.raw:
            patch_indices = torch.cat([c['indices'] for c in self.chunks])
            indices = torch.cat((self.patches[patch_indices], self.specials.expand(len(patch_indices), -1)), -1)
            origin = torch.cat([c['origin'] for c in self.chunks])
            provenance = origin.clone()
            slots = torch.cat([c['validated_slots'] for c in self.chunks])
            rr = torch.arange(len(slots), device=q.device)[:, None].expand_as(slots)
            provenance[rr, slots] = torch.where(origin[rr, slots] == 0, 3, origin[rr, slots])
            raw = dict(fine_cluster_id=z.int().cpu(), fine_rep_index=reps.int().cpu(),
                fine_to_coarse=mapping.int().cpu(), fine_cluster_sizes=count.cpu(), coarse_group_sizes=coarse_size.cpu(),
                coarse_attention=None if self.p is None else self.p.half().cpu(),
                coarse_attention_skipped_uniform_bootstrap=self.uniform,
                group_budget=torch.cat([c['group_budget'] for c in self.chunks]).int().cpu(),
                previous_validated_edges=None if self.incoming is None else self.incoming.cpu(),
                validated_edges=history.cpu(), patch_token_global_indices=self.patches.int().cpu(),
                special_global_indices=self.specials.int().cpu(), final_sparse_indices=indices.int().cpu(),
                selection_origin=torch.cat((origin, origin.new_full((len(origin), len(self.specials)), 4)), -1).cpu(),
                provenance=torch.cat((provenance, provenance.new_full((len(origin), len(self.specials)), 4)), -1).cpu(),
                validated_slots=slots.int().cpu(),
                statistics={key: row[key] for key in ('alpha', 'concentration', 'specificity', 'attention_budget', 'value_budget')})
        else:
            indices = None
        if self.dense_output:
            with torch.autocast(device_type=q.device.type, enabled=False):
                # VGGT q/k norm can return FP32 while V remains BF16. Formal
                # autocast SDPA converts Q/K to V's BF16 type; reproduce that
                # here to keep the dense reference on memory-bounded Flash.
                dense = F.scaled_dot_product_attention(q[None].to(v.dtype), k[None].to(v.dtype),
                                                       v[None], dropout_p=0., scale=scale)[0]
                error2 = (sparse_output.float()-dense.float()).square().sum(-1)
                reference2 = dense.float().square().sum(-1).clamp_min(1e-12)
                query_nmse = error2/reference2
                nmse = error2.sum(-1)/reference2.sum(-1)
                for head in range(len(q)):
                    self.head_rows.append(dict(layer_zero_based=self.layer, head=head,
                        output_nmse=float(nmse[head]), relative_l2=float(nmse[head].sqrt()),
                        query_nmse_mean=float(query_nmse[head].mean()),
                        query_nmse_median=float(query_nmse[head].median()),
                        query_nmse_p90=float(torch.quantile(query_nmse[head], .9))))
                row['output_nmse_mean'] = float(nmse.mean())
                row['output_nmse_median'] = float(nmse.median())
                row['output_nmse_p90'] = float(torch.quantile(nmse, .9))
                if raw is not None:
                    raw.update(dense_output=dense.cpu(), sparse_output=sparse_output.cpu(),
                               per_query_head_output_nmse=query_nmse.cpu())
        if self.dense_oracle:
            with torch.autocast(device_type=q.device.type, enabled=False):
                self.oracle(q, k, indices, scale, raw, row)
        self.rows.append(row)
        if raw is not None:
            torch.save(raw, self.output_dir/f'layer_{self.layer:02d}.pt')
        self.previous_hierarchy = None
        self.chunks.clear()

    def oracle(self, q, k, indices, scale, raw, row):
        """Small-input dense oracle, same current QKV, chunked over queries."""
        topk_rows, sparse_logits, masses, recalls = [], [], [], []
        group_recalls = {fraction: [] for fraction in (.1, .2, .3)}
        history_group_recalls = {fraction: [] for fraction in (.1, .2, .3)}
        z, reps, mapping, count = self.hierarchy
        inverse = torch.full((len(k[0]),), -1, device=q.device, dtype=torch.long)
        inverse[self.patches] = torch.arange(len(self.patches), device=q.device)
        for start in range(0, len(q[0]), 128):
            end = min(start+128, len(q[0]))
            logits = q[:, start:end].float()@k.float().transpose(-1, -2)*scale
            dense_topk = logits.topk(indices.shape[1], -1).indices
            selected = indices[start:end][None].expand(len(q), -1, -1)
            mass = logits.softmax(-1).gather(-1, selected).sum(-1)
            ordered = selected.sort(-1).values.contiguous()
            found = torch.searchsorted(ordered, dense_topk.contiguous()).clamp_max(ordered.shape[-1]-1)
            recall = (ordered.gather(-1, found) == dense_topk).float().mean(-1)
            topk_rows.append(dense_topk.int().cpu())
            sparse_logits.append(logits.gather(-1, selected).half().cpu())
            masses.append(mass.cpu()); recalls.append(recall.cpu())
            if self.p is not None:
                patch_top = inverse[dense_topk]
                valid_patch = patch_top >= 0
                top_groups = mapping[z[patch_top.clamp_min(0)]]
                p = self.p[start:end]
                for fraction in group_recalls:
                    ng = max(1, math.ceil(p.shape[-1]*fraction))
                    high = p.topk(ng, -1).indices
                    # Shared coarse groups, per-head dense oracle.
                    high_mask = torch.zeros_like(p, dtype=torch.bool).scatter_(1, high, True)
                    hit = high_mask[None].expand(len(q), -1, -1).gather(-1, top_groups) & valid_patch
                    group_recalls[fraction].append((hit.sum(-1)/valid_patch.sum(-1).clamp_min(1)).cpu())
                    if self.incoming is not None and self.incoming.shape[1]:
                        old = self.patches[self.incoming[start:end].long()]
                        old = old.sort(-1).values.contiguous()[None].expand(len(q), -1, -1).contiguous()
                        pos = torch.searchsorted(old, dense_topk.contiguous()).clamp_max(old.shape[-1]-1)
                        history_hit = old.gather(-1, pos) == dense_topk
                        history_group_recalls[fraction].append(((hit & history_hit).sum(-1)/hit.sum(-1).clamp_min(1)).cpu())
        mass, recall = torch.cat(masses, 1), torch.cat(recalls, 1)
        raw.update(dense_topk_indices=torch.cat(topk_rows, 1), exact_sparse_logits=torch.cat(sparse_logits, 1),
                   exact_sparse_logits_source='same-input diagnostic dense oracle; not a second timed QK',
                   retained_dense_mass=mass, dense_topk_recall=recall)
        row.update(retained_dense_mass=float(mass.mean()), dense_topk_recall=float(recall.mean()))
        for fraction in group_recalls:
            label = f'{round(fraction*100)}pct'
            row['coarse_group_recall_top'+label] = float(torch.cat(group_recalls[fraction], 1).mean()) if group_recalls[fraction] else None
            row['history_fine_recall_within_top'+label] = float(torch.cat(history_group_recalls[fraction], 1).mean()) if history_group_recalls[fraction] else None
        for head in range(len(q)):
            hr = next(r for r in self.head_rows if r['layer_zero_based'] == self.layer and r['head'] == head)
            hr.update(retained_dense_mass=float(mass[head].mean()), dense_topk_recall=float(recall[head].mean()))

    def save(self, metadata, metrics=None):
        def write_csv(name, rows):
            if not rows:
                return
            keys = list(dict.fromkeys(k for row in rows for k in row))
            with (self.output_dir/name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader(); writer.writerows(rows)
        write_csv('runtime_breakdown.csv', self.runtime_rows)
        write_csv('accuracy_per_head.csv', self.head_rows)
        write_csv('mask_composition.csv', [{k: r.get(k) for k in ('layer_zero_based', 'attention_budget', 'value_budget', 'incoming', 'reused', 'explored', 'validated', 'promoted_value', 'value_only')} for r in self.rows])
        write_csv('history_quality.csv', [{k: r.get(k) for k in ('layer_zero_based', 'history_survival_rate', 'exploration_precision', 'dense_topk_recall', 'retained_dense_mass', 'rejected_attention_candidates')} for r in self.rows])
        write_csv('accuracy_per_layer.csv', [{k: r.get(k) for k in ('layer_zero_based', 'output_nmse_mean', 'output_nmse_median', 'output_nmse_p90', 'retained_dense_mass', 'dense_topk_recall')} for r in self.rows])
        (self.output_dir/'summary.json').write_text(json.dumps(dict(metadata=metadata, final_pose=metrics,
            layers=self.rows, per_head=self.head_rows,
            notes=['Separate synchronized diagnostic run, NOT formal performance samples.',
                   'QK/softmax/AV are genuinely fused and are timed together, not artificially split.',
                   'Raw coarse attention is layer-shared; per-head arrays are error/oracle analysis only.',
                   'NMSE is squared error energy / dense output energy; relative_l2 is its square root.',
                   'Dense references use the SAME current-layer QKV, isolating local operator error.',
                   'Dense output uses autocast-equivalent Q/K cast to V dtype; small dense-logit oracles use FP32 QK.',
                   'Origin: 0=V,1=A-reuse,2=A-explore. Provenance upgrades validated V to 3=AV.',
                   'Code 4 marks mandatory special keys, outside the patch A/V split.',
                   'Special keys are mandatory; provenance/history pertain to patch edges.',
                   'Rates with zero denominator are null (N/A). Uniform layer 0 has no coarse attention.']), indent=2)+'\n')
