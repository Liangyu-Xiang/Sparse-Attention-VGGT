"""Low-cost, layer-index-free A/V routing after current Value updates."""
import json
import math
from pathlib import Path
import torch


class InternalAVGate:
    def __init__(self,parameters):
        self.p=parameters
        if (parameters.get('version')!=1 or parameters.get('feature')!='log_sqrt_head_kl_times_group_js'
            or not math.isfinite(parameters['midpoint']) or not math.isfinite(parameters['temperature'])
            or parameters['temperature']<=0 or not 0<parameters['min_alpha']<=parameters['max_alpha']<1
            or not 0<parameters['alpha_step']<=1 or parameters['groups']<1):
            raise ValueError('Invalid internal A/V gate')

    @classmethod
    def read(cls,path):return cls(json.loads(Path(path).read_text()))

    def alpha(self,head_kl,group_js):
        score=math.log(max(1e-12,math.sqrt(max(0,head_kl)*max(0,group_js))))
        z=max(-50.,min(50.,(score-self.p['midpoint'])/self.p['temperature']))
        a=self.p['min_alpha']+(self.p['max_alpha']-self.p['min_alpha'])/(1+math.exp(-z))
        a=round(a/self.p['alpha_step'])*self.p['alpha_step']
        return max(self.p['min_alpha'],min(self.p['max_alpha'],a)),score

    def decide(self,q,k,state,layer,scale):
        s=state;s.policy_parent_keys=None
        if layer==0:
            s.policy_info=dict(gate_requested_alpha=0.,gate_effective_alpha=0.,gate_seed=True,
                               gate_extra_qk_per_head=0,gate_scalar_transfers=0)
            return
        # Same pooled Parent Key as final routing; cached and consumed once by refresh_hierarchy.
        fine=k[:,s.patches[s.value.reps]]
        pooled=torch.zeros((k.shape[0],s.parent_count,k.shape[-1]),device=k.device,dtype=torch.float32)
        pooled.index_add_(1,s.parent,fine.float())
        pooled=(pooled/s.parent_fine_count.clamp_min(1)[None,:,None]).to(k.dtype)
        s.policy_parent_keys=pooled
        population=(s.value.table>=0).sum(1).float();prior=population/population.sum()
        ng=math.ceil(s.n/32);n=min(ng,self.p['groups'])
        gs=((torch.arange(n,device=q.device)+.5)*ng/n).long()
        probes=(gs[:,None]*32+torch.tensor([4,12,20,28],device=q.device)).clamp_max(s.n-1)
        with torch.autocast(q.device.type,enabled=False):
            z=torch.einsum('hgtd,hpd->hgtp',q[:,probes].float(),pooled.float())*scale+population.log()
            prob=z.softmax(-1);p=prob.mean((0,2));m=p.mean(0)
            normalizer=max(1.,math.log(s.parent_count))
            head_kl=(prob*(prob.clamp_min(1e-30).log()-prior.clamp_min(1e-30).log())).sum(-1).mean()/normalizer
            js=(p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1).mean()/normalizer
        # One explicit scalar transfer per nonseed layer, included in complete timing.
        h,j=torch.stack((head_kl,js)).tolist();alpha,score=self.alpha(h,j)
        incoming=0 if s.history is None else s.history.shape[1]
        attention=max(round(alpha*s.total[layer]),incoming)
        if attention>s.total[layer]:raise ValueError('Incoming history exceeds current total budget')
        s.ka[layer]=attention;s.kv[layer]=s.total[layer]-attention
        s.policy_info=dict(gate_requested_alpha=alpha,gate_effective_alpha=attention/s.total[layer],
            gate_head_parent_kl=h,gate_query_group_js=j,gate_log_score=score,
            gate_incoming_history_cap=incoming,gate_seed=False,
            gate_extra_qk_per_head=4*n*s.parent_count,gate_scalar_transfers=1)

    def outgoing_history(self,state,layer):
        if layer==23:return 0
        # No knowledge of next-layer signals. Use half CURRENT A budget, L0 minimum seed.
        current=max(state.ka[layer],round(self.p['min_alpha']*state.total[layer]))
        next_cap=round(self.p['max_alpha']*state.total[layer+1])
        return min(state.total[layer],math.floor(state.config.history_fraction*min(current,next_cap)))
