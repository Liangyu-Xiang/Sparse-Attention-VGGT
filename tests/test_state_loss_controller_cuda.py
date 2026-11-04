"""Numerical checks of loss/packing/Graph execution, not quality evaluations."""
import unittest
import torch
import triton
from saf3r.patch.state_loss_reference import uniform_unobserved,group_history_losses,transport_losses,reduce_cluster
from saf3r.triton import state_loss_controller as kernel
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch import reuse_inference_ops as rule
from saf3r.patch.reuse_window_graph import Window


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ControllerTests(unittest.TestCase):
    def test_history_support_is_union_with_rep_and_costs_unique_keys(self):
        # 45 patches + five always-kept Specials: cap=5 gives 20% total Keys.
        device='cuda';g=3;n=45;d=4
        labels=torch.tensor([0]*3+[1]*42,device=device)
        reps=torch.tensor([0,3],device=device);pop=torch.tensor([3.,42.],device=device)
        prior=torch.full((g,n),1/50,device=device);pi=reduce_cluster(prior,labels,2)
        old=torch.tensor([[0,2,4,-1],[1,2,3,5],[0,3,-1,-1]],device=device,dtype=torch.int32)
        y=torch.arange(n*d,device=device,dtype=torch.float32).reshape(n,d)/100
        members=torch.full((2,64),-1,device=device,dtype=torch.int32)
        members[0,:3]=torch.arange(3,device=device);members[1,:42]=torch.arange(3,n,device=device)
        geometry=dict(width=64,members=members,drop_bins=prepare(pop),loss_patch_values=y,
            loss_special_values=torch.zeros(5,d,device=device),loss_history_stride=8,
            loss_operator='residual_rep',state_loss_variant='centered_residual_energy')
        context=kernel.gate(prior,pi,old,labels,reps,pop,True)
        stats=kernel.statistics(prior,pi,labels,reps,pop,geometry,context)
        errors,costs,_=kernel.state_losses(prior,pi,torch.ones_like(pi,dtype=torch.bool),pop,reps,
            geometry,stats,torch.full((g,5),1/50,device=device))
        qualified=context['qualified'];union=qualified.clone();union.scatter_(1,reps[None].expand(g,-1),True)
        packed_context=dict(context,qualified=union,eligible_count=context['eligible_count']+(~context['observed_rep']).int())
        selected=torch.full((g,2),5,device=device,dtype=torch.int64)
        ids,_,record=rule.pack(selected,None,reps,labels,pop,5,packed_context)
        weights,_=kernel.execution_weights(selected,ids,prior,pi,labels,reps,pop,geometry,packed_context,stats)
        torch.testing.assert_close(ids,torch.tensor([[0,2,3,4,-1],[0,1,2,3,5],[0,3,-1,-1,-1]],device=device,dtype=torch.int32))
        expected=torch.tensor([[2.,1.,41.,1.,0.],[1.,1.,1.,41.,1.],[3.,42.,0.,0.,0.]],device=device)
        torch.testing.assert_close(weights,expected,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(costs[:,:,2].sum(1),record['counts'].float())
        self.assertTrue(bool(((record['counts']+5)/50<=.2).all()))
        self.assertTrue(bool((record['native_cost_count_error']==0).all()))
        torch.testing.assert_close(errors[2,:,2],errors[2,:,1])
        torch.testing.assert_close(costs[2,:,2],costs[2,:,1])
        # Evaluate the displayed formula independently for the exact emitted support.
        for row in range(g):
            for c in range(2):
                ckeys=torch.where(labels==c)[0]
                h=old[row][old[row]>=0].long();h=h[labels[h]==c]
                rep=reps[c]
                lr=(prior[row,ckeys,None]*(y[ckeys]-y[rep]).square()).sum()
                recovered=(prior[row,h,None]*(y[h]-y[rep]).square()).sum()
                torch.testing.assert_close(errors[row,c,2],lr-recovered,atol=2e-6,rtol=2e-5)

    def test_kernels_and_graph_replay(self):
        torch.manual_seed(34);device='cuda';g=9;d=8
        pop=torch.tensor([1,2,3,4,7,9,17,32,3,5,6,11,2],device=device,dtype=torch.float32)
        f=len(pop);n=int(pop.sum());labels=torch.repeat_interleave(torch.arange(f,device=device),pop.long()).contiguous()
        reps=(pop.long().cumsum(0)-pop.long()).contiguous();width=triton.next_power_of_2(int(pop.max()))
        members=torch.full((f,width),-1,device=device,dtype=torch.int32)
        for c in range(f):members[c,:int(pop[c])]=torch.where(labels==c)[0].int()
        prior=torch.rand(g,n,device=device)*.01;pi=reduce_cluster(prior,labels,f)
        ids=torch.stack([torch.randperm(n,device=device)[:n//3] for _ in range(g)]).int();ids[:,-2:]=-1
        y=torch.randn(n,d,device=device);sy=torch.randn(3,d,device=device);sp=torch.rand(g,3,device=device)*.1
        allowed=torch.rand(g,f,device=device)>.5
        sampled=torch.arange(g,device=device)%8==0
        phat=uniform_unobserved(prior,pi,ids,labels,pop,sampled)
        geom=dict(width=width,members=members,drop_bins=prepare(pop),loss_patch_values=y,
            loss_special_values=sy,loss_history_stride=8,shared_value_loss=True,loss_gate='observed')
        for observed in [False,True]:
            context=kernel.gate(prior,pi,ids,labels,reps,pop,observed)
            known=torch.zeros(g,n,device=device,dtype=torch.int32).scatter_add_(1,ids.long().clamp_min(0),(ids>=0).int())>0
            qualified=known if observed else known&((prior>pi[:,labels]/pop[labels])|torch.isin(torch.arange(n,device=device),reps)[None])
            self.assertTrue(torch.equal(context['qualified'],qualified))
            for operator,reference in [('group_history',group_history_losses),('residual_rep',transport_losses)]:
                geom['loss_operator']=operator;stats=kernel.statistics(prior,pi,labels,reps,pop,geom,context)
                from saf3r.triton import state_loss_sparse_statistics as fast
                fast.prepare(geom)
                sparse_stats=fast.statistics(prior,pi,labels,reps,pop,geom,context)
                torch.testing.assert_close(sparse_stats,stats,atol=2e-6,rtol=3e-4)
                torch.testing.assert_close(stats[:,:,:d],reduce_cluster(phat[:,:,None]*y[None],labels,f),atol=2e-6,rtol=2e-5)
                definitions,costs,*_=reference(phat,pi,y,sy,sp,labels,reps,pop,qualified,allowed)
                names=list(definitions) if operator=='group_history' else ['residual_energy','centered_residual_energy','transport_moment']
                for name in names:
                    geom['state_loss_variant']=name
                    errors,actual_costs,objective=kernel.state_losses(prior,pi,allowed,pop,reps,geom,stats,sp)
                    torch.testing.assert_close(errors,definitions[name],atol=3e-6,rtol=2e-4,msg=name)
                    torch.testing.assert_close(actual_costs,costs,msg=name)
                    torch.testing.assert_close(objective,errors-errors[:,:,1:2],equal_nan=True)
                # Every state, including duplicate Rep/History cases and invalid padding.
                states=torch.tensor([0,1,5,10],device=device).repeat((g*f+3)//4)[:g*f].reshape(g,f)
                pack_context=context
                if operator=='residual_rep':
                    union=qualified.clone();union[:,reps]=True
                    pack_context=dict(context,qualified=union,eligible_count=context['eligible_count']+(~context['observed_rep']).int())
                packed,real,_=rule.pack(states,None,reps,labels,pop,n,pack_context)
                weights,_=kernel.execution_weights(states,packed,prior,pi,labels,reps,pop,geom,pack_context,stats)
                for row in range(g):
                    selected=packed[row][packed[row]>=0].long();w=weights[row,:len(selected)]
                    self.assertEqual(len(selected),len(selected.unique()))
                    for c in range(f):
                        ix=labels[selected]==c;state=int(states[row,c]);keys=selected[ix]
                        if not len(keys):continue
                        if state==10:expected=torch.ones_like(w[ix])
                        elif state==1:expected=pi[row,c]/phat[row,keys]
                        elif operator=='group_history':expected=torch.ones_like(w[ix])*pi[row,c]/phat[row,keys].sum()
                        else:
                            expected=torch.ones_like(w[ix]);ordinary=keys!=reps[c]
                            expected[~ordinary]=(pi[row,c]-phat[row,keys[ordinary]].sum())/phat[row,reps[c]]
                        torch.testing.assert_close(w[ix],expected,atol=2e-5,rtol=2e-4)
        # Graph inputs include current Specials; constants refresh must replace sketches.
        from saf3r.patch.state_loss_controller import select_window
        z=torch.empty(0,device=device);cap=n//2
        for operator,name in [('group_history','group_centered'),('residual_rep','centered_residual_energy')]:
            geom.update(loss_operator=operator,state_loss_variant=name)
            window=Window(prior,pi,ids,allowed,z,reps,labels,pop,geom,cap,16,1.,bucket_bins=True,special_p=sp)
            for factor in [1.,1.7]:
                geom['loss_patch_values']=y*factor
                window.update_constants(z,reps,labels,pop,geom)
                actual=window(prior,pi,ids,allowed,special_p=sp*factor)
                expected=select_window(prior,pi,ids,allowed,reps,labels,pop,geom,cap,16,sp*factor)
                for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
