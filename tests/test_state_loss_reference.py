import unittest
import torch
from saf3r.patch.state_loss_reference import losses,transport_losses,uniform_unobserved,group_history_losses


class StateLossTests(unittest.TestCase):
    def sample(self):
        torch.manual_seed(4)
        y=torch.randn(5,3);special_y=torch.randn(1,3)
        probability=torch.tensor([[.04,.16,.20,.10,.30]])
        special_p=torch.tensor([[.20]])
        labels=torch.zeros(5,dtype=torch.long);reps=torch.tensor([0]);population=torch.tensor([5.])
        qualified=torch.tensor([[True,False,True,False,False]])
        return probability,probability.sum(1,keepdim=True),y,special_y,special_p,labels,reps,population,qualified,torch.ones(1,1,dtype=torch.bool)

    def test_same_support_same_weights_gives_same_loss(self):
        args=list(self.sample());args[-2]=torch.tensor([[True,False,False,False,False]])
        definitions,costs,_=losses(*args,'unit')
        for name,e in definitions.items():torch.testing.assert_close(e[:,:,1],e[:,:,2],msg=name)
        torch.testing.assert_close(costs[:,:,1],costs[:,:,2])

    def test_centered_moment_matches_actual_normalized_operator(self):
        args=self.sample();p,pi,y,sy,sp,labels,reps,pop,qualified,allowed=args
        for policy in ['unit','rep_population']:
            definitions,_,target=losses(*args,policy)
            for state,indices in enumerate([[],[0],[0,2],list(range(5))]):
                ix=torch.tensor(indices,dtype=torch.long)
                w=torch.ones(len(ix))
                if state==1 and policy=='rep_population':w[:]=5.
                mass=torch.cat((p[0,ix]*w,sp[0]))
                values=torch.cat((y[ix],sy))
                output=(mass[:,None]*values).sum(0)/mass.sum()
                residual=(output-target[0])*mass.sum()
                torch.testing.assert_close(definitions['centered_moment'][0,0,state],residual.square().sum(),atol=2e-6,rtol=2e-5)

    def test_transport_history_rep_duplicate_is_identical(self):
        args=list(self.sample());args[-2]=torch.tensor([[True,False,False,False,False]])
        definitions,costs,_,_=transport_losses(*args)
        for name,e in definitions.items():torch.testing.assert_close(e[:,:,1],e[:,:,2],msg=name)
        torch.testing.assert_close(costs[:,:,1],costs[:,:,2])

    def test_uniform_unknown_preserves_sampled_evidence_and_cluster_mass(self):
        p,pi,_,_,_,labels,_,pop,_,_=self.sample()
        old=torch.tensor([[0,2,-1]])
        q=uniform_unobserved(p,pi,old,labels,pop,torch.tensor([True]))
        torch.testing.assert_close(q[:,[0,2]],p[:,[0,2]])
        torch.testing.assert_close(q[:,[1,3,4]],torch.full((1,3),float((pi-p[:,0]-p[:,2])/3)))
        torch.testing.assert_close(q.sum(1),pi[:,0])
        no_history=uniform_unobserved(p,pi,old,labels,pop,torch.tensor([False]))
        torch.testing.assert_close(no_history,pi/5*torch.ones_like(p))

    def test_all_definitions_exact_state_has_zero_error(self):
        args=self.sample()
        for policy in ['unit','rep_population']:
            definitions,_,_=losses(*args,policy)
            for e in definitions.values():self.assertEqual(float(e[:,:,3]),0.)
        definitions,_,_,_=transport_losses(*args)
        for e in definitions.values():self.assertEqual(float(e[:,:,3]),0.)

    def test_group_history_single_rep_and_full_support(self):
        args=list(self.sample());args[-2]=torch.tensor([[True,False,False,False,False]])
        definitions,costs,_=group_history_losses(*args)
        for name,e in definitions.items():torch.testing.assert_close(e[:,:,1],e[:,:,2],msg=name)
        torch.testing.assert_close(costs[:,:,1],costs[:,:,2])
        args[-2]=torch.ones_like(args[-2])
        definitions,_,_=group_history_losses(*args)
        for name,e in definitions.items():torch.testing.assert_close(e[:,:,2],e[:,:,3],atol=1e-6,rtol=0,msg=name)

    def test_group_moment_matches_actual_mass_calibration(self):
        args=self.sample();p,pi,y,sy,sp,labels,reps,pop,qualified,allowed=args
        definitions,_,_=group_history_losses(*args)
        target=(p@y+sp@sy)[0]
        for state,ix in [(1,torch.tensor([0])),(2,torch.tensor([0,2]))]:
            weight=pi[0,0]/p[0,ix].sum()
            output=(p[0,ix,None]*weight*y[ix]).sum(0)+(sp@sy)[0]
            torch.testing.assert_close(definitions['group_moment'][0,0,state],(target-output).square().sum())

    def test_separable_loss_can_prefer_wrong_mask_even_with_uniform_truth(self):
        # Equal logits for all five tokens: no unknown QK is needed here.
        p=torch.full((1,4),.2);pi=torch.full((1,2),.4)
        y=torch.tensor([[0.],[2.],[0.],[-2.]]);sy=torch.tensor([[1.]]);sp=torch.tensor([[.2]])
        labels=torch.tensor([0,0,1,1]);reps=torch.tensor([0,2]);pop=torch.tensor([2.,2.])
        q=torch.tensor([[True,False,True,False]]);allow=torch.ones(1,2,dtype=torch.bool)
        definitions,_,_=group_history_losses(p,pi,y,sy,sp,labels,reps,pop,q,allow)
        dense=(p@y+sp@sy)[0,0]
        two_reps=(p[:,reps]*2@y[reps]+sp@sy)[0,0]
        exact_drop=(p[:,:2]@y[:2]+sp@sy)[0,0]/(p[:,:2].sum()+sp.sum())
        torch.testing.assert_close(two_reps,dense)
        self.assertGreater(float((exact_drop-dense).abs()),.79)
        for name in ['group_centered','group_moment']:
            e=definitions[name]
            self.assertLess(float(e[0,0,3]+e[0,1,0]),float(e[0,:,1].sum()),msg=name)
        from saf3r.patch.joint_state_loss_reference import refine
        _,costs,_=group_history_losses(p,pi,y,sy,sp,labels,reps,pop,q,allow)
        selected,before,after=refine(torch.tensor([[3,0]]),costs,definitions['group_centered'],p,pi,y,sy,sp,labels,reps,q,2,iterations=8)
        self.assertLess(float(after),float(before))
        self.assertLess(float(after),1e-12)
        self.assertLessEqual(float(costs.gather(2,selected[:,:,None]).sum()),2.)

    def test_separable_misranking_also_holds_at_twenty_percent_with_five_specials(self):
        # Exactly 35 tokens, five always-kept Specials, C=B=two clusters.
        p=torch.full((1,30),1/35);pi=torch.tensor([[2/35,28/35]])
        y=torch.zeros(30,1);y[1]=2.;y[3]=-2.
        sy=torch.zeros(5,1);sy[0]=1.;sp=torch.full((1,5),1/35)
        labels=torch.tensor([0,0]+[1]*28);reps=torch.tensor([0,2]);pop=torch.tensor([2.,28.])
        qualified=torch.zeros(1,30,dtype=torch.bool);qualified[:,reps]=True
        definitions,_,_=group_history_losses(p,pi,y,sy,sp,labels,reps,pop,qualified,torch.ones(1,2,dtype=torch.bool))
        dense=(p@y+sp@sy)[0,0];rep_rep=(p[:,reps]*pop@y[reps]+sp@sy)[0,0]
        exact_drop=(p[:,:2]@y[:2]+sp@sy)[0,0]/(p[:,:2].sum()+sp.sum())
        torch.testing.assert_close(rep_rep,dense)
        self.assertAlmostEqual((2+5)/35,.2)
        self.assertAlmostEqual(float(exact_drop-dense),.4,places=6)
        for name in ['group_centered','group_moment']:
            e=definitions[name];self.assertLess(float(e[0,0,3]+e[0,1,0]),float(e[0,:,1].sum()),msg=name)


if __name__=='__main__':unittest.main()
