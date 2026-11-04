"""Causal checks for representation mass and unchanged fixed Key budget."""
import unittest
import torch
from saf3r.patch.shared_value_repair import execution_weights


class ExecutionCalibrationTests(unittest.TestCase):
    def inputs(self):
        labels=torch.tensor([0,0,1,1,1,2,2,3])
        reps=torch.tensor([0,2,5,7])
        pop=torch.bincount(labels).float()
        prior=torch.tensor([[.05,.15,.10,.20,.10,.10,.20,.10]])
        pi=torch.zeros(1,4).scatter_add_(1,labels[None],prior)
        selected=torch.tensor([[1,5,10,0]])
        ids=torch.tensor([[0,2,5,6,-1]])
        return selected,ids,reps,labels,pop,prior,pi

    def test_only_rep_compensation_preserves_history_and_exact_semantics(self):
        weights=execution_weights('rep_population',*self.inputs())
        torch.testing.assert_close(weights,torch.tensor([[2.,1.,1.,1.,0.]]))

    def test_count_calibration_preserves_every_covered_cluster_population(self):
        x=self.inputs();weights=execution_weights('cluster_population',*x)
        labels=x[3][x[1].clamp_min(0)]
        total=torch.zeros_like(x[-1]).scatter_add_(1,labels,weights)
        torch.testing.assert_close(total,torch.tensor([[2.,3.,2.,0.]]))
        torch.testing.assert_close(weights,torch.tensor([[2.,3.,1.,1.,0.]]))

    def test_prior_calibration_preserves_predicted_covered_cluster_mass(self):
        x=self.inputs();weights=execution_weights('cluster_prior',*x)
        own=x[-2].gather(1,x[1].clamp_min(0))
        labels=x[3][x[1].clamp_min(0)]
        total=torch.zeros_like(x[-1]).scatter_add_(1,labels,weights*own)
        torch.testing.assert_close(total,torch.tensor([[.2,.4,.3,0.]]))

    def test_population_bias_is_exact_for_identical_cluster_members(self):
        # Two identical patch members and one special: a Rep has to carry
        # population 2 to recover the same softmax denominator and output.
        dense=torch.tensor([0.,0.,0.]).softmax(0)@torch.tensor([3.,3.,0.])
        unit=torch.tensor([0.,0.]).softmax(0)@torch.tensor([3.,0.])
        calibrated=torch.tensor([2.,1.]).log().softmax(0)@torch.tensor([3.,0.])
        torch.testing.assert_close(calibrated,dense)
        self.assertGreater(float((unit-dense).abs()),.4)

    def test_unit_mode_is_exactly_original_and_does_not_mutate_support(self):
        x=self.inputs();before=x[1].clone()
        weights=execution_weights('unit',*x)
        self.assertTrue(torch.equal(weights,(x[1]>=0).float()))
        self.assertTrue(torch.equal(x[1],before))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_fused_weights_equal_cpu_reference_and_graph_replay(self):
        from saf3r.triton.shared_value_repair import ratio_table
        n=torch.arange(513,device='cuda',dtype=torch.float32)
        self.assertTrue(torch.equal(ratio_table(n.device),n[:,None]/n.clamp_min(1)[None,:]))
        x=self.inputs()
        expected=execution_weights('cluster_population',*x)
        device=[v.cuda() for v in x]
        eligible=torch.tensor([[0,1,0,0]],device='cuda',dtype=torch.int32)
        def run():return execution_weights('cluster_population',*device,eligible_count=eligible)
        torch.testing.assert_close(run().cpu(),expected)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=run()
        graph.replay()
        torch.testing.assert_close(result.cpu(),expected)


if __name__=='__main__':
    unittest.main()
