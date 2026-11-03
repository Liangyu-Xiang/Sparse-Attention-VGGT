"""CPU-only regression tests for the opt-in feasible-budget repair.

Run: python -m unittest discover -s tests -p 'test_lightweight_budget_repair.py'
"""
import importlib.util
import pathlib
import unittest

import torch

MODULE = pathlib.Path(__file__).resolve().parents[1] / "saf3r" / "patch" / "reuse_budget_repair.py"
SPEC = importlib.util.spec_from_file_location("reuse_budget_repair", MODULE)
repair_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair_module)
repair_underfilled = repair_module.repair_underfilled


class BudgetRepairTest(unittest.TestCase):
    def test_uses_slack_without_exceeding_budget_or_increasing_loss(self):
        # Drop / Rep / History / Exact with deliberately different costs.
        costs = torch.tensor([0.0, 1.0, 3.0, 5.0]).expand(2, 6, 4).clone()
        errors = torch.tensor([
            [10.0, 6.0, 3.0, 0.0],
            [7.0, 4.0, 2.0, 0.0],
            [11.0, 9.0, 4.0, 0.0],
            [4.0, 3.0, 2.0, 0.0],
            [9.0, 5.0, 2.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]).expand(2, 6, 4).clone()
        selected = torch.zeros((2, 6), dtype=torch.int64)
        for passes in (1, 2):
            corrected = repair_underfilled(errors, costs, selected, budget=12, passes=passes)
            self.assertTrue(torch.all(costs.gather(2, corrected[..., None]).sum(1) <= 12))
            self.assertTrue(torch.all(errors.gather(2, corrected[..., None]).sum(1) <
                                      errors.gather(2, selected[..., None]).sum(1)))
        self.assertTrue(torch.equal(selected, torch.zeros_like(selected)))

    def test_full_budget_selection_is_unchanged(self):
        costs = torch.tensor([[[0., 1., 2., 3.], [0., 1., 2., 3.]]])
        errors = torch.tensor([[[4., 3., 2., 0.], [4., 3., 2., 0.]]])
        selected = torch.tensor([[3, 3]])
        for passes in (1, 2):
            corrected = repair_underfilled(errors, costs, selected, budget=6, passes=passes)
            self.assertTrue(torch.equal(corrected, selected))

    def test_no_finite_improvement_is_unchanged(self):
        costs = torch.tensor([[[0., 1., 2., 3.]]])
        errors = torch.tensor([[[0., 2., float("inf"), 1.]]])
        selected = torch.tensor([[0]])
        for passes in (1, 2):
            self.assertTrue(torch.equal(repair_underfilled(errors, costs, selected, 3, passes=passes), selected))


if __name__ == "__main__":
    unittest.main()
