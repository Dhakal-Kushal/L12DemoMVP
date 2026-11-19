import unittest

import torch

from pipeline.stage1 import OBSCURED, VISIBLE, assess_pair


class FixedScores:
    def __init__(self, before, after):
        self.rows = (before, after)

    def scores(self, images, prompts):
        return torch.tensor([[row.get(p, 0) for p in prompts] for row in self.rows])


class Stage1GateTests(unittest.TestCase):
    def verdicts(self, before, after):
        decisions = assess_pair(FixedScores(before, after), [None, None], "roof", 2.0)
        return [ok for ok, _ in decisions]

    def test_clear_damage_after_can_pass_without_target(self):
        self.assertEqual(self.verdicts({"roof": 5}, {"roof": -8, VISIBLE[1]: 6}), [True, True])

    def test_cloud_margin_applies_to_each_image(self):
        for cloudy in (0, 1):
            rows = [{"roof": 5}, {"roof": 5}]
            rows[cloudy][OBSCURED] = 2
            self.assertFalse(self.verdicts(*rows)[cloudy])

    def test_uncertain_cloud_passes_with_reason(self):
        decisions = assess_pair(FixedScores({"roof": 5}, {OBSCURED: 1}), [], "roof", 2.0)
        self.assertTrue(decisions[1][0])
        self.assertIn("uncertain", decisions[1][1])

    def test_irrelevant_before_rejects_pair(self):
        self.assertEqual(self.verdicts({"open ocean or empty water with no structures": 8},
                                       {VISIBLE[0]: 5}), [False, True])
