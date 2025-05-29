import random
import unittest
from unittest import TestCase
import mdp

expected_policy = {
    "1": "right",
    "2": "right",
    "3": "right",
    "4": "up",
    "5": "up",
    "6": "up",
    "7": "right",
    "8": "up",
    "9": "left",
    "10": None,
    "11": None,
}


class TestMDP(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.mdp = mdp.MDP(map_file="rnGraph.actions", error=0.01)

    def test_init_no_map_file(self):
        m = mdp.MDP(map_file=None)
        self.assertEqual(len(m.goals), 0)
        self.assertEqual(len(m.transition_probs), 0)
        self.assertEqual(len(m.states), 0)
        self.assertEqual(len(m.actions), 0)

    def test_init_with_map_file(self):
        self.assertTrue(len(self.mdp.goals) > 0)
        self.assertTrue(
            len(self.mdp.transition_probs) > 0,
        )
        self.assertTrue(len(self.mdp.states) > 0)
        self.assertTrue(len(self.mdp.actions) > 0)

    def test_repr(self):
        r = repr(self.mdp)
        self.assertIn("Gamma:", r)
        self.assertIn("Error:", r)
        self.assertIn("Reward:", r)
        self.assertIn("Goals:", r)
        self.assertIn("Transitions:", r)

    def test_compute_eu_goal_state(self):
        if self.mdp.goals:
            goal_state, goal_utility = self.mdp.goals[0]
            action, eu = self.mdp.compute_eu(goal_state)
            self.assertIsNone(action)
            self.assertEqual(eu, goal_utility)

    def test_compute_eu_normal_state(self):
        self.mdp.utilities["1"] = 0.1
        self.mdp.utilities["2"] = 0.5
        self.mdp.utilities["3"] = 0.9

        test_state = "1"

        best_action, best_eu = self.mdp.compute_eu(test_state)
        self.assertIsNotNone(best_action)
        self.assertGreaterEqual(best_eu, 0.1)

    def test_compute_policy(self):
        self.mdp.utilities["1"] = 0.23
        self.mdp.utilities["2"] = 0.45
        self.mdp.utilities["3"] = 0.68
        self.mdp.utilities["4"] = 0.06
        self.mdp.utilities["5"] = 0.33
        self.mdp.utilities["6"] = -0.03
        self.mdp.utilities["7"] = -0.01
        self.mdp.utilities["8"] = 0.13
        self.mdp.utilities["9"] = -0.07
        policy = self.mdp.compute_policy()
        self.assertEqual(policy, expected_policy)

    def test_simulate_value_iteration(self):
        self.mdp.value_iteration()
        outcomes = self.mdp.simulate_policy("1", 100)

        positive_goals = {s for s, u in self.mdp.goals if float(u) > 0}
        negative_goals = {s for s, u in self.mdp.goals if float(u) <= 0}
        count_pos = sum(1 for state in outcomes if state in positive_goals)
        count_neg = sum(1 for state in outcomes if state in negative_goals)
        self.assertGreater(count_pos, count_neg)

    def test_simulate_policy_iteration(self):
        self.mdp.policy_iteration()
        outcomes = self.mdp.simulate_policy("1", 100)
        positive_goals = {s for s, u in self.mdp.goals if float(u) > 0}
        negative_goals = {s for s, u in self.mdp.goals if float(u) <= 0}
        count_pos = sum(1 for state in outcomes if state in positive_goals)
        count_neg = sum(1 for state in outcomes if state in negative_goals)
        self.assertGreater(count_pos, count_neg)


if __name__ == "__main__":
    unittest.main()
