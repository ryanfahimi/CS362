import unittest
import pandas as pd
from MyNaiveBayes import MyNaiveBayes


class TestMyNaiveBayes(unittest.TestCase):

    def setUp(self):
        # A simple 2-row dataset for testing __init__, fit, predict, and score.
        self.simple_data = pd.DataFrame(
            [
                [
                    "recurrence-events",
                    "young",
                    "premeno",
                    "small",
                ],
                [
                    "no-recurrence-events",
                    "old",
                    "ge40",
                    "large",
                ],
            ],
            columns=[
                "Class",
                "Age",
                "Menopause",
                "Tumor Size",
            ],
        )
        self.nb = MyNaiveBayes("test_filename", data=self.simple_data)

    def test_init_with_data(self):
        self.assertTrue(self.nb.data.equals(self.simple_data))
        self.assertEqual(self.nb.filename, "test_filename")

    def test_fit_probabilities(self):
        self.nb.fit()
        for attribute in self.simple_data.columns[1:]:
            pos_total = sum(self.nb.pos_examples[attribute].values())
            neg_total = sum(self.nb.neg_examples[attribute].values())
            self.assertAlmostEqual(
                pos_total,
                1.0,
                places=5,
            )
            self.assertAlmostEqual(
                neg_total,
                1.0,
                places=5,
            )

    def test_fit_specific_values(self):
        self.nb.fit()
        self.assertEqual(self.nb.pos_examples["Age"], {"young": 1.0})
        self.assertEqual(self.nb.neg_examples["Age"], {"old": 1.0})

    def test_predict(self):
        self.nb.fit()
        predictions = self.nb.predict(self.simple_data)
        self.assertEqual(predictions[0], "recurrence-events")
        self.assertEqual(predictions[1], "no-recurrence-events")

    def test_score_perfect_match(self):
        perfect_predictions = ["recurrence-events", "no-recurrence-events"]
        score = self.nb.score(perfect_predictions, self.simple_data["Class"])
        self.assertEqual(score, 1.0)

    def test_score_mismatch(self):
        mispredictions = ["no-recurrence-events", "no-recurrence-events"]
        score = self.nb.score(mispredictions, self.simple_data["Class"])
        self.assertEqual(score, 0.0)

    def test_score_partial_match(self):
        half_predictions = ["recurrence-events", "recurrence-events"]
        score = self.nb.score(half_predictions, self.simple_data["Class"])
        self.assertAlmostEqual(score, 2 / 3, 5)


if __name__ == "__main__":
    unittest.main()
